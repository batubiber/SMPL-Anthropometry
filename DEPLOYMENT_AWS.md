# AWS Deployment Guide — Body Measurement API

GPU-accelerated body measurement API using HMR 2.0 + SMPL-Anthropometry.

---

## Architecture Overview

```
Client (React Native)
        │
        ▼
   [ALB / HTTPS]
        │
        ▼
   EC2 GPU Instance
   ┌──────────────────┐
   │  Docker Container │
   │  ┌──────────────┐ │
   │  │  FastAPI      │ │
   │  │  :8000        │ │
   │  │  HMR 2.0      │ │
   │  │  ViTDet       │ │
   │  │  SMPL         │ │
   │  └──────────────┘ │
   │  NVIDIA GPU       │
   └──────────────────┘
```

---

## 1. Instance Selection

| Instance | GPU | VRAM | vCPU | RAM | On-Demand $/hr | Spot $/hr (est.) |
|----------|-----|------|------|-----|-----------------|------------------|
| **g4dn.xlarge** | T4 | 16 GB | 4 | 16 GB | ~$0.526 | ~$0.16 |
| g4dn.2xlarge | T4 | 16 GB | 8 | 32 GB | ~$0.752 | ~$0.23 |
| g5.xlarge | A10G | 24 GB | 4 | 16 GB | ~$1.006 | ~$0.30 |

**Recommended: `g4dn.xlarge`** — cheapest GPU instance, T4 has 16GB VRAM (more than enough, tested on RTX 3070 with 8GB). Single API worker doesn't need more than 4 vCPUs.

**AMI:** Deep Learning AMI (Ubuntu 22.04) — comes with NVIDIA drivers and Docker pre-installed.

---

## 2. Launch EC2 Instance

### 2a. Via AWS Console

1. Go to **EC2 > Launch Instance**
2. **Name:** `body-measurement-api`
3. **AMI:** Search for `Deep Learning AMI GPU PyTorch` (Ubuntu 22.04)
4. **Instance type:** `g4dn.xlarge`
5. **Key pair:** Create or select an existing SSH key
6. **Network:**
   - Create a Security Group with:
     - SSH (port 22) — your IP only
     - HTTP (port 80) — 0.0.0.0/0 (or your app's IP range)
     - HTTPS (port 443) — 0.0.0.0/0
     - Custom TCP (port 8000) — 0.0.0.0/0 (direct access, optional)
7. **Storage:** 100 GB gp3 (Docker image ~15GB + checkpoint ~3GB + OS)
8. Launch

### 2b. Via AWS CLI

```bash
aws ec2 run-instances \
  --image-id ami-0xxxxxxxxxxxxxxxx \
  --instance-type g4dn.xlarge \
  --key-name your-key-pair \
  --security-group-ids sg-xxxxxxxx \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=body-measurement-api}]' \
  --count 1
```

---

## 3. Server Setup

SSH into the instance:

```bash
ssh -i your-key.pem ubuntu@<EC2_PUBLIC_IP>
```

### 3a. Verify GPU and Docker

```bash
# Check GPU
nvidia-smi

# Check Docker + GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 nvidia-smi
```

If Docker is not installed (non-DLAMI):

```bash
# Install Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install NVIDIA Container Toolkit
distribution=$(. /etc/os-release; echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3b. Clone Repository

```bash
git clone https://github.com/YOUR_ORG/SMPL-Anthropometry.git
cd SMPL-Anthropometry
```

### 3c. Download HMR2 Checkpoint

The HMR2 checkpoint (~2.7GB) must be available at `~/.cache/4DHumans/`. It downloads automatically on first model load, but you can pre-download it:

```bash
mkdir -p ~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/
wget -O ~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt \
  "https://people.eecs.berkeley.edu/~jathushan/projects/4dhumans/hmr2_model/epoch=35-step=1000000.ckpt"
```

> If the download fails (403 Forbidden), copy the checkpoint from your local machine:
> ```bash
> scp -i your-key.pem ~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/epoch=35-step=1000000.ckpt \
>   ubuntu@<EC2_PUBLIC_IP>:~/.cache/4DHumans/logs/train/multiruns/hmr2/0/checkpoints/
> ```

---

## 4. Build and Run

### 4a. Create docker-compose.prod.yml

```yaml
services:
  body-measurement-api:
    build:
      context: .
      dockerfile: docker/Dockerfile
    ports:
      - "8000:8000"
    environment:
      - DEVICE=cuda
      - CORS_ORIGINS=https://yourdomain.com
      - MAX_IMAGE_SIZE_MB=10
      - REQUEST_TIMEOUT_SEC=30
      - MOCK_PYRENDER=true
    volumes:
      - ~/.cache/4DHumans:/root/.cache/4DHumans:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    restart: unless-stopped
    logging:
      driver: json-file
      options:
        max-size: "50m"
        max-file: "5"
```

### 4b. Build

```bash
docker compose -f docker-compose.prod.yml build
```

Build takes ~10-15 minutes (detectron2 compiles CUDA extensions).

### 4c. Start

```bash
docker compose -f docker-compose.prod.yml up -d
```

### 4d. Verify

```bash
# Watch startup logs (model loading takes ~60-90s)
docker compose -f docker-compose.prod.yml logs -f

# Wait for "[STARTUP] Ready! Accepting requests."

# Health check
curl http://localhost:8000/health
# Expected: {"status":"ready","gpu":true,"gpu_name":"Tesla T4"}

# Test measurement
curl -X POST -F "images=@test.jpg" -F "gender=MALE" -F "height=180" \
  http://localhost:8000/api/measure
```

---

## 5. HTTPS with Caddy (Recommended)

Use Caddy as a reverse proxy for automatic HTTPS via Let's Encrypt.

### 5a. Install Caddy

```bash
sudo apt install -y debian-keyring debian-archive-keyring apt-transport-https
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | sudo gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg
curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | sudo tee /etc/apt/sources.list.d/caddy-stable.list
sudo apt update && sudo apt install caddy
```

### 5b. Configure Caddyfile

```bash
sudo tee /etc/caddy/Caddyfile << 'EOF'
api.yourdomain.com {
    reverse_proxy localhost:8000

    # Max upload size (images)
    request_body {
        max_size 20MB
    }
}
EOF

sudo systemctl restart caddy
```

### 5c. DNS

Point `api.yourdomain.com` to your EC2 Elastic IP. Caddy automatically provisions the TLS certificate.

### 5d. Update CORS

```bash
# In docker-compose.prod.yml, update:
- CORS_ORIGINS=https://api.yourdomain.com
```

---

## 6. Alternative: Application Load Balancer (ALB)

For production with auto-scaling or multiple instances:

1. **Target Group:** Create a target group pointing to port 8000
   - Health check path: `/health`
   - Healthy threshold: 2
   - Interval: 30s
   - Timeout: 10s
2. **ALB:** Create an Application Load Balancer
   - Listener: HTTPS (443) with ACM certificate
   - Forward to target group
3. **Security Group:** Allow ALB to reach EC2 on port 8000

---

## 7. Environment Variables Reference

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `cuda` | `cuda` or `cpu` |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `MAX_IMAGE_SIZE_MB` | `10` | Max upload size per image |
| `MIN_IMAGE_WIDTH` | `480` | Min image width in pixels |
| `MIN_IMAGE_HEIGHT` | `640` | Min image height in pixels |
| `REQUEST_TIMEOUT_SEC` | `30` | Request timeout |
| `MOCK_PYRENDER` | `true` | Mock pyrender (always true in Docker) |

---

## 8. API Endpoints

### GET /health

```json
{"status": "ready", "gpu": true, "gpu_name": "Tesla T4"}
```

### POST /api/measure

**Parameters (multipart/form-data):**

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `images` | File[] | Yes | 1-2 photos (JPEG/PNG) |
| `gender` | string | No | `MALE` / `FEMALE` / `NEUTRAL` (default) |
| `height` | float | No | Known height in cm for normalization |

**Response:**

```json
{
  "success": true,
  "measurements": [
    {"name": "chest circumference", "label": "D", "value_cm": 104.1, "type": "circumference"},
    {"name": "waist circumference", "label": "E", "value_cm": 93.8, "type": "circumference"}
  ],
  "raw_measurements": {"chest circumference": 104.1, "waist circumference": 93.8},
  "betas": [0.067, 0.188, ...],
  "images_processed": 2,
  "inference_time_sec": 5.2,
  "height_normalized": true,
  "gender": "MALE",
  "quality": {
    "detection_confidence": 0.999,
    "view_angle_diff": 74.1,
    "view_quality": "good",
    "height_ratio": 1.0,
    "warnings": []
  }
}
```

**Measurement Labels:**

| Label | Measurement |
|-------|-------------|
| A | Head circumference |
| B | Neck circumference |
| C | Shoulder to crotch height |
| D | Chest circumference |
| E | Waist circumference |
| F | Hip circumference |
| G | Wrist right circumference |
| H | Bicep right circumference |
| I | Forearm right circumference |
| J | Arm right length |
| K | Inside leg height |
| L | Thigh left circumference |
| M | Calf left circumference |
| N | Ankle left circumference |
| O | Shoulder breadth |
| P | Height |

---

## 9. Monitoring

### CloudWatch Logs

```bash
# Install CloudWatch agent (optional)
sudo apt install -y amazon-cloudwatch-agent

# Or just use Docker log driver:
# In docker-compose.prod.yml, add:
logging:
  driver: awslogs
  options:
    awslogs-region: us-east-1
    awslogs-group: body-measurement-api
    awslogs-stream-prefix: api
```

### Health Check Script

```bash
#!/bin/bash
# /opt/healthcheck.sh — run via cron every minute
RESPONSE=$(curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health)
if [ "$RESPONSE" != "200" ]; then
  echo "$(date) - API unhealthy (HTTP $RESPONSE), restarting..."
  cd /home/ubuntu/SMPL-Anthropometry
  docker compose -f docker-compose.prod.yml restart
fi
```

```bash
# Add to crontab
echo "* * * * * /opt/healthcheck.sh >> /var/log/healthcheck.log 2>&1" | crontab -
```

### GPU Monitoring

```bash
# Check VRAM usage
docker exec $(docker ps -q) nvidia-smi

# Expected: ~6-7 GB out of 16 GB (T4)
```

---

## 10. Cost Optimization

### Spot Instances

For non-critical or dev workloads, use Spot instances to save ~70%:

```bash
aws ec2 run-instances \
  --instance-type g4dn.xlarge \
  --instance-market-options '{"MarketType":"spot","SpotOptions":{"MaxPrice":"0.25","SpotInstanceType":"persistent"}}' \
  ...
```

### Auto Stop/Start

If the API is only used during business hours:

```bash
# EventBridge rules to stop at 8 PM and start at 8 AM
aws events put-rule --name stop-gpu --schedule-expression "cron(0 20 * * ? *)"
aws events put-rule --name start-gpu --schedule-expression "cron(0 8 * * ? *)"
```

### Right-Sizing

- **g4dn.xlarge** ($0.526/hr = ~$380/month) — recommended for production
- Consider **Spot** for dev/staging (~$115/month)
- The API uses ~6-7 GB VRAM; T4's 16 GB leaves headroom

---

## 11. Security Checklist

- [ ] SSH access restricted to your IP only (Security Group)
- [ ] API port (8000) not exposed directly — use ALB or Caddy
- [ ] CORS_ORIGINS set to your specific domain (not `*`)
- [ ] HTTPS enabled via Caddy or ALB + ACM
- [ ] EC2 instance has IAM role with minimal permissions
- [ ] Docker container runs with `--read-only` if possible
- [ ] Regular OS updates: `sudo apt update && sudo apt upgrade`
- [ ] CloudWatch alarms for CPU/GPU utilization

---

## 12. Rebuilding After Code Changes

After modifying any Python file (`measurement_api.py`, `measure.py`, etc.):

```bash
# 1. Rebuild image (only changed layers re-run, usually <30 seconds)
docker compose build

# 2. Restart container with new image
docker compose up -d

# 3. Watch logs until ready (~2-3 min for model loading)
docker compose logs -f

# Wait for: [STARTUP] Ready! Accepting requests.
# Press Ctrl+C to stop watching logs (container keeps running)
```

**What triggers a slow vs fast rebuild:**

| Change | Rebuild Time |
|--------|-------------|
| Python files (`*.py`) | ~30 seconds (only COPY layer) |
| `docker/requirements.txt` | ~2-5 min (pip install layer) |
| `docker/Dockerfile` | ~10-15 min (full rebuild) |
| `data/smpl/` model files | ~1 min (COPY layer) |

**Useful commands:**

```bash
# Check container status
docker compose ps

# View live logs
docker compose logs -f

# Restart without rebuild (same image)
docker compose restart

# Full stop + remove
docker compose down

# Stop, rebuild, and start in one line
docker compose down && docker compose build && docker compose up -d
```

---

## 13. Quick Start (Copy-Paste)

```bash
# 1. SSH into your g4dn.xlarge instance
ssh -i your-key.pem ubuntu@<IP>

# 2. Clone and enter repo
git clone https://github.com/YOUR_ORG/SMPL-Anthropometry.git
cd SMPL-Anthropometry

# 3. Copy checkpoint from local machine (run on your local machine)
# scp -i your-key.pem -r ~/.cache/4DHumans ubuntu@<IP>:~/.cache/

# 4. Build and start
docker compose -f docker-compose.prod.yml build    # ~10-15 min
docker compose -f docker-compose.prod.yml up -d     # starts in background

# 5. Watch logs until ready
docker compose -f docker-compose.prod.yml logs -f
# Wait for: [STARTUP] Ready! Accepting requests.

# 6. Test
curl http://localhost:8000/health
curl -X POST -F "images=@test.jpg" -F "gender=MALE" -F "height=180" http://localhost:8000/api/measure
```
