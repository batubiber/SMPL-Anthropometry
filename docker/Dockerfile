# ===========================================================================
# Body Measurement API — GPU-enabled Docker image
# HMR 2.0 + SMPL-Anthropometry
#
# Build from project root:
#   docker build -f docker/Dockerfile -t body-measurement-api .
# ===========================================================================
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HOME=/root

# System dependencies (includes build tools for detectron2 compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3-pip python3.10-venv \
    build-essential ninja-build \
    git wget libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3 \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA 11.8
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install detectron2 — needs torch at build time (--no-build-isolation)
# and nvcc from devel image for CUDA extensions
RUN pip install --no-cache-dir --no-build-isolation \
    'git+https://github.com/facebookresearch/detectron2.git'

WORKDIR /app

# Install Python dependencies first (cache-friendly layer)
COPY docker/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Install chumpy stub — needed by smplx for unpickling SMPL .pkl files.
# The real chumpy has build issues on modern pip; this stub provides the
# minimal Ch class needed for pickle deserialization.
# Ch is a plain object (NOT ndarray subclass) matching real chumpy's design.
COPY docker/chumpy_stub.py /app/chumpy_stub.py
RUN python -c "\
import os, shutil; \
d = '/usr/local/lib/python3.10/dist-packages/chumpy'; \
os.makedirs(d, exist_ok=True); \
shutil.copy('/app/chumpy_stub.py', os.path.join(d, '__init__.py')); \
os.makedirs(os.path.join(d, 'ch'), exist_ok=True); \
open(os.path.join(d, 'ch', '__init__.py'), 'w').write( \
  'from chumpy import Ch\nimport numpy as np\narray = np.array\n'); \
os.makedirs(os.path.join(d, 'utils'), exist_ok=True); \
open(os.path.join(d, 'utils', '__init__.py'), 'w').write(''); \
print('chumpy stub installed with ch + utils submodules') \
" && rm /app/chumpy_stub.py

# Copy 4D-Humans (HMR 2.0)
COPY 4D-Humans/ /app/4D-Humans/

# Copy application code
COPY measurement_api.py photo_to_measurements.py measure.py \
     measurement_definitions.py landmark_definitions.py \
     joint_definitions.py utils.py visualize.py \
     /app/

# Copy SMPL model data (API uses SMPL only, not SMPLX — saves ~1.8GB)
COPY data/smpl/ /app/data/smpl/

# HMR 2.0 checkpoint (~2.7GB) is mounted at runtime via docker-compose volume
# from host's ~/.cache/4DHumans/ to /root/.cache/4DHumans/
# This avoids re-downloading during build.

# Environment defaults
ENV DEVICE=cuda
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV MOCK_PYRENDER=true
ENV CORS_ORIGINS=*
ENV MAX_IMAGE_SIZE_MB=10
ENV REQUEST_TIMEOUT_SEC=30

EXPOSE 8000

# Health check with 120s start period for model loading
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# Single worker (model is not multiprocess-safe)
CMD ["python", "-m", "uvicorn", "measurement_api:app", \
     "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
