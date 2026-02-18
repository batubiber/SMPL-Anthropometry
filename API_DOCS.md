# Body Measurement API Documentation

REST API that extracts 16 body measurements from photos using **HMR 2.0 + SMPL-Anthropometry**.
Returns measurements in centimeters, quality metadata, and an annotated 3D body model (GLB).

---

## Base URL

| Environment | URL |
|---|---|
| Docker | `https://postpupillary-governmentally-glenna.ngrok-free.dev/api/measure` (host 8001 → container 8000) |
| Local dev | `http://localhost:8000` |

---

## Endpoints

### `GET /health`

Check if the API and models are ready.

```bash
curl https://postpupillary-governmentally-glenna.ngrok-free.dev/api/measure/health
```

**Response:**
```json
{
  "status": "ready",
  "gpu": true,
  "gpu_name": "NVIDIA GeForce RTX 3070"
}
```

| Field | Description |
|---|---|
| `status` | `"ready"` or `"loading"` (models still initializing) |
| `gpu` | Whether CUDA GPU is available |
| `gpu_name` | GPU device name, or `null` if CPU-only |

---

### `POST /api/measure`

Extract body measurements from 1 or 2 photos.

#### Request

**Content-Type:** `multipart/form-data`

| Field | Type | Required | Default | Description |
|---|---|---|---|---|
| `images` | File(s) | Yes | — | 1 or 2 photos (JPEG, PNG, WebP, HEIC). Best accuracy: front + 90-degree side view. |
| `gender` | String | No | `NEUTRAL` | `MALE`, `FEMALE`, or `NEUTRAL` |
| `height` | Float | No | `null` | Known height in cm (50–250). Strongly recommended for accuracy. |

#### Image Requirements

| Constraint | Value |
|---|---|
| Min resolution | 480 x 640 px (or 640 x 480 landscape) |
| Max file size | 10 MB per image |
| Max aspect ratio | 3:1 |
| Accepted formats | JPEG, PNG, WebP, HEIC |

#### Examples

**2 images — best accuracy (front + side):**
```bash
curl -X POST \
  -F "images=@front.jpeg" \
  -F "images=@side.jpeg" \
  -F "gender=MALE" \
  -F "height=180" \
  https://postpupillary-governmentally-glenna.ngrok-free.dev/api/measure
```

**1 image:**
```bash
curl -X POST \
  -F "images=@photo.jpeg" \
  -F "gender=FEMALE" \
  -F "height=165" \
  https://postpupillary-governmentally-glenna.ngrok-free.dev/api/measure
```

**Save the 3D GLB model to a file:**
```bash
curl -s -X POST \
  -F "images=@front.jpeg" \
  -F "images=@side.jpeg" \
  -F "gender=MALE" \
  -F "height=180" \
  https://postpupillary-governmentally-glenna.ngrok-free.dev/api/measure -o response.json

python -c "
import json, base64
data = json.load(open('response.json'))
glb = base64.b64decode(data['model_glb'])
open('body_model.glb', 'wb').write(glb)
print(f'Saved {len(glb):,} bytes')
"
```

#### Success Response (200)

```json
{
  "success": true,
  "measurements": [
    {"name": "height", "label": "P", "value_cm": 180.0, "type": "length"},
    {"name": "chest circumference", "label": "D", "value_cm": 103.3, "type": "circumference"},
    {"name": "waist circumference", "label": "E", "value_cm": 92.8, "type": "circumference"},
    {"name": "hip circumference", "label": "F", "value_cm": 101.3, "type": "circumference"},
    {"name": "shoulder breadth", "label": "O", "value_cm": 37.4, "type": "length"},
    {"name": "arm right length", "label": "J", "value_cm": 55.3, "type": "length"},
    {"name": "inside leg height", "label": "K", "value_cm": 76.2, "type": "length"}
  ],
  "raw_measurements": {
    "height": 180.0,
    "chest circumference": 103.3,
    "waist circumference": 92.8
  },
  "betas": [-0.002, 0.301, -0.227, 0.109, 0.023, -0.021, -0.070, 0.099, 0.082, -0.080],
  "images_processed": 2,
  "inference_time_sec": 7.36,
  "height_normalized": true,
  "gender": "MALE",
  "quality": {
    "detection_confidence": 0.999,
    "view_angle_diff": 74.1,
    "view_quality": "good",
    "height_ratio": 1.0,
    "warnings": []
  },
  "model_glb": "<base64-encoded GLB>"
}
```

#### Response Fields

| Field | Type | Description |
|---|---|---|
| `success` | bool | `true` if measurements were computed |
| `measurements` | array | Structured list of 16 measurements (see table below) |
| `raw_measurements` | object | Flat `{name: value}` dict for backward compatibility |
| `betas` | array | 10 SMPL shape parameters |
| `images_processed` | int | Number of successfully processed images |
| `inference_time_sec` | float | Total processing time in seconds |
| `height_normalized` | bool | Whether height normalization was applied |
| `gender` | string | Gender model used |
| `quality` | object | Detection and view quality metadata |
| `model_glb` | string | Base64-encoded GLB 3D model with colored measurement visualization |

#### 16 Measurements

| Label | Name | Type |
|---|---|---|
| **A** | head circumference | circumference |
| **B** | neck circumference | circumference |
| **C** | shoulder to crotch height | length |
| **D** | chest circumference | circumference |
| **E** | waist circumference | circumference |
| **F** | hip circumference | circumference |
| **G** | wrist right circumference | circumference |
| **H** | bicep right circumference | circumference |
| **I** | forearm right circumference | circumference |
| **J** | arm right length | length |
| **K** | inside leg height | length |
| **L** | thigh left circumference | circumference |
| **M** | calf left circumference | circumference |
| **N** | ankle left circumference | circumference |
| **O** | shoulder breadth | length |
| **P** | height | length |

#### Quality Object

| Field | Type | Description |
|---|---|---|
| `detection_confidence` | float | Person detection score (0–1). Above 0.8 is reliable. |
| `view_angle_diff` | float \| null | Angle difference between 2 photos in degrees. ~90 is ideal. `null` for single image. |
| `view_quality` | string | `"good"` (>=60 deg), `"fair"` (30–60 deg or single image), `"poor"` (<30 deg) |
| `height_ratio` | float \| null | `known_height / model_height`. Near 1.0 = normal. Outside 0.7–1.3 triggers warning. |
| `warnings` | array | List of warning strings. Empty = everything looks good. |

#### 3D Model (GLB)

The `model_glb` field contains a base64-encoded [GLB](https://www.khronos.org/gltf/) 3D scene with:

- **Body mesh** — semi-transparent skin-colored SMPL body
- **Measurement tubes** — colored rings (circumferences) and lines (lengths) on the body

Each measurement label (A–P) has a distinct color for easy identification. The model is scaled to the user's height if provided.

To view: decode base64, save as `.glb`, open in any 3D viewer (e.g., [glTF Viewer](https://gltf-viewer.donmccurdy.com/)).

#### Error Response

```json
{
  "success": false,
  "error": "No person detected in any photo.",
  "error_code": "NO_PERSON_DETECTED"
}
```

#### Error Codes

| Code | HTTP | Description |
|---|---|---|
| `MODEL_NOT_READY` | 503 | Models still loading at startup (~60–90s) |
| `INVALID_GENDER` | 400 | Gender not `MALE`/`FEMALE`/`NEUTRAL` |
| `NO_IMAGES` | 400 | No images uploaded |
| `TOO_MANY_IMAGES` | 400 | More than 2 images |
| `HEIGHT_OUT_OF_RANGE` | 400 | Height not between 50–250 cm |
| `UNSUPPORTED_FORMAT` | 400 | Not JPEG/PNG/WebP/HEIC |
| `IMAGE_TOO_LARGE` | 400 | Exceeds 10 MB |
| `IMAGE_TOO_SMALL` | 400 | Below 480x640 resolution |
| `IMAGE_BAD_ASPECT` | 400 | Aspect ratio > 3:1 |
| `NO_PERSON_DETECTED` | 422 | No person found in any image |
| `REQUEST_TIMEOUT` | 408 | Processing exceeded 30s timeout |

---

## React Native Integration

```javascript
const measureBody = async (frontUri, sideUri, gender, heightCm) => {
  const formData = new FormData();

  formData.append('images', {
    uri: frontUri,
    type: 'image/jpeg',
    name: 'front.jpg',
  });

  if (sideUri) {
    formData.append('images', {
      uri: sideUri,
      type: 'image/jpeg',
      name: 'side.jpg',
    });
  }

  formData.append('gender', gender);       // "MALE" | "FEMALE" | "NEUTRAL"
  formData.append('height', String(heightCm)); // e.g. "180"

  const response = await fetch('https://YOUR_SERVER/api/measure', {
    method: 'POST',
    body: formData,
    // Do NOT set Content-Type — fetch sets it with the correct boundary
  });

  const data = await response.json();

  if (!data.success) {
    throw new Error(`${data.error_code}: ${data.error}`);
  }

  return {
    measurements: data.measurements,       // [{name, label, value_cm, type}, ...]
    quality: data.quality,                 // {detection_confidence, view_quality, ...}
    glbBase64: data.model_glb,             // base64 GLB for 3D viewer
    inferenceTime: data.inference_time_sec,
  };
};

// Usage
const result = await measureBody(
  'file:///path/to/front.jpg',
  'file:///path/to/side.jpg',
  'MALE',
  180
);

console.log(result.measurements);
// [
//   {name: "height", label: "P", value_cm: 180.0, type: "length"},
//   {name: "chest circumference", label: "D", value_cm: 103.3, type: "circumference"},
//   ...
// ]

console.log(result.quality.view_quality); // "good"
```

---

## Docker Deployment

**Build:**
```bash
docker build -f docker/Dockerfile -t body-measurement-api .
```

**Run with docker-compose:**
```bash
docker compose up -d
```

**Environment Variables:**

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | auto | `cuda`, `cpu`, or empty for auto-detect |
| `API_HOST` | `0.0.0.0` | Bind address |
| `API_PORT` | `8000` | Container port |
| `CORS_ORIGINS` | `*` | Comma-separated allowed origins |
| `MAX_IMAGE_SIZE_MB` | `10` | Max upload size per image |
| `MIN_IMAGE_WIDTH` | `480` | Min image width |
| `MIN_IMAGE_HEIGHT` | `640` | Min image height |
| `REQUEST_TIMEOUT_SEC` | `30` | Max processing time |
| `MOCK_PYRENDER` | `false` | Set `true` to mock pyrender (auto on Windows) |

**Note:** The HMR 2.0 checkpoint (~2.7 GB) is mounted as a volume, not baked into the image:
```yaml
volumes:
  - ~/.cache/4DHumans:/root/.cache/4DHumans:ro
```

---

## Tips for Best Accuracy

1. **Use 2 photos** — front-facing + 90-degree side view (10–18% more accurate than single photo)
2. **Provide height** — dramatically improves all measurements (~20–30% more accurate)
3. **Good lighting** — even lighting, avoid harsh shadows
4. **Full body visible** — head to feet in frame, person occupies 30–80% of image
5. **Standing straight** — arms slightly away from body, feet shoulder-width apart
6. **Minimal clothing** — tight-fitting clothes give better shape estimation

---

## Pipeline

```
Photo(s) → ViTDet (person detection) → HMR 2.0 (SMPL shape estimation)
         → Beta averaging (if 2 images) → SMPL-Anthropometry (16 measurements)
         → Height normalization (if provided) → Annotated GLB + JSON response
```
