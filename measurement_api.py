"""
FastAPI Backend for Body Measurements
HMR 2.0 + SMPL-Anthropometry Pipeline

Usage:
    python -m uvicorn measurement_api:app --host 0.0.0.0 --port 8000

Then expose with ngrok:
    ngrok http 8000
"""

import os
import sys
import time
import io
import base64
import importlib.abc
import importlib.machinery

# Windows: set HOME before any hmr2 imports
if os.name == 'nt':
    os.environ.setdefault('HOME', os.path.expanduser('~'))

# ---------------------------------------------------------------------------
# Configuration from environment variables
# ---------------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
DEVICE_OVERRIDE = os.environ.get("DEVICE", "")  # "cuda", "cpu", or "" for auto
CORS_ORIGINS = os.environ.get("CORS_ORIGINS", "*").split(",")
MAX_IMAGE_SIZE_MB = float(os.environ.get("MAX_IMAGE_SIZE_MB", "10"))
MIN_IMAGE_WIDTH = int(os.environ.get("MIN_IMAGE_WIDTH", "480"))
MIN_IMAGE_HEIGHT = int(os.environ.get("MIN_IMAGE_HEIGHT", "640"))
REQUEST_TIMEOUT_SEC = float(os.environ.get("REQUEST_TIMEOUT_SEC", "30"))
MOCK_PYRENDER = os.environ.get("MOCK_PYRENDER", "").lower() == "true" or os.name == "nt"

# ---------------------------------------------------------------------------
# Mock pyrender to avoid OpenGL/EGL crash (Windows or explicit opt-in)
# ---------------------------------------------------------------------------
if MOCK_PYRENDER:
    class _PyrenderMockFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path, target=None):
            if fullname == 'pyrender' or fullname.startswith('pyrender.'):
                return importlib.machinery.ModuleSpec(fullname, _PyrenderMockLoader())
            return None

    class _PyrenderMockLoader(importlib.abc.Loader):
        def create_module(self, spec):
            class _Mod:
                OffscreenRenderer = type('OffscreenRenderer', (), {
                    '__init__': lambda s, *a, **k: None,
                    'render': lambda s, *a, **k: (
                        __import__('numpy').zeros((1,1,3), dtype=__import__('numpy').uint8),
                        __import__('numpy').zeros((1,1), dtype=__import__('numpy').float32)),
                    'delete': lambda s: None,
                })
                class Mesh:
                    @staticmethod
                    def from_trimesh(*a, **k): return None
                class Scene:
                    def __init__(s, *a, **k): pass
                    def add(s, *a, **k): return None
                class PerspectiveCamera:
                    def __init__(s, *a, **k): pass
                class DirectionalLight:
                    def __init__(s, *a, **k): pass
                class Node:
                    def __init__(s, *a, **k): pass
                RenderFlags = type('RenderFlags', (), {'RGBA': 1, 'DEPTH_ONLY': 2, 'FLAT': 4})()
                def __getattr__(self, name):
                    return lambda *a, **k: None
            return _Mod()
        def exec_module(self, module):
            pass

    sys.meta_path.insert(0, _PyrenderMockFinder())

# Add project root and 4D-Humans to sys.path
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)
_4dhumans_dir = os.path.join(_script_dir, '4D-Humans')
if os.path.isdir(_4dhumans_dir) and _4dhumans_dir not in sys.path:
    sys.path.insert(0, _4dhumans_dir)

import numpy as np
import torch
import cv2
from typing import Optional, List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from scipy.spatial.transform import Rotation

from photo_to_measurements import (
    load_hmr2_model,
    load_detector,
    detect_person,
    run_hmr2_inference,
    aggregate_betas,
    compute_measurements,
)
from measurement_definitions import STANDARD_LABELS, MEASUREMENT_TYPES, MeasurementType
from utils import filter_body_part_slices, convex_hull_from_3D_points

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Body Measurement API",
    description="Extract body measurements from photos using HMR 2.0 + SMPL-Anthropometry",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Allowed upload content types
ALLOWED_CONTENT_TYPES = {
    "image/jpeg", "image/png", "image/webp", "image/heic",
    # Some clients send these variants
    "image/jpg", "image/heif",
}

# ---------------------------------------------------------------------------
# Global state for loaded models
# ---------------------------------------------------------------------------
_state = {
    "model": None,
    "model_cfg": None,
    "detector": None,
    "device": None,
    "ready": False,
}


# ---------------------------------------------------------------------------
# Startup: load models once
# ---------------------------------------------------------------------------
@app.on_event("startup")
async def startup_load_models():
    print("\n[STARTUP] Loading models...")

    if DEVICE_OVERRIDE:
        device = torch.device(DEVICE_OVERRIDE)
    else:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if device.type == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  WARNING: CUDA not available, using CPU (slow)")

    _state["device"] = device

    print("  Loading HMR 2.0 model...")
    _state["model"], _state["model_cfg"] = load_hmr2_model(device)
    print("  HMR 2.0 loaded.")

    print("  Loading ViTDet person detector...")
    _state["detector"] = load_detector(device)
    print("  ViTDet loaded.")

    _state["ready"] = True
    print("[STARTUP] Ready! Accepting requests.\n")


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------
class MeasurementItem(BaseModel):
    name: str          # e.g. "waist circumference"
    name_tr: str       # e.g. "Bel cevresi"
    label: str         # e.g. "E"
    value_cm: float    # e.g. 82.3
    type: str          # "circumference" or "length"
    color: str         # hex color matching GLB visualization, e.g. "#0096FF"


class QualityInfo(BaseModel):
    detection_confidence: float           # average detection score
    view_angle_diff: Optional[float] = None  # yaw diff in degrees (if 2 images)
    view_quality: str                     # "good" / "fair" / "poor"
    height_ratio: Optional[float] = None
    warnings: List[str]


class MeasurementResponse(BaseModel):
    success: bool
    measurements: List[MeasurementItem]
    raw_measurements: dict               # backward-compat flat dict
    betas: List[float]
    images_processed: int
    inference_time_sec: float
    height_normalized: bool
    gender: str
    quality: QualityInfo
    model_glb: str                       # base64-encoded GLB 3D body model


class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    error_code: str


# ---------------------------------------------------------------------------
# Structured error helper
# ---------------------------------------------------------------------------
def _error_response(status_code: int, error: str, error_code: str):
    """Return a JSONResponse with structured error body."""
    return JSONResponse(
        status_code=status_code,
        content={"success": False, "error": error, "error_code": error_code},
    )


# ---------------------------------------------------------------------------
# Turkish measurement names + color legend for mobile app
# ---------------------------------------------------------------------------
_TURKISH_NAMES = {
    "head circumference": "Baş Çevresi",
    "neck circumference": "Boyun Çevresi",
    "shoulder to crotch height": "Omuz-Kasık Yüksekliği",
    "chest circumference": "Göğüs Çevresi",
    "waist circumference": "Bel Çevresi",
    "hip circumference": "Kalça Çevresi",
    "wrist right circumference": "Sağ Bilek Çevresi",
    "bicep right circumference": "Sağ Biceps Çevresi",
    "forearm right circumference": "Sağ Ön Kol Çevresi",
    "arm right length": "Sağ Kol Uzunluğu",
    "inside leg height": "İç Bacak Yüksekliği",
    "thigh left circumference": "Sol Uyluk Çevresi",
    "calf left circumference": "Sol Baldir Çevresi",
    "ankle left circumference": "Sol Ayak Bileği Çevresi",
    "shoulder breadth": "Omuz Genişliği",
    "height": "Boy Uzunluğu",
}

# ---------------------------------------------------------------------------
# Helper: build annotated 3D GLB with measurement visualization
# ---------------------------------------------------------------------------
# Distinct color per measurement label (A-P), RGBA
_LABEL_COLORS = {
    "A": [255, 50, 50, 255],     # red — head circumference
    "B": [255, 140, 0, 255],     # orange — neck circumference
    "C": [255, 220, 50, 255],    # yellow — shoulder to crotch
    "D": [50, 200, 50, 255],     # green — chest circumference
    "E": [0, 150, 255, 255],     # blue — waist circumference
    "F": [140, 50, 255, 255],    # purple — hip circumference
    "G": [255, 100, 150, 255],   # pink — wrist circumference
    "H": [0, 220, 200, 255],     # teal — bicep circumference
    "I": [180, 180, 0, 255],     # olive — forearm circumference
    "J": [255, 80, 180, 255],    # magenta — arm length
    "K": [100, 200, 255, 255],   # light blue — inside leg height
    "L": [200, 100, 50, 255],    # brown — thigh circumference
    "M": [50, 255, 150, 255],    # mint — calf circumference
    "N": [180, 100, 255, 255],   # lavender — ankle circumference
    "O": [255, 200, 100, 255],   # gold — shoulder breadth
    "P": [200, 200, 200, 255],   # silver — height
}


def _create_tube(p1, p2, radius, color_rgba):
    """Create a cylinder (tube) mesh between two 3D points."""
    import trimesh
    segment = np.array([p1, p2])
    length = np.linalg.norm(segment[1] - segment[0])
    if length < 1e-6:
        return None
    mid = (segment[0] + segment[1]) / 2
    direction = (segment[1] - segment[0]) / length

    cyl = trimesh.creation.cylinder(radius=radius, height=length, sections=8)
    # Align cylinder to the segment direction
    z_axis = np.array([0, 0, 1])
    rot_axis = np.cross(z_axis, direction)
    rot_axis_norm = np.linalg.norm(rot_axis)
    if rot_axis_norm > 1e-6:
        rot_axis = rot_axis / rot_axis_norm
        angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
        K = np.array([[0, -rot_axis[2], rot_axis[1]],
                      [rot_axis[2], 0, -rot_axis[0]],
                      [-rot_axis[1], rot_axis[0], 0]])
        R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = mid
        cyl.apply_transform(T)
    else:
        # Already aligned (or opposite)
        if np.dot(z_axis, direction) < 0:
            cyl.vertices[:, 2] *= -1
        cyl.vertices += mid

    cyl.visual.face_colors = color_rgba
    return cyl


def _build_annotated_glb(measurer, height_scale=None):
    """Build a GLB scene with body mesh + colored measurement tubes."""
    import trimesh

    verts = measurer.verts.copy()
    faces = measurer.faces
    joints = measurer.joints.copy()

    if height_scale and height_scale != 1.0:
        verts = verts * height_scale
        joints = joints * height_scale

    # Body mesh — skin color, semi-transparent
    body = trimesh.Trimesh(vertices=verts, faces=faces)
    body.visual.face_colors = [200, 170, 150, 160]
    scene = trimesh.Scene()
    scene.add_geometry(body, node_name="body")

    tube_radius = 0.004 * (height_scale or 1.0)  # ~4mm, scales with body

    # Reverse lookup: measurement name → label code
    name_to_label = {v: k for k, v in STANDARD_LABELS.items()}

    # --- CIRCUMFERENCE measurements: mesh-plane intersection → tube ring ---
    mesh_for_slice = trimesh.Trimesh(vertices=verts, faces=faces)

    for m_name, m_def in measurer.circumf_definitions.items():
        if m_name not in measurer.measurements:
            continue
        label = name_to_label.get(m_name)
        if not label:
            continue
        color = _LABEL_COLORS.get(label, [255, 255, 255, 255])

        try:
            circumf_landmarks = m_def["LANDMARKS"]
            circumf_landmark_indices = [measurer.landmarks[l] for l in circumf_landmarks]
            j1_name, j2_name = m_def["JOINTS"]
            j1, j2 = measurer.joint2ind[j1_name], measurer.joint2ind[j2_name]

            plane_origin = np.mean(verts[circumf_landmark_indices, :], axis=0)
            plane_normal = joints[j1, :] - joints[j2, :]

            slice_segments, sliced_faces = trimesh.intersections.mesh_plane(
                mesh_for_slice,
                plane_normal=plane_normal,
                plane_origin=plane_origin,
                return_faces=True,
            )

            slice_segments = filter_body_part_slices(
                slice_segments, sliced_faces, m_name,
                measurer.circumf_2_bodypart, measurer.face_segmentation,
            )

            hull_segments = convex_hull_from_3D_points(slice_segments)

            # Create tube segments along the hull ring
            tubes = []
            for i in range(hull_segments.shape[0]):
                tube = _create_tube(hull_segments[i, 0], hull_segments[i, 1],
                                    tube_radius, color)
                if tube:
                    tubes.append(tube)

            if tubes:
                ring = trimesh.util.concatenate(tubes)
                scene.add_geometry(ring, node_name=f"meas_{label}_{m_name}")
        except Exception:
            continue  # skip measurements that fail geometry

    # --- LENGTH measurements: tube between two landmarks ---
    for m_name, landmark_inds in measurer.length_definitions.items():
        if m_name not in measurer.measurements:
            continue
        label = name_to_label.get(m_name)
        if not label:
            continue
        color = _LABEL_COLORS.get(label, [255, 255, 255, 255])

        try:
            # Only handle standard 2-landmark lengths
            if len(landmark_inds) != 2:
                continue

            points = []
            for idx in landmark_inds:
                if isinstance(idx, tuple):
                    pt = np.mean(verts[list(idx), :], axis=0)
                else:
                    pt = verts[idx]
                points.append(pt)

            tube = _create_tube(points[0], points[1], tube_radius, color)
            if tube:
                scene.add_geometry(tube, node_name=f"meas_{label}_{m_name}")
        except Exception:
            continue

    return scene


# ---------------------------------------------------------------------------
# Helper: validate image before processing
# ---------------------------------------------------------------------------
def _validate_image(img_cv2: np.ndarray, contents_bytes: bytes, filename: str,
                    content_type: Optional[str]) -> Optional[dict]:
    """Validate image constraints. Returns error dict or None if valid.

    Error dict: {"error": str, "error_code": str, "status_code": int}
    """
    # Format check
    if content_type and content_type not in ALLOWED_CONTENT_TYPES:
        return {
            "error": f"Unsupported image format '{content_type}' for {filename}. "
                     f"Accepted: JPEG, PNG, WebP, HEIC.",
            "error_code": "UNSUPPORTED_FORMAT",
            "status_code": 400,
        }

    # File size check
    max_bytes = int(MAX_IMAGE_SIZE_MB * 1024 * 1024)
    if len(contents_bytes) > max_bytes:
        size_mb = len(contents_bytes) / (1024 * 1024)
        return {
            "error": f"Image {filename} is {size_mb:.1f} MB, maximum is {MAX_IMAGE_SIZE_MB} MB.",
            "error_code": "IMAGE_TOO_LARGE",
            "status_code": 400,
        }

    # Resolution check
    h, w = img_cv2.shape[:2]
    min_w = min(MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT)  # allow portrait or landscape
    min_h = min(MIN_IMAGE_WIDTH, MIN_IMAGE_HEIGHT)
    if w < min_w or h < min_h:
        return {
            "error": f"Image {filename} is {w}x{h}, minimum resolution is "
                     f"{MIN_IMAGE_WIDTH}x{MIN_IMAGE_HEIGHT} (or {MIN_IMAGE_HEIGHT}x{MIN_IMAGE_WIDTH} landscape).",
            "error_code": "IMAGE_TOO_SMALL",
            "status_code": 400,
        }

    # Aspect ratio check
    aspect = max(w, h) / max(min(w, h), 1)
    if aspect > 3.0:
        return {
            "error": f"Image {filename} has extreme aspect ratio ({aspect:.1f}:1). "
                     f"Please use a standard photo, not a cropped strip.",
            "error_code": "IMAGE_BAD_ASPECT",
            "status_code": 400,
        }

    return None


# ---------------------------------------------------------------------------
# Helper: read uploaded image into cv2 format + raw bytes
# ---------------------------------------------------------------------------
async def _read_upload(upload: UploadFile):
    """Read an UploadFile, return (img_cv2, contents_bytes) or raise ValueError."""
    contents = await upload.read()
    if len(contents) == 0:
        raise ValueError(f"Empty file: {upload.filename}")
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not decode image: {upload.filename}")
    return img, contents


# ---------------------------------------------------------------------------
# Helper: extract yaw angle from global orient axis-angle
# ---------------------------------------------------------------------------
def _get_yaw_angle(global_orient: torch.Tensor) -> float:
    """Extract yaw (Y-axis rotation) from axis-angle global orient (1,3)."""
    rotvec = global_orient.numpy().flatten()[:3]
    rot = Rotation.from_rotvec(rotvec)
    euler = rot.as_euler('YXZ', degrees=True)
    return float(euler[0])


def _angular_diff(a: float, b: float) -> float:
    """Compute absolute angular difference in degrees (0-180)."""
    diff = abs(a - b) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


# ---------------------------------------------------------------------------
# Helper: process one image through detect + HMR 2.0
# ---------------------------------------------------------------------------
def _process_image(img_cv2: np.ndarray, filename: str):
    """Detect person and run HMR 2.0.

    Returns:
        (betas, yaw_angle, detection_score, bbox_coverage) or
        (None, None, None, None) if detection fails.
    """
    model = _state["model"]
    model_cfg = _state["model_cfg"]
    detector = _state["detector"]
    device = _state["device"]

    # Detect person
    boxes, scores = detect_person(detector, img_cv2)
    if len(boxes) == 0:
        return None, None, None, None

    # Pick largest person if multiple
    if len(boxes) > 1:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = np.argmax(areas)
        boxes = boxes[idx:idx + 1]
        det_score = float(scores[idx])
    else:
        det_score = float(scores[0])

    # Compute bounding box coverage (fraction of image area)
    h_img, w_img = img_cv2.shape[:2]
    img_area = h_img * w_img
    box = boxes[0]
    bbox_area = (box[2] - box[0]) * (box[3] - box[1])
    bbox_coverage = bbox_area / img_area if img_area > 0 else 0.0

    # HMR 2.0 inference
    betas, global_orient = run_hmr2_inference(model, model_cfg, img_cv2, boxes, device)
    if betas is None:
        return None, None, None, None

    yaw = _get_yaw_angle(global_orient)

    return betas, yaw, det_score, float(bbox_coverage)


# ---------------------------------------------------------------------------
# Main endpoint
# ---------------------------------------------------------------------------
@app.post("/api/measure")
async def measure(
    images: List[UploadFile] = File(..., description="1 or 2 photos (JPEG/PNG)"),
    gender: str = Form("NEUTRAL", description="MALE / FEMALE / NEUTRAL"),
    height: Optional[float] = Form(None, description="Known height in cm (optional)"),
):
    """
    Extract body measurements from photos.

    - 1 photo: single-view estimation
    - 2 photos: front + side views, beta averaging for higher accuracy
    - If height is provided, measurements are normalized to known height
    """
    # Validate state
    if not _state["ready"]:
        return _error_response(503, "Models are still loading, please wait.", "MODEL_NOT_READY")

    # Validate gender
    gender = gender.upper()
    if gender not in ("MALE", "FEMALE", "NEUTRAL"):
        return _error_response(400, f"Invalid gender: {gender}. Must be MALE/FEMALE/NEUTRAL.", "INVALID_GENDER")

    # Validate image count
    if len(images) == 0:
        return _error_response(400, "At least 1 photo is required.", "NO_IMAGES")
    if len(images) > 2:
        return _error_response(400, "Maximum 2 photos allowed.", "TOO_MANY_IMAGES")

    # Validate height range (Step 2)
    if height is not None:
        if height < 50 or height > 250:
            return _error_response(
                400,
                f"Height {height} cm is out of valid range (50-250 cm). "
                f"Make sure height is in centimeters.",
                "HEIGHT_OUT_OF_RANGE",
            )

    start_time = time.time()
    all_betas = []
    all_yaws = []
    all_det_scores = []
    all_coverages = []
    warnings = []
    errors = []

    # Process each image
    for upload in images:
        # Read image
        try:
            img_cv2, contents_bytes = await _read_upload(upload)
        except ValueError as e:
            errors.append(str(e))
            continue

        # Validate image (Step 1)
        validation_err = _validate_image(
            img_cv2, contents_bytes, upload.filename or "image", upload.content_type
        )
        if validation_err is not None:
            return _error_response(
                validation_err["status_code"],
                validation_err["error"],
                validation_err["error_code"],
            )

        # Check for timeout
        elapsed = time.time() - start_time
        if elapsed > REQUEST_TIMEOUT_SEC:
            return _error_response(408, "Request timed out during processing.", "REQUEST_TIMEOUT")

        # Detect + infer
        betas, yaw, det_score, coverage = _process_image(img_cv2, upload.filename or "image")
        if betas is None:
            errors.append(f"No person detected: {upload.filename}")
            continue

        all_betas.append(betas)
        all_yaws.append(yaw)
        all_det_scores.append(det_score)
        all_coverages.append(coverage)

        # Coverage warnings
        if coverage < 0.10:
            warnings.append(f"Person is very small in {upload.filename} (<10% of image). Move closer for better accuracy.")
        elif coverage > 0.95:
            warnings.append(f"Person fills >95% of {upload.filename}. Some body parts may be cropped.")

    if len(all_betas) == 0:
        error_msg = "No person detected in any photo."
        if errors:
            error_msg += " Errors: " + "; ".join(errors)
        return _error_response(422, error_msg, "NO_PERSON_DETECTED")

    # Aggregate betas
    if len(all_betas) > 1:
        final_betas = aggregate_betas(all_betas)
    else:
        final_betas = all_betas[0]

    # Compute measurements
    measurer = compute_measurements(final_betas, gender, height)

    # Generate annotated 3D body model as GLB (body + measurement tubes)
    height_scale = None
    if height is not None:
        old_height = measurer.measurements.get("height")
        if old_height and old_height > 0:
            height_scale = height / old_height
    scene = _build_annotated_glb(measurer, height_scale)
    glb_buffer = io.BytesIO()
    scene.export(glb_buffer, file_type='glb')
    model_glb_base64 = base64.b64encode(glb_buffer.getvalue()).decode('ascii')

    inference_time = time.time() - start_time

    # Build betas list
    betas_list = final_betas.numpy().flatten().tolist()

    # Use height-normalized measurements if height provided, else raw
    if height and hasattr(measurer, 'height_normalized_measurements'):
        m_source = measurer.height_normalized_measurements
    else:
        m_source = measurer.measurements

    # Build structured measurements list + raw dict
    measurements_list = []
    raw_measurements = {}
    for label_code, m_name in STANDARD_LABELS.items():
        val = m_source.get(m_name)
        if val is not None:
            rounded = round(float(val), 1)
            raw_measurements[m_name] = rounded
            m_type = MEASUREMENT_TYPES.get(m_name, "length")
            rgba = _LABEL_COLORS.get(label_code, [255, 255, 255, 255])
            color_hex = "#{:02X}{:02X}{:02X}".format(rgba[0], rgba[1], rgba[2])
            measurements_list.append(MeasurementItem(
                name=m_name,
                name_tr=_TURKISH_NAMES.get(m_name, m_name),
                label=label_code,
                value_cm=rounded,
                type=m_type,
                color=color_hex,
            ))

    # Height ratio warning (Step 2)
    height_ratio = None
    if height is not None:
        model_height = m_source.get("height") or measurer.measurements.get("height")
        if model_height and float(model_height) > 0:
            height_ratio = round(height / float(model_height), 3)
            if height_ratio < 0.7 or height_ratio > 1.3:
                warnings.append(
                    f"Height ratio ({height_ratio:.2f}) is unusual. "
                    f"The person may not be standing straight, or detection was off."
                )

    # View quality assessment (Step 3c)
    avg_det_score = round(float(np.mean(all_det_scores)), 3) if all_det_scores else 0.0
    view_angle_diff = None
    if len(all_yaws) >= 2:
        view_angle_diff = round(_angular_diff(all_yaws[0], all_yaws[1]), 1)
        if view_angle_diff >= 60:
            view_quality = "good"
        elif view_angle_diff >= 30:
            view_quality = "fair"
        else:
            view_quality = "poor"
            warnings.append(
                f"Both photos appear to be from a similar angle "
                f"({view_angle_diff:.0f} deg apart). Use front + side views for best accuracy."
            )
    else:
        view_quality = "fair"  # single photo
        if len(all_betas) == 1:
            warnings.append("Only 1 photo processed. Use front + side photos for better accuracy.")

    quality = QualityInfo(
        detection_confidence=avg_det_score,
        view_angle_diff=view_angle_diff,
        view_quality=view_quality,
        height_ratio=height_ratio,
        warnings=warnings,
    )

    return MeasurementResponse(
        success=True,
        measurements=measurements_list,
        raw_measurements=raw_measurements,
        betas=betas_list,
        images_processed=len(all_betas),
        inference_time_sec=round(inference_time, 2),
        height_normalized=height is not None,
        gender=gender,
        quality=quality,
        model_glb=model_glb_base64,
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    return {
        "status": "ready" if _state["ready"] else "loading",
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
