"""
Photo to Body Measurements using HMR 2.0 + SMPL-Anthropometry

Extracts SMPL body shape parameters from one or more photos using HMR 2.0
(4D-Humans), then computes anthropometric measurements using SMPL-Anthropometry.

Multi-image mode averages betas across all views for improved accuracy.
Best results: front + side photos + known height.

Pipeline:
    Photo(s) -> detectron2 ViTDet (person detection) -> HMR 2.0 (SMPL estimation)
             -> beta averaging (if multi-image) -> SMPL-Anthropometry (measurements)

Usage (.venv must be active):
    Single image:
        python photo_to_measurements.py --images photo.jpg --gender MALE
    Multi-image (recommended: front + side):
        python photo_to_measurements.py --images front.jpg side.jpg --gender MALE --height 178
"""

import os
import sys
import importlib.abc
import importlib.machinery

# Windows: set HOME before any hmr2 imports (hmr2 uses HOME for cache dir)
if os.name == 'nt':
    os.environ.setdefault('HOME', os.path.expanduser('~'))

# ---- Mock pyrender to avoid OpenGL/EGL crash on Windows ----
# hmr2 imports pyrender for rendering, but we only need inference.
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
                'render': lambda s, *a, **k: (__import__('numpy').zeros((1,1,3), dtype=__import__('numpy').uint8),
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
# ---- End pyrender mock ----

# Add 4D-Humans to sys.path for hmr2 module
_script_dir = os.path.dirname(os.path.abspath(__file__))
_4dhumans_dir = os.path.join(_script_dir, '4D-Humans')
if os.path.isdir(_4dhumans_dir):
    sys.path.insert(0, _4dhumans_dir)

import argparse
import numpy as np
import torch
import cv2
from pathlib import Path


def load_hmr2_model(device):
    """Load HMR 2.0 model and config."""
    from hmr2.configs import CACHE_DIR_4DHUMANS
    from hmr2.models import download_models, load_hmr2, DEFAULT_CHECKPOINT

    # PyTorch 2.6+ defaults to weights_only=True, but HMR2 checkpoint
    # contains OmegaConf objects. Patch torch.load to use weights_only=False.
    _original_torch_load = torch.load
    def _patched_torch_load(*args, **kwargs):
        kwargs['weights_only'] = False
        return _original_torch_load(*args, **kwargs)
    torch.load = _patched_torch_load

    # Ensure checkpoint exists (downloads ~700MB on first run)
    ckpt_path = Path(DEFAULT_CHECKPOINT)
    if not ckpt_path.exists():
        print("   Checkpoint indiriliyor (ilk calistirmada ~700MB)...")
        download_models(CACHE_DIR_4DHUMANS)

    model, model_cfg = load_hmr2(DEFAULT_CHECKPOINT)
    model = model.to(device)
    model.eval()
    return model, model_cfg


def load_detector(device):
    """Load ViTDet person detector via detectron2."""
    from detectron2.config import LazyConfig
    import hmr2

    cfg_path = Path(hmr2.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = (
        "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
        "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25

    from hmr2.utils.utils_detectron2 import DefaultPredictor_Lazy
    detector = DefaultPredictor_Lazy(detectron2_cfg)
    return detector


def detect_person(detector, img_cv2):
    """Detect persons in image, return (bounding_boxes, scores)."""
    det_out = detector(img_cv2)
    det_instances = det_out['instances']
    valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
    boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
    scores = det_instances.scores[valid_idx].cpu().numpy()
    return boxes, scores


def run_hmr2_inference(model, model_cfg, img_cv2, boxes, device):
    """Run HMR 2.0 on detected person, return (betas, global_orient).

    Returns:
        betas: torch.Tensor (1, 10) shape parameters
        global_orient: torch.Tensor (1, 3) axis-angle global orientation
    Returns (None, None) on failure.
    """
    from hmr2.utils import recursive_to
    from hmr2.datasets.vitdet_dataset import ViTDetDataset

    dataset = ViTDetDataset(model_cfg, img_cv2, boxes[:1])  # first person only
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=1, shuffle=False, num_workers=0
    )

    for batch in dataloader:
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        betas = out['pred_smpl_params']['betas']  # (1, 10)
        global_orient = out['pred_smpl_params']['global_orient']  # (1, 1, 3, 3) or (1, 3)
        global_orient_cpu = global_orient.cpu()

        # Convert rotation matrix to axis-angle if needed
        if global_orient_cpu.dim() == 4:
            # (1, 1, 3, 3) rotation matrix -> flatten to (3, 3)
            from scipy.spatial.transform import Rotation
            rot_mat = global_orient_cpu[0, 0].numpy()  # (3, 3)
            rotvec = Rotation.from_matrix(rot_mat).as_rotvec()
            global_orient_cpu = torch.tensor(rotvec, dtype=torch.float32).unsqueeze(0)  # (1, 3)
        elif global_orient_cpu.dim() == 3:
            global_orient_cpu = global_orient_cpu.squeeze(1)  # (1, 3)

        return betas.cpu(), global_orient_cpu

    return None, None


def compute_measurements(betas, gender, known_height=None):
    """Compute body measurements from SMPL betas."""
    from measure import MeasureBody
    from measurement_definitions import STANDARD_LABELS

    measurer = MeasureBody("smpl")
    measurer.from_body_model(gender=gender, shape=betas)

    measurement_names = measurer.all_possible_measurements
    measurer.measure(measurement_names)
    measurer.label_measurements(STANDARD_LABELS)

    if known_height:
        measurer.height_normalize_measurements(known_height)

    return measurer


def print_results(measurer, known_height=None, n_images=1):
    """Print measurement results in a formatted table."""
    labels = measurer.labeled_measurements
    names = measurer.labels2names

    print("\n" + "=" * 60)
    print("            VUCUT OLCULERI (cm)")
    if known_height:
        print(f"        (Boy normalizasyonu: {known_height} cm)")
    if n_images > 1:
        print(f"        ({n_images} fotografin beta ortalamasi)")
    print("=" * 60)

    label_order = [
        'P', 'A', 'B', 'O', 'D', 'E', 'F',
        'H', 'I', 'G', 'J', 'C', 'K', 'L', 'M', 'N'
    ]

    turkish_names = {
        'height': 'Boy',
        'head circumference': 'Bas cevresi',
        'neck circumference': 'Boyun cevresi',
        'shoulder breadth': 'Omuz genisligi',
        'chest circumference': 'Gogus cevresi',
        'waist circumference': 'Bel cevresi',
        'hip circumference': 'Kalca cevresi',
        'bicep right circumference': 'Sag biceps cevresi',
        'forearm right circumference': 'Sag onkol cevresi',
        'wrist right circumference': 'Sag bilek cevresi',
        'arm right length': 'Sag kol uzunlugu',
        'shoulder to crotch height': 'Omuz-kasik yuksekligi',
        'inside leg height': 'Ic bacak yuksekligi',
        'thigh left circumference': 'Sol uyluk cevresi',
        'calf left circumference': 'Sol baldir cevresi',
        'ankle left circumference': 'Sol ayak bilegi cevresi',
    }

    for label in label_order:
        if label in labels:
            name = names[label]
            tr_name = turkish_names.get(name, name)

            if known_height and hasattr(measurer, 'height_normalized_measurements'):
                value = measurer.height_normalized_measurements.get(name, labels[label])
            else:
                value = labels[label]

            print(f"  {label}  {tr_name:<30s} {float(value):>8.1f} cm")

    print("=" * 60)

    tips = []
    if not known_height:
        tips.append("Boyunuzu girin: --height 178")
    if n_images == 1:
        tips.append("On + yan fotograf verin: --images on.jpg yan.jpg")

    if tips:
        print("\n  IPUCU: Daha dogru sonuclar icin:")
        for tip in tips:
            print(f"    - {tip}")


def process_single_image(image_path, model, model_cfg, detector, device, image_idx=None, total_images=None):
    """Process a single image: detect person + run HMR 2.0, return betas.

    Returns:
        betas (torch.Tensor): Shape parameters (1, 10), or None if failed.
    """
    prefix = f"   [{image_idx}/{total_images}] " if image_idx else "   "
    img_name = os.path.basename(image_path)

    img_cv2 = cv2.imread(image_path)
    if img_cv2 is None:
        print(f"{prefix}WARNING: Could not read image, skipping: {img_name}")
        return None

    # Detect person
    boxes, scores = detect_person(detector, img_cv2)
    if len(boxes) == 0:
        print(f"{prefix}WARNING: {img_name} - no person detected, skipping.")
        return None

    # Pick largest person if multiple detected
    if len(boxes) > 1:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = np.argmax(areas)
        boxes = boxes[idx:idx + 1]

    bbox_str = boxes[0].astype(int).tolist()

    # Run HMR 2.0
    betas, _global_orient = run_hmr2_inference(model, model_cfg, img_cv2, boxes, device)
    if betas is None:
        print(f"{prefix}WARNING: {img_name} - HMR 2.0 failed, skipping.")
        return None

    betas_np = betas.numpy().flatten()
    print(f"{prefix}{img_name}: bbox={bbox_str}, betas=[{', '.join(f'{b:.3f}' for b in betas_np[:5])}]")
    return betas


def aggregate_betas(betas_list):
    """Average betas from multiple views for more robust shape estimation.

    Args:
        betas_list: List of (1, 10) tensors from individual images.

    Returns:
        Averaged betas tensor (1, 10).
    """
    stacked = torch.stack(betas_list, dim=0)  # (N, 1, 10)
    mean_betas = stacked.mean(dim=0)           # (1, 10)
    return mean_betas


def main():
    parser = argparse.ArgumentParser(
        description='Fotograftan vucut olculeri cikar (HMR 2.0 + SMPL-Anthropometry)'
    )
    # Support both --images (new) and --image (backward compatible)
    parser.add_argument('--images', type=str, nargs='+',
                        help='Giris fotograf(lar)inin yolu (birden fazla verilebilir)')
    parser.add_argument('--image', type=str,
                        help='Tek fotograf (geriye uyumluluk icin, --images tercih edin)')
    parser.add_argument('--gender', type=str, default='NEUTRAL',
                        choices=['MALE', 'FEMALE', 'NEUTRAL'],
                        help='Cinsiyet (varsayilan: NEUTRAL)')
    parser.add_argument('--height', type=float, default=None,
                        help='Bilinen boy (cm) - olculeri normalize eder (istege bagli)')
    args = parser.parse_args()

    # Resolve image list: --images takes priority, fallback to --image
    image_paths = args.images or ([args.image] if args.image else None)
    if not image_paths:
        parser.error("En az bir fotograf gerekli: --images front.jpg side.jpg veya --image photo.jpg")

    # Validate all input files exist
    for img_path in image_paths:
        if not os.path.exists(img_path):
            sys.exit(f"Fotograf bulunamadi: {img_path}")

    multi_mode = len(image_paths) > 1
    n_images = len(image_paths)

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cpu':
        print("[UYARI] CUDA bulunamadi, CPU kullaniliyor (yavas olacak)")
    else:
        print(f"[INFO] GPU kullaniliyor: {torch.cuda.get_device_name(0)}")

    if multi_mode:
        print(f"[INFO] Multi-image modu: {n_images} fotograf isleniyor (beta ortalamasi)")

    # Step 1: Load models
    print("\n[1/4] Modeller yukleniyor...")
    model, model_cfg = load_hmr2_model(device)
    print("   HMR 2.0 modeli yuklendi.")

    print("   Person detector yukleniyor...")
    detector = load_detector(device)
    print("   ViTDet detector yuklendi.")

    # Step 2 & 3: Process each image (detect + HMR 2.0)
    print(f"\n[2/4] Kisi tespiti + HMR 2.0 tahmin ({n_images} fotograf)...")
    all_betas = []
    for i, img_path in enumerate(image_paths, 1):
        betas = process_single_image(
            img_path, model, model_cfg, detector, device,
            image_idx=i, total_images=n_images
        )
        if betas is not None:
            all_betas.append(betas)

    if len(all_betas) == 0:
        sys.exit("Hicbir fotograftan shape parametresi cikarilamadi!")

    # Step 3: Aggregate betas
    if multi_mode:
        print(f"\n[3/4] Beta ortalamasi hesaplaniyor ({len(all_betas)}/{n_images} basarili)...")
        if len(all_betas) < n_images:
            print(f"   UYARI: {n_images - len(all_betas)} fotograf islenmedi, "
                  f"kalan {len(all_betas)} fotograf kullaniliyor.")
        final_betas = aggregate_betas(all_betas)
        betas_np = final_betas.numpy().flatten()
        print(f"   Ortalama betas (ilk 5): [{', '.join(f'{b:.3f}' for b in betas_np[:5])}]")
    else:
        print("\n[3/4] Tek fotograf modu - beta ortalamasi gerekmez.")
        final_betas = all_betas[0]
        betas_np = final_betas.numpy().flatten()
        print(f"   Betas (ilk 5): [{', '.join(f'{b:.3f}' for b in betas_np[:5])}]")

    # Step 4: Compute measurements
    print(f"\n[4/4] Vucut olculeri hesaplaniyor (cinsiyet: {args.gender})...")
    measurer = compute_measurements(final_betas, args.gender, args.height)

    # Print results
    print_results(measurer, args.height, n_images=len(all_betas))


if __name__ == '__main__':
    main()
