"""
Evaluate HMR 2.0 + SMPL-Anthropometry pipeline on SSP-3D dataset.

For each image in SSP-3D:
  1. Run HMR 2.0 to predict SMPL betas
  2. Compute measurements from predicted betas (SMPL-Anthropometry)
  3. Compute measurements from ground truth betas (SMPL-Anthropometry)
  4. Compare using MAE

Usage:
    python evaluate_ssp3d.py
    python evaluate_ssp3d.py --max-samples 20       # quick test with 20 images
    python evaluate_ssp3d.py --max-samples 50 --save-csv results.csv
"""

import os
import sys
import importlib.abc
import importlib.machinery

# Windows HOME
if os.name == 'nt':
    os.environ.setdefault('HOME', os.path.expanduser('~'))

# Mock pyrender
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

# Add 4D-Humans
_script_dir = os.path.dirname(os.path.abspath(__file__))
_4dhumans_dir = os.path.join(_script_dir, '4D-Humans')
if os.path.isdir(_4dhumans_dir):
    sys.path.insert(0, _4dhumans_dir)

import argparse
import time
import numpy as np
import torch
import cv2
import pandas as pd
from pathlib import Path

from evaluate import evaluate_mae
from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS

# Import pipeline functions from photo_to_measurements
from photo_to_measurements import (
    load_hmr2_model, load_detector, detect_person, run_hmr2_inference
)


def compute_measurements_from_betas(betas, gender):
    """Compute body measurements from SMPL betas. Returns dict of {name: value_cm}."""
    measurer = MeasureBody("smpl")
    measurer.from_body_model(gender=gender, shape=betas)

    measurement_names = measurer.all_possible_measurements
    measurer.measure(measurement_names)
    measurer.label_measurements(STANDARD_LABELS)

    return measurer.measurements


def main():
    parser = argparse.ArgumentParser(description='SSP-3D uzerinde pipeline evaluation')
    parser.add_argument('--ssp3d-dir', type=str,
                        default='datasets/SSP-3D/ssp_3d',
                        help='SSP-3D dataset dizini')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Max fotograf sayisi (hizli test icin)')
    parser.add_argument('--save-csv', type=str, default=None,
                        help='Sonuclari CSV olarak kaydet')
    args = parser.parse_args()

    # Load SSP-3D labels
    labels_path = os.path.join(args.ssp3d_dir, 'labels.npz')
    if not os.path.exists(labels_path):
        sys.exit("SSP-3D labels bulunamadi: {}".format(labels_path))

    data = np.load(labels_path)
    fnames = data['fnames']
    gt_shapes = data['shapes']      # (311, 10)
    genders = data['genders']        # 'm' or 'f'
    images_dir = os.path.join(args.ssp3d_dir, 'images')

    n_total = len(fnames)
    n_eval = min(args.max_samples, n_total) if args.max_samples else n_total
    print("=" * 60)
    print("  SSP-3D Evaluation: {} / {} fotograf".format(n_eval, n_total))
    print("=" * 60)

    # Setup device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        print("[INFO] GPU: {}".format(torch.cuda.get_device_name(0)))
    else:
        print("[UYARI] CPU kullaniliyor")

    # Load models
    print("\n[1/3] Modeller yukleniyor...")
    model, model_cfg = load_hmr2_model(device)
    print("   HMR 2.0 yuklendi.")
    detector = load_detector(device)
    print("   ViTDet detector yuklendi.")

    # Process each image
    print("\n[2/3] Fotograflar isleniyor...")
    all_errors = []           # list of dicts {measurement: error_cm}
    all_betas_errors = []     # list of per-sample mean betas L2 error
    n_success = 0
    n_failed = 0
    start_time = time.time()

    for i in range(n_eval):
        fname = str(fnames[i])
        gt_betas = gt_shapes[i]                   # (10,)
        gender_code = str(genders[i])
        gender = "MALE" if gender_code == 'm' else "FEMALE"

        img_path = os.path.join(images_dir, fname)
        if not os.path.exists(img_path):
            print("  [{}/{}] ATLA: {} bulunamadi".format(i+1, n_eval, fname))
            n_failed += 1
            continue

        # Read image
        img_cv2 = cv2.imread(img_path)
        if img_cv2 is None:
            print("  [{}/{}] ATLA: {} okunamadi".format(i+1, n_eval, fname))
            n_failed += 1
            continue

        # Detect person
        boxes, _scores = detect_person(detector, img_cv2)
        if len(boxes) == 0:
            print("  [{}/{}] SKIP: {} no person detected".format(i+1, n_eval, fname))
            n_failed += 1
            continue

        # Pick largest
        if len(boxes) > 1:
            areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
            idx = np.argmax(areas)
            boxes = boxes[idx:idx + 1]

        # HMR 2.0 inference
        pred_betas, _orient = run_hmr2_inference(model, model_cfg, img_cv2, boxes, device)
        if pred_betas is None:
            print("  [{}/{}] ATLA: {} HMR2 basarisiz".format(i+1, n_eval, fname))
            n_failed += 1
            continue

        pred_betas_np = pred_betas.numpy().flatten()  # (10,)

        # Betas L2 error
        betas_l2 = np.sqrt(np.sum((gt_betas - pred_betas_np) ** 2))
        all_betas_errors.append(betas_l2)

        # Compute measurements from GT betas
        gt_betas_tensor = torch.tensor(gt_betas, dtype=torch.float32).unsqueeze(0)
        gt_measurements = compute_measurements_from_betas(gt_betas_tensor, gender)

        # Compute measurements from predicted betas
        pred_measurements = compute_measurements_from_betas(pred_betas, gender)

        # MAE per measurement
        errors = evaluate_mae(gt_measurements, pred_measurements)
        all_errors.append(errors)
        n_success += 1

        # Progress
        elapsed = time.time() - start_time
        avg_time = elapsed / (i + 1)
        remaining = avg_time * (n_eval - i - 1)

        if (i + 1) % 10 == 0 or (i + 1) == n_eval:
            print("  [{}/{}] basarili={}, hata={}, betas_L2={:.3f}, kalan~{:.0f}s".format(
                i+1, n_eval, n_success, n_failed, betas_l2, remaining))

    # Aggregate results
    print("\n[3/3] Sonuclar hesaplaniyor...")
    print("=" * 60)
    print("  Basarili: {} / {}".format(n_success, n_eval))
    print("  Basarisiz: {}".format(n_failed))

    if n_success == 0:
        sys.exit("Hicbir fotograf basariyla islenmedi!")

    # Mean betas L2 error
    mean_betas_l2 = np.mean(all_betas_errors)
    print("  Ortalama Betas L2 Hatasi: {:.4f}".format(mean_betas_l2))

    # Compute mean MAE across all samples for each measurement
    measurement_names = set()
    for errors in all_errors:
        measurement_names.update(errors.keys())

    results = {}
    for m_name in sorted(measurement_names):
        values = [e[m_name] for e in all_errors if m_name in e]
        results[m_name] = {
            'MAE_cm': np.mean(values),
            'STD_cm': np.std(values),
            'Median_cm': np.median(values),
            'N': len(values),
        }

    # Print results table
    print("\n" + "=" * 70)
    print("  OLCU BAZINDA MAE (cm) - {} fotograf uzerinden".format(n_success))
    print("=" * 70)
    print("  {:<35s} {:>8s} {:>8s} {:>8s}".format("Olcu", "MAE", "STD", "Median"))
    print("  " + "-" * 60)

    total_mae = []
    for m_name in sorted(results.keys()):
        r = results[m_name]
        total_mae.append(r['MAE_cm'])
        print("  {:<35s} {:>7.2f}  {:>7.2f}  {:>7.2f}".format(
            m_name, r['MAE_cm'], r['STD_cm'], r['Median_cm']))

    print("  " + "-" * 60)
    print("  {:<35s} {:>7.2f}".format("GENEL ORTALAMA MAE", np.mean(total_mae)))
    print("=" * 70)

    # Save to CSV
    if args.save_csv:
        rows = []
        for m_name, r in results.items():
            rows.append({
                'measurement': m_name,
                'MAE_cm': round(r['MAE_cm'], 3),
                'STD_cm': round(r['STD_cm'], 3),
                'Median_cm': round(r['Median_cm'], 3),
                'N': r['N']
            })
        df = pd.DataFrame(rows)
        df.to_csv(args.save_csv, index=False)
        print("\nSonuclar kaydedildi: {}".format(args.save_csv))

    elapsed_total = time.time() - start_time
    print("\nToplam sure: {:.1f} saniye ({:.1f}s/fotograf)".format(
        elapsed_total, elapsed_total / n_eval))


if __name__ == '__main__':
    main()
