"""
Multi-view vs Single-view Evaluation on SSP-3D dataset.

SSP-3D has multiple images per person (from video frames at different angles).
This script compares:
  - Single-photo: 1 random image per person -> betas -> measurements
  - Multi-photo (2): 2 random images per person -> avg betas -> measurements
  - Multi-photo (3): 3 random images -> avg betas -> measurements
  - Multi-photo (all): all images -> avg betas -> measurements

All compared against ground truth betas from SSP-3D.

Usage:
    python evaluate_multiview.py
    python evaluate_multiview.py --max-persons 10   # quick test
    python evaluate_multiview.py --runs 5           # repeat for stability
"""

import os
import sys
import importlib.abc
import importlib.machinery

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

_script_dir = os.path.dirname(os.path.abspath(__file__))
_4dhumans_dir = os.path.join(_script_dir, '4D-Humans')
if os.path.isdir(_4dhumans_dir):
    sys.path.insert(0, _4dhumans_dir)

import argparse
import time
import random
from collections import defaultdict
import numpy as np
import torch
import cv2

from evaluate import evaluate_mae
from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS
from photo_to_measurements import (
    load_hmr2_model, load_detector, detect_person, run_hmr2_inference
)


def compute_measurements_from_betas(betas, gender):
    """Compute body measurements from SMPL betas."""
    measurer = MeasureBody("smpl")
    measurer.from_body_model(gender=gender, shape=betas)
    measurement_names = measurer.all_possible_measurements
    measurer.measure(measurement_names)
    measurer.label_measurements(STANDARD_LABELS)
    return measurer.measurements


def group_by_person(fnames, shapes, genders):
    """Group images by person (same shape = same person)."""
    groups = defaultdict(list)
    for i, shape in enumerate(shapes):
        key = tuple(shape.round(4))
        groups[key].append(i)

    persons = []
    for key, indices in groups.items():
        if len(indices) >= 2:  # need at least 2 images for multi-view test
            persons.append({
                'indices': indices,
                'gt_shape': shapes[indices[0]],
                'gender_code': str(genders[indices[0]]),
                'gender': "MALE" if str(genders[indices[0]]) == 'm' else "FEMALE",
                'fnames': [str(fnames[i]) for i in indices],
            })
    return persons


def infer_betas_for_image(img_path, model, model_cfg, detector, device):
    """Run detection + HMR2 on a single image, return betas or None."""
    img_cv2 = cv2.imread(img_path)
    if img_cv2 is None:
        return None

    boxes, _scores = detect_person(detector, img_cv2)
    if len(boxes) == 0:
        return None

    if len(boxes) > 1:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        idx = np.argmax(areas)
        boxes = boxes[idx:idx + 1]

    betas, _orient = run_hmr2_inference(model, model_cfg, img_cv2, boxes, device)
    return betas


def compute_mae_for_betas(pred_betas, gt_betas_np, gender):
    """Given predicted and GT betas, compute measurement MAE dict."""
    gt_betas_tensor = torch.tensor(gt_betas_np, dtype=torch.float32).unsqueeze(0)
    gt_measurements = compute_measurements_from_betas(gt_betas_tensor, gender)
    pred_measurements = compute_measurements_from_betas(pred_betas, gender)
    errors = evaluate_mae(gt_measurements, pred_measurements)
    return errors


def aggregate_errors(all_errors):
    """Aggregate list of error dicts into {measurement: mean_MAE}."""
    measurement_names = set()
    for e in all_errors:
        measurement_names.update(e.keys())

    results = {}
    for m_name in sorted(measurement_names):
        values = [e[m_name] for e in all_errors if m_name in e]
        if values:
            results[m_name] = np.mean(values)
    return results


# Key measurements we care about most
KEY_MEASUREMENTS = [
    'height', 'chest circumference', 'waist circumference',
    'hip circumference', 'shoulder breadth', 'bicep right circumference',
    'thigh left circumference', 'arm right length', 'inside leg height',
    'neck circumference',
]


def main():
    parser = argparse.ArgumentParser(
        description='Multi-view vs Single-view karsilastirma (SSP-3D)')
    parser.add_argument('--ssp3d-dir', type=str,
                        default='datasets/SSP-3D/ssp_3d')
    parser.add_argument('--max-persons', type=int, default=None,
                        help='Max kisi sayisi (hizli test)')
    parser.add_argument('--runs', type=int, default=3,
                        help='Tekrar sayisi (randomness icin, varsayilan 3)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    labels_path = os.path.join(args.ssp3d_dir, 'labels.npz')
    if not os.path.exists(labels_path):
        sys.exit("SSP-3D bulunamadi: " + labels_path)

    data = np.load(labels_path)
    images_dir = os.path.join(args.ssp3d_dir, 'images')

    persons = group_by_person(data['fnames'], data['shapes'], data['genders'])

    # Filter persons with enough images
    persons_3plus = [p for p in persons if len(p['indices']) >= 3]

    n_persons = len(persons_3plus)
    if args.max_persons:
        n_persons = min(args.max_persons, n_persons)
        persons_3plus = persons_3plus[:n_persons]

    print("=" * 70)
    print("  MULTI-VIEW vs SINGLE-VIEW KARSILASTIRMA")
    print("  Kisi sayisi: {} (3+ fotografa sahip)".format(n_persons))
    print("  Tekrar sayisi: {}".format(args.runs))
    print("=" * 70)

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        print("[INFO] GPU: {}".format(torch.cuda.get_device_name(0)))

    # Load models
    print("\n[1/3] Modeller yukleniyor...")
    model, model_cfg = load_hmr2_model(device)
    print("   HMR 2.0 yuklendi.")
    detector = load_detector(device)
    print("   ViTDet detector yuklendi.")

    # Phase 1: Pre-compute betas for ALL images of all persons
    print("\n[2/3] Tum fotograflar icin betas hesaplaniyor...")
    # Cache: image_index -> predicted betas tensor
    betas_cache = {}
    total_images = sum(len(p['indices']) for p in persons_3plus)
    processed = 0
    failed = 0
    start_time = time.time()

    for p_idx, person in enumerate(persons_3plus):
        for img_idx in person['indices']:
            fname = str(data['fnames'][img_idx])
            img_path = os.path.join(images_dir, fname)

            betas = infer_betas_for_image(img_path, model, model_cfg, detector, device)
            if betas is not None:
                betas_cache[img_idx] = betas
            else:
                failed += 1

            processed += 1
            if processed % 20 == 0 or processed == total_images:
                elapsed = time.time() - start_time
                remaining = (elapsed / processed) * (total_images - processed)
                print("  [{}/{}] cache={}, fail={}, kalan~{:.0f}s".format(
                    processed, total_images, len(betas_cache), failed, remaining))

    print("  Toplam: {} basarili, {} basarisiz".format(len(betas_cache), failed))

    # Phase 2: Compare single vs multi-view
    print("\n[3/3] Karsilastirma yapiliyor...")

    # Strategies to compare
    strategies = {
        '1-foto (tek)': 1,
        '2-foto (ort)': 2,
        '3-foto (ort)': 3,
        'Tum-foto (ort)': None,  # all available
    }

    all_strategy_results = {name: [] for name in strategies}

    for run in range(args.runs):
        for strat_name, n_photos in strategies.items():
            run_errors = []

            for person in persons_3plus:
                gender = person['gender']
                gt_betas = person['gt_shape']

                # Get available cached betas for this person
                available = [i for i in person['indices'] if i in betas_cache]
                if len(available) == 0:
                    continue

                # Select images based on strategy
                if n_photos is None:
                    selected = available
                else:
                    if len(available) < n_photos:
                        continue  # skip if not enough images
                    selected = random.sample(available, n_photos)

                # Average betas
                betas_list = [betas_cache[i] for i in selected]
                stacked = torch.stack(betas_list, dim=0)
                mean_betas = stacked.mean(dim=0)

                # Compute MAE
                errors = compute_mae_for_betas(mean_betas, gt_betas, gender)
                run_errors.append(errors)

            all_strategy_results[strat_name].append(run_errors)

    # Aggregate across runs
    print("\n" + "=" * 70)
    print("  SONUCLAR (ortalama {} run uzerinden)".format(args.runs))
    print("=" * 70)

    # Collect per-strategy aggregated MAEs
    strategy_summary = {}  # {strategy: {measurement: mean_mae}}
    for strat_name in strategies:
        combined_errors = []
        for run_errors in all_strategy_results[strat_name]:
            combined_errors.extend(run_errors)
        strategy_summary[strat_name] = aggregate_errors(combined_errors)

    # Print comparison table for key measurements
    strat_names = list(strategies.keys())
    header = "  {:<28s}".format("Olcu (MAE cm)")
    for sn in strat_names:
        header += " {:>12s}".format(sn)
    header += " {:>10s}".format("Iyilesme%")
    print(header)
    print("  " + "-" * (28 + 12 * len(strat_names) + 10))

    overall_maes = {sn: [] for sn in strat_names}

    for m_name in KEY_MEASUREMENTS:
        row = "  {:<28s}".format(m_name[:28])
        values = {}
        for sn in strat_names:
            val = strategy_summary[sn].get(m_name, float('nan'))
            values[sn] = val
            overall_maes[sn].append(val)
            row += " {:>11.2f} ".format(val)

        # Improvement: single vs all-photo
        single = values[strat_names[0]]
        best = values[strat_names[-1]]
        if single > 0:
            improvement = (single - best) / single * 100
            row += " {:>+8.1f}%".format(improvement)
        print(row)

    # Overall average
    print("  " + "-" * (28 + 12 * len(strat_names) + 10))
    row = "  {:<28s}".format("GENEL ORTALAMA")
    for sn in strat_names:
        avg = np.nanmean(overall_maes[sn])
        row += " {:>11.2f} ".format(avg)

    single_avg = np.nanmean(overall_maes[strat_names[0]])
    best_avg = np.nanmean(overall_maes[strat_names[-1]])
    if single_avg > 0:
        improvement = (single_avg - best_avg) / single_avg * 100
        row += " {:>+8.1f}%".format(improvement)
    print(row)
    print("=" * 70)

    # Betas L2 comparison
    print("\n  BETAS L2 HATASI:")
    for strat_name in strategies:
        l2_errors = []
        for run_errors_list in all_strategy_results[strat_name]:
            # recompute L2 for this run - we need to redo the averaging
            pass  # L2 already implicit in measurement MAE, skip for clarity

    elapsed_total = time.time() - start_time
    print("\nToplam sure: {:.1f} saniye".format(elapsed_total))


if __name__ == '__main__':
    main()
