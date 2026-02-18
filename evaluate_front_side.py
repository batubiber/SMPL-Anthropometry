"""
Front+Side vs Random pair evaluation on SSP-3D.

For each person with multiple images:
  1. Single photo (baseline)
  2. Random 2-photo pair (average betas)
  3. Front+Side pair: pick 2 photos with ~90 deg yaw difference (average betas)
  4. Most diverse pair: pick 2 photos with maximum yaw difference

Compare all against GT betas.

Usage:
    python evaluate_front_side.py
    python evaluate_front_side.py --max-persons 10
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
import itertools
from collections import defaultdict
import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation

from evaluate import evaluate_mae
from measure import MeasureBody
from measurement_definitions import STANDARD_LABELS
from photo_to_measurements import (
    load_hmr2_model, load_detector, detect_person, run_hmr2_inference
)


def compute_measurements_from_betas(betas, gender):
    measurer = MeasureBody("smpl")
    measurer.from_body_model(gender=gender, shape=betas)
    measurement_names = measurer.all_possible_measurements
    measurer.measure(measurement_names)
    return measurer.measurements


def get_yaw_angle(pose_params):
    """Extract yaw (Y-axis rotation) from SMPL global orient."""
    global_orient = pose_params[:3]
    rot = Rotation.from_rotvec(global_orient)
    euler = rot.as_euler('YXZ', degrees=True)
    return euler[0]


def angular_diff(a, b):
    """Compute absolute angular difference in degrees (0-180)."""
    diff = abs(a - b) % 360
    if diff > 180:
        diff = 360 - diff
    return diff


def find_best_front_side_pair(indices, yaws):
    """Find the pair of images with yaw difference closest to 90 degrees."""
    best_pair = None
    best_score = float('inf')

    for i, j in itertools.combinations(indices, 2):
        diff = angular_diff(yaws[i], yaws[j])
        score = abs(diff - 90)  # how close to 90 degree difference
        if score < best_score:
            best_score = score
            best_pair = (i, j)

    return best_pair, best_score


def find_most_diverse_pair(indices, yaws):
    """Find the pair with maximum angular difference."""
    best_pair = None
    best_diff = 0

    for i, j in itertools.combinations(indices, 2):
        diff = angular_diff(yaws[i], yaws[j])
        if diff > best_diff:
            best_diff = diff
            best_pair = (i, j)

    return best_pair, best_diff


def infer_betas_for_image(img_path, model, model_cfg, detector, device):
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
    betas, _global_orient = run_hmr2_inference(model, model_cfg, img_cv2, boxes, device)
    return betas


def compute_strategy_mae(betas_list, gt_betas_np, gender):
    """Average betas from list, compute measurement MAE vs GT."""
    if not betas_list:
        return None
    stacked = torch.stack(betas_list, dim=0)
    mean_betas = stacked.mean(dim=0)
    gt_tensor = torch.tensor(gt_betas_np, dtype=torch.float32).unsqueeze(0)
    gt_meas = compute_measurements_from_betas(gt_tensor, gender)
    pred_meas = compute_measurements_from_betas(mean_betas, gender)
    return evaluate_mae(gt_meas, pred_meas)


KEY_MEASUREMENTS = [
    'height', 'chest circumference', 'waist circumference',
    'hip circumference', 'shoulder breadth', 'bicep right circumference',
    'thigh left circumference', 'arm right length', 'inside leg height',
    'neck circumference',
]


def main():
    parser = argparse.ArgumentParser(
        description='On+Yan vs Rastgele cift karsilastirma')
    parser.add_argument('--ssp3d-dir', type=str, default='datasets/SSP-3D/ssp_3d')
    parser.add_argument('--max-persons', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load dataset
    data = np.load(os.path.join(args.ssp3d_dir, 'labels.npz'))
    fnames = data['fnames']
    shapes = data['shapes']
    poses = data['poses']
    genders = data['genders']
    images_dir = os.path.join(args.ssp3d_dir, 'images')

    # Compute yaw for all images
    yaws = {}
    for i in range(len(fnames)):
        yaws[i] = get_yaw_angle(poses[i])

    # Group by person (3+ images needed)
    groups = defaultdict(list)
    for i, shape in enumerate(shapes):
        key = tuple(shape.round(4))
        groups[key].append(i)

    persons = []
    for key, indices in groups.items():
        if len(indices) >= 3:
            gender_code = str(genders[indices[0]])
            gender = "MALE" if gender_code == 'm' else "FEMALE"

            # Find best front+side pair (~90 deg diff)
            fs_pair, fs_score = find_best_front_side_pair(indices, yaws)
            # Find most diverse pair (max diff)
            div_pair, div_diff = find_most_diverse_pair(indices, yaws)

            persons.append({
                'indices': indices,
                'gt_shape': shapes[indices[0]],
                'gender': gender,
                'front_side_pair': fs_pair,
                'fs_angle_diff': angular_diff(yaws[fs_pair[0]], yaws[fs_pair[1]]) if fs_pair else 0,
                'diverse_pair': div_pair,
                'div_angle_diff': div_diff,
            })

    # Sort by front_side quality (best ~90 deg pairs first)
    persons.sort(key=lambda p: abs(p['fs_angle_diff'] - 90))

    if args.max_persons:
        persons = persons[:args.max_persons]

    n = len(persons)
    print("=" * 70)
    print("  ON+YAN vs RASTGELE CIFT KARSILASTIRMA")
    print("  Kisi sayisi: {}".format(n))
    print("=" * 70)

    # Show selected pairs' angle info
    print("\n  Secilen on+yan ciftlerinin aci farklari:")
    for i, p in enumerate(persons):
        print("    Kisi {}: on+yan={:.0f}deg, max_diverse={:.0f}deg, foto_sayisi={}".format(
            i+1, p['fs_angle_diff'], p['div_angle_diff'], len(p['indices'])))

    # Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    if device.type == 'cuda':
        print("\n[INFO] GPU: {}".format(torch.cuda.get_device_name(0)))

    # Load models
    print("\n[1/3] Modeller yukleniyor...")
    model, model_cfg = load_hmr2_model(device)
    print("   HMR 2.0 yuklendi.")
    detector = load_detector(device)
    print("   ViTDet detector yuklendi.")

    # Phase 1: Cache all needed betas
    print("\n[2/3] Betas hesaplaniyor...")
    all_needed_indices = set()
    for p in persons:
        all_needed_indices.update(p['indices'])

    betas_cache = {}
    total = len(all_needed_indices)
    start_time = time.time()

    for count, img_idx in enumerate(sorted(all_needed_indices), 1):
        fname = str(fnames[img_idx])
        img_path = os.path.join(images_dir, fname)
        betas = infer_betas_for_image(img_path, model, model_cfg, detector, device)
        if betas is not None:
            betas_cache[img_idx] = betas
        if count % 10 == 0 or count == total:
            elapsed = time.time() - start_time
            remaining = (elapsed / count) * (total - count)
            print("  [{}/{}] cache={}, kalan~{:.0f}s".format(
                count, total, len(betas_cache), remaining))

    # Phase 2: Compare strategies
    print("\n[3/3] Stratejiler karsilastiriliyor...")

    strategies = {
        '1-foto (tek)': 'single',
        '2-foto (rastgele)': 'random',
        '2-foto (on+yan ~90)': 'front_side',
        '2-foto (max diverse)': 'diverse',
    }

    all_results = {name: [] for name in strategies}

    for p in persons:
        gt = p['gt_shape']
        gender = p['gender']
        available = [i for i in p['indices'] if i in betas_cache]

        if len(available) < 2:
            continue

        # Single photo (first available)
        single_betas = [betas_cache[available[0]]]
        err = compute_strategy_mae(single_betas, gt, gender)
        if err:
            all_results['1-foto (tek)'].append(err)

        # Random pair
        rand_pair = random.sample(available, 2)
        rand_betas = [betas_cache[i] for i in rand_pair]
        err = compute_strategy_mae(rand_betas, gt, gender)
        if err:
            all_results['2-foto (rastgele)'].append(err)

        # Front+side pair (~90 deg)
        fs = p['front_side_pair']
        if fs and fs[0] in betas_cache and fs[1] in betas_cache:
            fs_betas = [betas_cache[fs[0]], betas_cache[fs[1]]]
            err = compute_strategy_mae(fs_betas, gt, gender)
            if err:
                all_results['2-foto (on+yan ~90)'].append(err)

        # Most diverse pair
        div = p['diverse_pair']
        if div and div[0] in betas_cache and div[1] in betas_cache:
            div_betas = [betas_cache[div[0]], betas_cache[div[1]]]
            err = compute_strategy_mae(div_betas, gt, gender)
            if err:
                all_results['2-foto (max diverse)'].append(err)

    # Aggregate and print
    print("\n" + "=" * 85)
    print("  SONUCLAR")
    print("=" * 85)

    strat_names = list(strategies.keys())

    # Header
    header = "  {:<28s}".format("Olcu (MAE cm)")
    for sn in strat_names:
        header += " {:>14s}".format(sn[:14])
    print(header)
    print("  " + "-" * 82)

    overall = {sn: [] for sn in strat_names}

    for m_name in KEY_MEASUREMENTS:
        row = "  {:<28s}".format(m_name[:28])
        for sn in strat_names:
            errors = all_results[sn]
            values = [e[m_name] for e in errors if m_name in e]
            mae = np.mean(values) if values else float('nan')
            overall[sn].append(mae)
            row += " {:>13.2f} ".format(mae)
        print(row)

    # Overall
    print("  " + "-" * 82)
    row = "  {:<28s}".format("GENEL ORTALAMA")
    baseline_avg = None
    for sn in strat_names:
        avg = np.nanmean(overall[sn])
        if baseline_avg is None:
            baseline_avg = avg
        row += " {:>13.2f} ".format(avg)
    print(row)

    # Improvement row
    row = "  {:<28s}".format("Iyilesme vs tek foto")
    for sn in strat_names:
        avg = np.nanmean(overall[sn])
        if baseline_avg and baseline_avg > 0:
            imp = (baseline_avg - avg) / baseline_avg * 100
            row += " {:>+12.1f}% ".format(imp)
        else:
            row += " {:>14s}".format("-")
    print(row)
    print("=" * 85)

    elapsed_total = time.time() - start_time
    print("\nToplam sure: {:.1f} saniye".format(elapsed_total))


if __name__ == '__main__':
    main()
