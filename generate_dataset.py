#!/usr/bin/env python3
"""
============================================================
DRIFT-SENSE AI - Master Dataset Generation Pipeline
============================================================
Usage:
    python generate_dataset.py

When executed, this script:
1. Asks you how many total dataset samples you want (e.g. 76, 500, 10000).
2. Runs all 76 base DRAM generator scripts (from Dataset_generation_script/)
   to produce the 76 base image pairs into results/generated_dataset_images/.
3. Augments the 76 base pairs (SEM physics noise, rotations, blur, etc.)
   to reach the requested total sample count.
4. Splits the final dataset into train / val / test and saves into dataset/.

Output folder structure:
    dataset/
    |-- train/
    |   |-- dram_00001/ (reference_100x.png, search_10x.png, ground_truth.json)
    |   ...
    |-- val/
    |   ...
    |-- test/
    |   ...

Split ratios: Train=70%, Val=15%, Test=15%
(If total < 76, only runs the needed generators and skips augmentation.)
============================================================
"""

import importlib.util
import json
import math
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

# ============================================================
# PATHS
# ============================================================
ROOT         = Path(__file__).resolve().parent
GEN_DIR      = ROOT / "Dataset_generation_script"   # 76 generation scripts
BASE_DIR     = ROOT / "results" / "generated_dataset_images"  # output of the 76 scripts
DATASET_DIR  = ROOT / "dataset"                     # final train/val/test output

NUM_GENERATORS = 76

# ============================================================
# SPLIT RATIOS
# ============================================================
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.15
TEST_RATIO  = 0.15


# ============================================================
# STEP 1 - Run the 76 base generator scripts
# ============================================================

def load_and_run_generator(script_path: Path, output_base_dir: Path):
    """Loads a generator module and calls its generate() or main() function."""
    spec = importlib.util.spec_from_file_location(script_path.stem, script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module {script_path}")
    module = importlib.util.module_from_spec(spec)

    # Patch the OUTPUT_DIR in the module to point into our results folder
    sys.path.insert(0, str(GEN_DIR))
    spec.loader.exec_module(module)
    sys.path.pop(0)

    if hasattr(module, "main"):
        module.main()
    elif hasattr(module, "generate"):
        module.generate()
    else:
        raise RuntimeError(f"Generator {script_path.name} has no main() or generate() function.")


def run_all_generators(needed: int):
    """
    Runs the first `needed` generator scripts (max 76).
    Generates base pairs into results/generated_dataset_images/dram_XX/
    """
    needed = min(needed, NUM_GENERATORS)
    print(f"\nStep 1: Running {needed} base DRAM generator scripts...")

    BASE_DIR.mkdir(parents=True, exist_ok=True)

    for i in range(1, needed + 1):
        script_name = f"generate_dram_{i:02d}.py"
        script_path = GEN_DIR / script_name

        if not script_path.exists():
            print(f"  [WARNING] Missing script: {script_name}, skipping.")
            continue

        pair_dir = BASE_DIR / f"dram_{i:02d}"
        if (pair_dir / "reference_100x.png").exists() and (pair_dir / "ground_truth.json").exists():
            print(f"  [SKIP] dram_{i:02d} already generated.")
            continue

        print(f"  Running {script_name} ...", flush=True)
        load_and_run_generator(script_path, BASE_DIR)

    base_folders = sorted([
        d for d in BASE_DIR.iterdir()
        if d.is_dir() and d.name.startswith("dram_")
    ])
    print(f"  Base pairs ready: {len(base_folders)}")
    return base_folders


# ============================================================
# STEP 2 - Augmentation helpers (SEM physics)
# ============================================================

def apply_lens_distortion(img, k):
    if abs(k) < 1e-4:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    r2 = xx**2 + yy**2
    factor = 1 + k * r2
    map_x = (xx * factor * cx + cx).astype(np.float32)
    map_y = (yy * factor * cy + cy).astype(np.float32)
    return cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REFLECT)


def transform_point_lens_distortion(pt, k, img_size=1000):
    if abs(k) < 1e-4:
        return pt
    cx = cy = img_size / 2.0
    nx = (pt[0] - cx) / cx
    ny = (pt[1] - cy) / cy
    r2 = nx**2 + ny**2
    factor = 1 + k * r2
    return [nx * factor * cx + cx, ny * factor * cy + cy]


def apply_charging_streaks(img, count, intensity):
    if count <= 0 or intensity <= 0:
        return img
    h, w = img.shape[:2]
    result = img.astype(np.float32)
    num_streaks = max(1, int(round(count * (h / 100.0))))
    for _ in range(num_streaks):
        row        = random.randint(0, h - 1)
        streak_h   = random.randint(1, 5)
        streak_val = random.uniform(-1.0, 1.0) * intensity * 25.0
        col_start  = random.randint(0, w // 2)
        col_end    = random.randint(col_start + 50, w)
        row_end    = min(row + streak_h, h)
        result[row:row_end, col_start:col_end] += streak_val
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_vignette(img, strength):
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx**2 + yy**2) / np.sqrt(2.0)
    mask = np.clip(1.0 - strength * (r**2), 0.0, 1.0)
    if len(img.shape) == 3:
        mask = mask[:, :, np.newaxis]
    return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)


def apply_sem_physics(img, is_search, params):
    out = img.copy()
    h, w = out.shape[:2]

    # CD bias
    cd = params["cd_bias_nm"]
    if abs(cd) > 0.5:
        ks = max(1, int(np.clip(abs(cd) / 2.0, 1, 7)))
        if ks % 2 == 0: ks += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ks, ks))
        out = cv2.dilate(out, kernel) if cd > 0 else cv2.erode(out, kernel)

    # Beam blur + astigmatism
    sx = max(0.3, params["beam_spot_nm"] / 6.0)
    sy = max(0.3, (params["beam_spot_nm"] * params["astigmatism_ratio"]) / 6.0)
    kx = max(3, int(round(sx * 4)) | 1)
    ky = max(3, int(round(sy * 4)) | 1)
    out = cv2.GaussianBlur(out, (kx, ky), sigmaX=sx, sigmaY=sy)

    # Raster drift / row jitter (search only)
    if is_search:
        drift   = params["raster_drift_px"]
        jitter  = params["row_jitter_px"]
        if drift > 0 or jitter > 0:
            temp = np.zeros_like(out)
            for r in range(h):
                shift = (r / float(h)) * drift
                if jitter > 0:
                    shift += random.uniform(-jitter, jitter)
                M_row = np.float32([[1, 0, shift], [0, 1, 0]])
                temp[r:r+1, :] = cv2.warpAffine(out[r:r+1, :], M_row, (w, 1),
                                                  borderMode=cv2.BORDER_REFLECT)
            out = temp

    # Acquisition noise
    dose        = params["search_dose"] if is_search else params["ref_dose"]
    noise_scale = np.sqrt(2000.0 / max(dose, 10.0)) * 12.0
    gauss_noise = np.random.normal(0, noise_scale, out.shape)
    out = np.clip(out.astype(np.float32) + gauss_noise, 0, 255).astype(np.uint8)

    # Speckle
    sp = params["speckle_sigma"]
    if sp > 0.01:
        speckle = np.random.normal(1.0, sp * 0.15, out.shape)
        out = np.clip(out.astype(np.float32) * speckle, 0, 255).astype(np.uint8)

    # Salt & pepper
    spp = params["salt_pepper_prob"]
    if spp > 0.001:
        rnd = np.random.random(out.shape[:2])
        out[rnd < (spp / 2.0)] = 0
        out[rnd > (1.0 - spp / 2.0)] = 255

    out = apply_charging_streaks(out, params["charging_streaks"], params["streak_intensity"])
    out = apply_vignette(out, params["vignette_strength"])

    # Gamma
    g = params["gamma"]
    if abs(g - 1.0) > 0.02:
        table = np.array([((i / 255.0) ** (1.0 / g)) * 255 for i in range(256)]).astype("uint8")
        out = cv2.LUT(out, table)

    return out


def extract_gt_center(gt):
    """Robustly extract (cx, cy) in pixel space [0..1000]."""
    if "ground_truth" in gt and isinstance(gt["ground_truth"], dict):
        return float(gt["ground_truth"]["x"]) * 1000.0, float(gt["ground_truth"]["y"]) * 1000.0
    if "gt_x" in gt and "gt_y" in gt:
        return float(gt["gt_x"]), float(gt["gt_y"])
    if "target" in gt:
        t = gt["target"]
        if isinstance(t, dict):
            if "search_center_xy" in t:
                return float(t["search_center_xy"][0]), float(t["search_center_xy"][1])
            if "search_box_xywh" in t:
                b = t["search_box_xywh"]
                return float(b[0]) + float(b[2]) / 2.0, float(b[1]) + float(b[3]) / 2.0
            if "physical_origin_nm" in t:
                return float(t["physical_origin_nm"][0]) / 10.0 + 50.0, float(t["physical_origin_nm"][1]) / 10.0 + 50.0
    return 500.0, 500.0


def generate_single_sample(task):
    """Generates one augmented sample. Called in parallel."""
    sample_idx, split_name, base_folder, dataset_dir = task

    ref_img    = cv2.imread(str(base_folder / "reference_100x.png"), cv2.IMREAD_COLOR)
    search_img = cv2.imread(str(base_folder / "search_10x.png"),    cv2.IMREAD_COLOR)
    with open(base_folder / "ground_truth.json") as f:
        gt_json = json.load(f)

    cx, cy = extract_gt_center(gt_json)

    # Random SEM physics parameters
    params = {
        "beam_spot_nm":        random.uniform(1.0,   20.0),
        "astigmatism_ratio":   random.uniform(0.5,    2.0),
        "cd_bias_nm":          random.uniform(-10.0, 10.0),
        "ref_dose":            random.uniform(100.0,5000.0),
        "search_dose":         random.uniform(20.0, 2000.0),
        "raster_drift_px":     random.uniform(0.0,    5.0),
        "row_jitter_px":       random.uniform(0.0,    3.0),
        "barrel_distortion":   random.uniform(-0.15,  0.15),
        "vignette_strength":   random.uniform(0.0,    1.0),
        "gamma":               random.uniform(0.4,    2.5),
        "charging_streaks":    random.uniform(0.0,    5.0),
        "streak_intensity":    random.uniform(0.0,    3.0),
        "speckle_sigma":       random.uniform(0.0,    1.0),
        "salt_pepper_prob":    random.uniform(0.0,    0.05),
    }

    # Apply SEM physics augmentation
    ref_aug    = apply_sem_physics(ref_img,    is_search=False, params=params)
    search_aug = apply_sem_physics(search_img, is_search=True,  params=params)

    # Geometric augmentation on search image
    angle_deg = random.uniform(-5.0,  5.0)
    scale     = random.uniform( 0.95, 1.05)
    dx        = random.uniform(-15.0, 15.0)
    dy        = random.uniform(-15.0, 15.0)

    h, w = search_aug.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, scale)
    M[0, 2] += dx
    M[1, 2] += dy
    search_aug = cv2.warpAffine(search_aug, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Update ground truth with affine transform
    pt_t  = M @ np.array([cx, cy, 1.0])
    new_cx = float(np.clip(pt_t[0], 10.0, w - 10.0))
    new_cy = float(np.clip(pt_t[1], 10.0, h - 10.0))

    # Lens distortion
    k = params["barrel_distortion"]
    if abs(k) > 0.01:
        search_aug = apply_lens_distortion(search_aug, k)
        new_cx, new_cy = transform_point_lens_distortion([new_cx, new_cy], k, img_size=w)
        new_cx = float(np.clip(new_cx, 10.0, w - 10.0))
        new_cy = float(np.clip(new_cy, 10.0, h - 10.0))

    gt_x_norm = new_cx / float(w)
    gt_y_norm = new_cy / float(h)

    # Save output
    sample_dir = Path(dataset_dir) / split_name / f"dram_{sample_idx:05d}"
    sample_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(sample_dir / "reference_100x.png"), ref_aug)
    cv2.imwrite(str(sample_dir / "search_10x.png"),    search_aug)

    gt_out = {
        "architecture": "DRAM",
        "pair_id":      f"dram_{sample_idx:05d}",
        "base_pair":    base_folder.name,
        "split":        split_name,
        "ground_truth": {"x": round(gt_x_norm, 6), "y": round(gt_y_norm, 6)},
        "target": {
            "search_center_xy": [round(new_cx, 2), round(new_cy, 2)],
        },
        "params": {k_: round(v, 4) for k_, v in params.items()},
        "transform": {
            "rotation_deg": round(angle_deg, 2),
            "scale":        round(scale, 4),
            "shift_xy":     [round(dx, 2), round(dy, 2)]
        }
    }
    with open(sample_dir / "ground_truth.json", "w") as f:
        json.dump(gt_out, f, indent=4)

    return sample_idx


# ============================================================
# STEP 3 - Expand base pairs to requested total
# ============================================================

def expand_dataset(base_folders, total_samples, dataset_dir):
    """
    Augments base_folders to produce `total_samples` augmented pairs
    split into train/val/test according to TRAIN/VAL/TEST ratios.
    """
    train_count = int(math.floor(total_samples * TRAIN_RATIO))
    test_count  = int(math.floor(total_samples * TEST_RATIO))
    val_count   = total_samples - train_count - test_count

    splits = (["train"] * train_count + ["val"] * val_count + ["test"] * test_count)
    random.seed(42)
    random.shuffle(splits)

    # Create output directories
    for sp in ["train", "val", "test"]:
        (Path(dataset_dir) / sp).mkdir(parents=True, exist_ok=True)

    print(f"\nStep 2: Augmenting {len(base_folders)} base pairs to {total_samples} samples...")
    print(f"  Train: {train_count}  |  Val: {val_count}  |  Test: {test_count}")

    tasks = []
    for i in range(1, total_samples + 1):
        base_folder = base_folders[(i - 1) % len(base_folders)]
        tasks.append((i, splits[i - 1], base_folder, dataset_dir))

    start = time.time()
    max_workers = min(os.cpu_count() or 4, 8)
    print(f"  Using {max_workers} parallel workers...\n")

    completed = 0
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(generate_single_sample, t) for t in tasks]
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % max(1, total_samples // 20) == 0 or completed == total_samples:
                pct     = completed / total_samples * 100
                elapsed = time.time() - start
                rate    = completed / elapsed if elapsed > 0 else 0
                eta     = (total_samples - completed) / rate if rate > 0 else 0
                print(f"  [{completed:>6}/{total_samples}]  {pct:5.1f}%  |  "
                      f"{rate:.1f} samples/sec  |  ETA: {eta:.0f}s", flush=True)

    elapsed_total = time.time() - start
    print(f"\nAugmentation complete in {elapsed_total:.1f}s")
    return train_count, val_count, test_count


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 65)
    print("  DRIFT-SENSE AI - MASTER DATASET GENERATION PIPELINE")
    print("=" * 65)
    print(f"  Base generator scripts : {GEN_DIR}")
    print(f"  Base images output     : {BASE_DIR}")
    print(f"  Final dataset output   : {DATASET_DIR}")
    print(f"  Number of base scripts : {NUM_GENERATORS}")
    print(f"  Split ratios           : Train={int(TRAIN_RATIO*100)}%  "
          f"Val={int(VAL_RATIO*100)}%  Test={int(TEST_RATIO*100)}%")
    print("=" * 65)

    # --- Ask user for total sample count ---
    while True:
        try:
            raw = input(
                "\nHow many dataset samples do you want to generate?\n"
                f"  Min: 1   Max: unlimited   (76 = just run base scripts, no augmentation)\n"
                "  Enter count: "
            ).strip()
            total_samples = int(raw)
            if total_samples < 1:
                print("Please enter a positive number.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter an integer number.")

    print(f"\nGenerating {total_samples} samples...")

    # Step 1: Run the 76 base generators
    # If user wants fewer than 76, run only what's needed.
    generators_needed = min(total_samples, NUM_GENERATORS)
    base_folders = run_all_generators(generators_needed)

    if not base_folders:
        print("ERROR: No base image pairs were generated. Check your generator scripts.")
        sys.exit(1)

    # If user asked for exactly <= 76 samples, skip augmentation step.
    if total_samples <= len(base_folders):
        print(f"\nRequested {total_samples} samples which is <= {len(base_folders)} base pairs.")
        print("Copying base pairs directly into train/val/test splits...")

        selected_folders = base_folders[:total_samples]
        train_c = int(math.floor(total_samples * TRAIN_RATIO))
        test_c  = int(math.floor(total_samples * TEST_RATIO))
        val_c   = total_samples - train_c - test_c

        splits  = (["train"] * train_c + ["val"] * val_c + ["test"] * test_c)
        random.seed(42)
        random.shuffle(splits)

        import shutil
        for sp in ["train", "val", "test"]:
            (DATASET_DIR / sp).mkdir(parents=True, exist_ok=True)

        for idx, (folder, split_name) in enumerate(zip(selected_folders, splits), start=1):
            dest = DATASET_DIR / split_name / f"dram_{idx:05d}"
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(folder, dest)

        train_c = len(list((DATASET_DIR / "train").glob("dram_*")))
        val_c   = len(list((DATASET_DIR / "val").glob("dram_*")))
        test_c  = len(list((DATASET_DIR / "test").glob("dram_*")))
    else:
        # Step 2: Augment to reach the requested total
        train_c, val_c, test_c = expand_dataset(base_folders, total_samples, DATASET_DIR)

    # Final summary
    print("\n" + "=" * 65)
    print("  DATASET GENERATION COMPLETE")
    print("=" * 65)
    print(f"  Total samples : {total_samples}")
    print(f"  Train         : {len(list((DATASET_DIR / 'train').glob('dram_*')))}")
    print(f"  Val           : {len(list((DATASET_DIR / 'val').glob('dram_*')))}")
    print(f"  Test          : {len(list((DATASET_DIR / 'test').glob('dram_*')))}")
    print(f"  Saved to      : {DATASET_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    main()
