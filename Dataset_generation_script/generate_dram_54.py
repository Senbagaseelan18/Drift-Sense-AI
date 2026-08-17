import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# DRAM-54
# Fishbone Bitline + Spiral Capacitor + Metal Stack
# Defect : Double Contact Merge
# ============================================================

SEED = 20260824
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_54"
SIZE = 10000
REF_SIZE = 1000
DEFECT_X = 5400
DEFECT_Y = 3600

# ============================================================
# DRAW HELPERS
# ============================================================

def line(img, x1, y1, x2, y2, val, w):
    cv2.line(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        int(val),
        int(w),
        lineType=cv2.LINE_AA,
    )


def dot(img, x, y, r, val):
    cv2.circle(
        img,
        (int(x), int(y)),
        int(r),
        int(val),
        -1,
        lineType=cv2.LINE_AA,
    )

# ============================================================
# STRUCTURE 1
# FISHBONE BITLINES
# ============================================================

def fishbone_array(img):
    for y in range(500, 4300, 160):
        for x in range(400, 4200, 200):
            line(img, x, y, x + 100, y + 80, 120, 5)
            line(img, x + 100, y + 80, x + 200, y, 120, 5)
            dot(img, x + 100, y + 80, 12, 230)

# ============================================================
# STRUCTURE 2
# SPIRAL CAPACITOR ARRAY
# ============================================================

def spiral_capacitors(img):
    for cy in range(700, 4500, 350):
        for cx in range(5000, 8500, 350):
            cv2.circle(img, (cx, cy), 45, 220, 8)
            cv2.circle(img, (cx, cy), 15, 250, -1)
            line(img, cx, cy, cx + 55, cy + 55, 150, 4)

# ============================================================
# STRUCTURE 3
# METAL STACK
# ============================================================

def metal_stack(img):
    for y in range(5200, 9300, 180):
        line(img, 500, y, 9500, y, 90, 10)
    for x in range(700, 9500, 220):
        line(img, x, 5400, x, 9500, 70, 6)

# ============================================================
# DEFECT
# ============================================================

def add_double_merge(img, x, y):
    dot(img, x - 60, y, 22, 230)
    dot(img, x + 60, y, 22, 230)
    line(img, x - 60, y, x + 60, y, 230, 14)

# ============================================================
# CREATE SCENE
# ============================================================

def create_scene():
    img = np.full((SIZE, SIZE), 25, dtype=np.uint8)
    fishbone_array(img)
    spiral_capacitors(img)
    metal_stack(img)
    line(img, 4500, 200, 4500, 5000, 50, 15)
    line(img, 200, 5000, 9800, 5000, 50, 15)
    add_double_merge(img, DEFECT_X, DEFECT_Y)
    return img

# ============================================================
# SEM PHYSICS
# ============================================================

def reference_sem(img):
    out = cv2.GaussianBlur(img, (3, 3), 0.25).astype(float)
    out += rng.normal(0, 1, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def search_sem(img):
    out = img.astype(float)
    blur1 = cv2.GaussianBlur(out, (9, 3), 1)
    blur2 = cv2.GaussianBlur(out, (3, 9), 1)
    out = (blur1 + blur2) / 2
    charge = cv2.GaussianBlur(rng.normal(0, 18, img.shape), (151, 151), 0)
    out += charge
    out *= rng.normal(1, 0.025, img.shape)
    mask = rng.random(img.shape) < 0.003
    out[mask] = 255
    mask2 = rng.random(img.shape) < 0.003
    out[mask2] = 0
    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# MAIN
# ============================================================

def main():
    print("DRAM-54 GENERATION")
    OUT.mkdir(parents=True, exist_ok=True)
    scene = create_scene()
    rx = DEFECT_X - 500
    ry = DEFECT_Y - 500
    reference = scene[ry : ry + REF_SIZE, rx : rx + REF_SIZE]
    search = cv2.resize(scene, (1000, 1000), interpolation=cv2.INTER_AREA)
    reference = reference_sem(reference)
    search = search_sem(search)
    cv2.imwrite(str(OUT / "reference_100x.png"), reference)
    cv2.imwrite(str(OUT / "search_10x.png"), search)
    gt = {
        "pair": "dram_54",
        "architecture": "fishbone_spiral_metal_dram",
        "structures": [
            "fishbone_bitline",
            "spiral_capacitor",
            "metal_stack",
        ],
        "defect": "double_contact_merge",
        "defect_position": [DEFECT_X, DEFECT_Y],
        "reference_origin": [rx, ry],
        "scale_ratio": 10,
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)
    print("DRAM-54 COMPLETE")


if __name__ == "__main__":
    main()
