import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# DRAM-52
# Folded Bitline + Staircase Wordline + Via Array
# Defect: Missing Via
# ============================================================

SEED = 20260822
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_52"
SIZE = 10000
REF = 1000
DEFECT_X = 7200
DEFECT_Y = 3800

# ============================================================
# HELPERS
# ============================================================

def line(img, x1, y1, x2, y2, v, w):
    cv2.line(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        int(v),
        int(w),
        lineType=cv2.LINE_AA,
    )


def dot(img, x, y, r, v):
    cv2.circle(
        img,
        (int(x), int(y)),
        int(r),
        int(v),
        -1,
        lineType=cv2.LINE_AA,
    )

# ============================================================
# STRUCTURE 1
# FOLDED BITLINE ARRAY
# ============================================================

def folded_array(img):
    for x in range(500, 4000, 90):
        line(img, x, 500, x, 4500, 120, 5)
    for row, y in enumerate(range(700, 4300, 140)):
        shift = 40 if row % 2 else 0
        for x in range(600 + shift, 4000, 180):
            dot(img, x, y, 12, 230)

# ============================================================
# STRUCTURE 2
# STAIRCASE WORDLINES
# ============================================================

def staircase_region(img):
    y = 800
    for i in range(8):
        line(
            img,
            4700 + i * 120,
            y + i * 250,
            6500 + i * 120,
            y + i * 250,
            150,
            10,
        )
        line(
            img,
            6500 + i * 120,
            y + i * 250,
            6500 + i * 120,
            y + i * 250 + 150,
            90,
            8,
        )

# ============================================================
# STRUCTURE 3
# VIA ARRAY
# ============================================================

def via_array(img):
    for y in range(700, 4500, 120):
        for x in range(7000, 9300, 120):
            dot(img, x, y, 18, 240)
            dot(img, x, y, 6, 40)

# ============================================================
# STRUCTURE 4
# PERIPHERAL EDGE
# ============================================================

def peripheral(img):
    for x in range(800, 9200, 180):
        line(img, x, 7000, x, 9500, 110, 8)
    for y in range(7200, 9500, 200):
        line(img, 700, y, 9500, y, 90, 8)

# ============================================================
# DEFECT
# ============================================================

def missing_via(img, x, y):
    cv2.circle(img, (x, y), 30, 20, -1)
    for i in range(15):
        dx = rng.integers(-40, 40)
        dy = rng.integers(-40, 40)
        dot(img, x + dx, y + dy, 3, 90)

# ============================================================
# SCENE
# ============================================================

def create_scene():
    img = np.full((SIZE, SIZE), 22, dtype=np.uint8)
    line(img, 4500, 200, 4500, 6500, 50, 12)
    line(img, 200, 6500, 9800, 6500, 50, 12)
    folded_array(img)
    staircase_region(img)
    via_array(img)
    peripheral(img)
    missing_via(img, DEFECT_X, DEFECT_Y)
    return img

# ============================================================
# SEM PHYSICS
# ============================================================

def reference_sem(img):
    out = cv2.GaussianBlur(img, (3, 3), 0.3).astype(float)
    out += rng.normal(0, 1, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def search_sem(img):
    out = cv2.GaussianBlur(img, (5, 5), 1).astype(float)
    gradient = np.linspace(-8, 8, img.shape[1])
    out += gradient
    charge = cv2.GaussianBlur(rng.normal(0, 15, img.shape), (121, 121), 0)
    out += charge
    out += rng.normal(0, 2, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene = create_scene()
    rx = DEFECT_X - 500
    ry = DEFECT_Y - 500
    reference = scene[ry : ry + REF, rx : rx + REF]
    search = cv2.resize(scene, (1000, 1000), interpolation=cv2.INTER_AREA)
    reference = reference_sem(reference)
    search = search_sem(search)
    cv2.imwrite(str(OUT / "reference_100x.png"), reference)
    cv2.imwrite(str(OUT / "search_10x.png"), search)
    gt = {
        "pair": "dram_52",
        "architecture": "folded_bitline_staircase_via_dram",
        "structures": [
            "folded_bitline_array",
            "staircase_wordline",
            "via_array",
            "peripheral_driver",
        ],
        "defect": "missing_via",
        "defect_position": [DEFECT_X, DEFECT_Y],
        "reference_origin": [rx, ry],
        "scale_ratio": 10,
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)
    print("DRAM-52 COMPLETE")


if __name__ == "__main__":
    main()
