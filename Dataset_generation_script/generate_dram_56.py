import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# DRAM-56
# Serpentine Wordline + Hex Capacitor + Via Ladder
# Defect : Contact Size Variation
# ============================================================

SEED = 20260826
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_56"
SIZE = 10000
REF = 1000
DEFECT_X = 6500
DEFECT_Y = 3200

# ============================================================
# DRAW HELPERS
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


def circle(img, x, y, r, v):
    cv2.circle(
        img,
        (int(x), int(y)),
        int(r),
        int(v),
        -1,
        lineType=cv2.LINE_AA,
    )

# ============================================================
# PATTERN 1
# SERPENTINE WORDLINES
# ============================================================

def serpentine(img):
    for row, y in enumerate(range(500, 4200, 150)):
        pts = []
        for x in range(400, 3400, 40):
            yy = y + int(35 * np.sin(x / 180))
            pts.append((x, yy))
        for i in range(len(pts) - 1):
            line(
                img,
                pts[i][0],
                pts[i][1],
                pts[i + 1][0],
                pts[i + 1][1],
                130,
                5,
            )

# ============================================================
# PATTERN 2
# HEX CAPACITOR ARRAY
# ============================================================

def hex_array(img):
    pitch = 170
    for row, y in enumerate(range(700, 4500, pitch)):
        shift = pitch // 2 if row % 2 else 0
        for x in range(4000 + shift, 7000, pitch):
            circle(img, x, y, 20, 230)
            circle(img, x, y, 6, 60)

# ============================================================
# PATTERN 3
# VIA LADDER
# ============================================================

def via_ladder(img):
    for x in range(7600, 9500, 180):
        line(img, x, 700, x, 4700, 110, 7)
        for y in range(900, 4600, 200):
            circle(img, x, y, 14, 240)

# ============================================================
# DEFECT
# ============================================================

def contact_variation(img, x, y):
    circle(img, x, y, 45, 245)
    circle(img, x, y, 15, 80)
    for i in range(12):
        dx = rng.integers(-50, 50)
        dy = rng.integers(-50, 50)
        circle(img, x + dx, y + dy, 3, 120)

# ============================================================
# CREATE DIE
# ============================================================

def create_scene():
    img = np.full((SIZE, SIZE), 22, dtype=np.uint8)
    serpentine(img)
    hex_array(img)
    via_ladder(img)
    line(img, 3600, 200, 3600, 5000, 50, 12)
    line(img, 7300, 200, 7300, 5000, 50, 12)
    line(img, 200, 5200, 9700, 5200, 50, 12)
    contact_variation(img, DEFECT_X, DEFECT_Y)
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
    a = cv2.GaussianBlur(out, (9, 3), 1)
    b = cv2.GaussianBlur(out, (3, 9), 1)
    out = (a + b) / 2
    bands = np.sin(np.arange(img.shape[0]) / 35)[:, None] * 3
    out += bands
    out *= rng.normal(1, 0.02, img.shape)
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
        "pair": "dram_56",
        "architecture": "serpentine_hex_via_dram",
        "patterns": [
            "serpentine_wordline",
            "hexagonal_capacitor_array",
            "via_ladder",
        ],
        "defect": "contact_size_variation",
        "defect_position": [DEFECT_X, DEFECT_Y],
        "reference_origin": [rx, ry],
        "scale_ratio": 10,
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)
    print("DRAM-56 COMPLETE")


if __name__ == "__main__":
    main()
