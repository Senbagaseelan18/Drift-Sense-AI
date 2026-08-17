import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# DRAM-55
# Multi density DRAM SEM mosaic
# Defect: missing capacitor contact
# ============================================================

SEED = 20260825
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_55"
SIZE = 12000
DIE = 2600
GAP = 450
REF = 1000
TARGET_DIE = (1, 2)

# ============================================================
# DRAW
# ============================================================

def line(img, x1, y1, x2, y2, v, w):
    cv2.line(
        img,
        (x1, y1),
        (x2, y2),
        int(v),
        int(w),
        lineType=cv2.LINE_AA,
    )


def dot(img, x, y, r, v):
    cv2.circle(
        img,
        (x, y),
        r,
        int(v),
        -1,
        lineType=cv2.LINE_AA,
    )

# ============================================================
# DRAM DIE PATTERNS
# ============================================================

def dense_die(img, x, y, pitch):
    for row, yy in enumerate(range(y + 200, y + DIE - 200, pitch)):
        offset = pitch // 2 if row % 2 else 0
        for xx in range(x + 200 + offset, x + DIE - 200, pitch):
            dot(img, xx, yy, 9, 220)
            dot(img, xx, yy, 3, 255)


def add_die_frame(img, x, y):
    cv2.rectangle(
        img,
        (x, y),
        (x + DIE, y + DIE),
        55,
        15,
    )

# ============================================================
# CREATE MOSAIC
# ============================================================

def create_scene():
    img = np.full((SIZE, SIZE), 20, dtype=np.uint8)
    pitches = [130, 95, 65, 110]
    positions = []
    idx = 0
    for r in range(4):
        for c in range(4):
            x = 300 + c * (DIE + GAP)
            y = 300 + r * (DIE + GAP)
            add_die_frame(img, x, y)
            dense_die(img, x, y, pitches[idx % 4])
            positions.append((x, y))
            idx += 1
    tx, ty = positions[TARGET_DIE[0] * 4 + TARGET_DIE[1]]
    dx = tx + 1300
    dy = ty + 1200
    cv2.circle(img, (dx, dy), 18, 20, -1)
    return img, dx, dy

# ============================================================
# SEM NOISE
# ============================================================

def reference_sem(img):
    out = cv2.GaussianBlur(img, (3, 3), 0.3).astype(float)
    out += rng.normal(0, 1, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def search_sem(img):
    out = cv2.GaussianBlur(img, (5, 5), 1).astype(float)
    field = cv2.resize(
        rng.normal(0, 15, (100, 100)),
        img.shape[::-1],
        interpolation=cv2.INTER_CUBIC,
    )
    out += field
    out += np.sin(np.arange(img.shape[0]) / 18)[:, None] * 2
    out *= rng.normal(1, 0.025, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene, dx, dy = create_scene()
    rx = dx - 500
    ry = dy - 500
    reference = scene[ry : ry + REF, rx : rx + REF]
    search = cv2.resize(scene, (1000, 1000), interpolation=cv2.INTER_AREA)
    reference = reference_sem(reference)
    search = search_sem(search)
    cv2.imwrite(str(OUT / "reference_100x.png"), reference)
    cv2.imwrite(str(OUT / "search_10x.png"), search)
    gt = {
        "pair": "dram_55",
        "architecture": "multi_density_dram_die_mosaic",
        "defect": "missing_capacitor_contact",
        "defect_position": [dx, dy],
        "reference_origin": [rx, ry],
        "scale_ratio": 10,
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)
    print("DRAM-55 COMPLETE")


if __name__ == "__main__":
    main()
