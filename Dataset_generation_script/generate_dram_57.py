import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# DRAM-57
# Single Fully Occupied DRAM Die
# 3 Dark Process Structures
# ============================================================

SEED = 20260827
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_57"
SIZE = 10000
REF = 1000
TARGET = (5200, 4300)

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
# DRAM ARRAY
# ============================================================

def create_memory_array(img):
    pitch = 110
    for row, y in enumerate(range(400, 9500, pitch)):
        offset = 55 if row % 2 else 0
        for x in range(400 + offset, 9500, pitch):
            dot(img, x, y, 14, 230)
            dot(img, x, y, 5, 70)
    for y in range(500, 9500, 220):
        line(img, 300, y, 9700, y, 100, 5)
    for x in range(500, 9500, 220):
        line(img, x, 300, x, 9700, 90, 4)

# ============================================================
# DARK STRUCTURES
# ============================================================

def add_dark_features(img):
    cv2.rectangle(img, (1800, 2300), (2500, 2800), 15, -1)
    cv2.rectangle(img, (4300, 5200), (6000, 5260), 10, -1)
    cv2.circle(img, (7600, 3000), 120, 12, -1)

# ============================================================
# SCENE
# ============================================================

def create_scene():
    img = np.full((SIZE, SIZE), 35, dtype=np.uint8)
    create_memory_array(img)
    add_dark_features(img)
    return img

# ============================================================
# SEM NOISE
# ============================================================

def reference_sem(img):
    out = cv2.GaussianBlur(img, (3, 3), 0.25).astype(float)
    out += rng.normal(0, 1, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def search_sem(img):
    out = cv2.GaussianBlur(img, (5, 5), 1).astype(float)
    charge = cv2.GaussianBlur(rng.normal(0, 18, img.shape), (151, 151), 0)
    out += charge
    out += np.sin(np.arange(img.shape[0]) / 25)[:, None] * 2
    out *= rng.normal(1, 0.03, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# MAIN
# ============================================================

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    scene = create_scene()
    rx = TARGET[0] - 500
    ry = TARGET[1] - 500
    reference = scene[ry : ry + REF, rx : rx + REF]
    search = cv2.resize(scene, (1000, 1000), interpolation=cv2.INTER_AREA)
    reference = reference_sem(reference)
    search = search_sem(search)
    cv2.imwrite(str(OUT / "reference_100x.png"), reference)
    cv2.imwrite(str(OUT / "search_10x.png"), search)
    gt = {
        "pair": "dram_57",
        "architecture": "single_full_die_memory_array",
        "patterns": [
            "capacitor_contacts",
            "wordlines",
            "bitlines",
        ],
        "dark_structures": 3,
        "target": [TARGET[0], TARGET[1]],
        "reference_origin": [rx, ry],
        "scale_ratio": 10,
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)
    print("DRAM-57 COMPLETE")


if __name__ == "__main__":
    main()
