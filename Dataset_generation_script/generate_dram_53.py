import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# DRAM-53
# Single Full Die
# Three Pattern Architecture
# Defect : Local Line Interruption
# ============================================================

SEED = 20260823
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_53"
SIZE = 10000
REF_SIZE = 1000
DEFECT_X = 6200
DEFECT_Y = 4200

# ============================================================
# DRAW FUNCTIONS
# ============================================================

def draw_line(img, x1, y1, x2, y2, val, width):
    cv2.line(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        int(val),
        int(width),
        lineType=cv2.LINE_AA,
    )


def draw_circle(img, x, y, r, val):
    cv2.circle(
        img,
        (int(x), int(y)),
        int(r),
        int(val),
        -1,
        lineType=cv2.LINE_AA,
    )

# ============================================================
# PATTERN 1
# DENSE FIN ARRAY
# ============================================================

def fin_array(img):
    for x in range(500, 3300, 70):
        draw_line(img, x, 500, x, 4700, 120, 5)
        for y in range(700, 4600, 130):
            draw_circle(img, x, y, 10, 230)

# ============================================================
# PATTERN 2
# CAPACITOR ARRAY
# ============================================================

def capacitor_array(img):
    for row, y in enumerate(range(600, 4700, 130)):
        offset = 65 if row % 2 else 0
        for x in range(4200 + offset, 7200, 130):
            draw_circle(img, x, y, 18, 240)
            draw_circle(img, x, y, 5, 60)

# ============================================================
# PATTERN 3
# METAL GRID
# ============================================================

def metal_grid(img):
    for y in range(5400, 9500, 150):
        draw_line(img, 700, y, 9300, y, 100, 8)
    for x in range(700, 9300, 150):
        draw_line(img, x, 5400, x, 9500, 100, 8)

# ============================================================
# DEFECT
# ============================================================

def add_line_break(img, x, y):
    cv2.rectangle(img, (x - 80, y - 20), (x + 80, y + 20), 20, -1)
    for i in range(15):
        dx = rng.integers(-100, 100)
        dy = rng.integers(-50, 50)
        draw_circle(img, x + dx, y + dy, 3, 80)

# ============================================================
# CREATE DIE
# ============================================================

def create_die():
    img = np.full((SIZE, SIZE), 25, dtype=np.uint8)
    fin_array(img)
    capacitor_array(img)
    metal_grid(img)
    draw_line(img, 3700, 200, 3700, 5000, 60, 12)
    draw_line(img, 200, 5100, 9800, 5100, 60, 12)
    add_line_break(img, DEFECT_X, DEFECT_Y)
    return img

# ============================================================
# SEM EFFECTS
# ============================================================

def reference_sem(img):
    out = cv2.GaussianBlur(img, (3, 3), 0.3).astype(float)
    out += rng.normal(0, 1, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def search_sem(img):
    out = cv2.GaussianBlur(img, (5, 5), 1).astype(float)
    field = cv2.GaussianBlur(rng.normal(0, 15, img.shape), (121, 121), 0)
    out += field
    out *= rng.normal(1, 0.02, img.shape)
    out += rng.normal(0, 2, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)

# ============================================================
# MAIN
# ============================================================

def main():
    print("DRAM-53 GENERATION")
    OUT.mkdir(parents=True, exist_ok=True)
    scene = create_die()
    rx = DEFECT_X - 500
    ry = DEFECT_Y - 500
    reference = scene[ry : ry + REF_SIZE, rx : rx + REF_SIZE]
    search = cv2.resize(scene, (1000, 1000), interpolation=cv2.INTER_AREA)
    reference = reference_sem(reference)
    search = search_sem(search)
    cv2.imwrite(str(OUT / "reference_100x.png"), reference)
    cv2.imwrite(str(OUT / "search_10x.png"), search)
    gt = {
        "pair": "dram_53",
        "architecture": "single_die_three_pattern_dram",
        "patterns": [
            "dense_fin_array",
            "capacitor_contact_matrix",
            "metal_routing_mesh",
        ],
        "defect": "local_line_interruption",
        "defect_position": [DEFECT_X, DEFECT_Y],
        "reference_origin": [rx, ry],
        "scale_ratio": 10,
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)
    print("DRAM-53 COMPLETE")


if __name__ == "__main__":
    main()
