import cv2
import json
import numpy as np
from pathlib import Path

# ============================================================
# DRAM-51
# Three Structure Mixed DRAM Die
# Defect: Local Bridge Formation
# ============================================================

SEED = 20260821
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_51"
SIZE = 10000
REF_SIZE = 1000
SEARCH_SIZE = 1000
DEFECT_X = 5200
DEFECT_Y = 4200

# ============================================================
# HELPERS
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


def circle(img, x, y, r, val):
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
# DENSE CONTACT ARRAY
# ============================================================

def contact_array(img):
    for row, y in enumerate(range(500, 4000, 100)):
        shift = 0
        if row % 2:
            shift = 50
        for x in range(500 + shift, 4000, 100):
            circle(img, x, y, 16, 230)
            circle(img, x, y, 5, 255)

# ============================================================
# STRUCTURE 2
# MEANDER BITLINES
# ============================================================

def meander_lines(img):
    for offset in range(500, 4200, 180):
        pts = []
        for y in range(500, 4200, 20):
            x = offset + int(60 * np.sin(y / 250))
            pts.append((x, y))
        for i in range(len(pts) - 1):
            line(
                img,
                pts[i][0],
                pts[i][1],
                pts[i + 1][0],
                pts[i + 1][1],
                120,
                6,
            )

# ============================================================
# STRUCTURE 3
# METAL GRID
# ============================================================

def metal_grid(img):
    for y in range(5500, 9500, 180):
        line(img, 5000, y, 9500, y, 90, 8)
    for x in range(5200, 9500, 180):
        line(img, x, 5300, x, 9500, 90, 8)

# ============================================================
# DEFECT
# ============================================================

def add_bridge(img, x, y):
    line(img, x - 120, y, x + 120, y, 180, 20)
    circle(img, x, y, 35, 200)

# ============================================================
# CREATE SCENE
# ============================================================

def create_scene():
    img = np.full((SIZE, SIZE), 20, dtype=np.uint8)
    line(img, 4500, 200, 4500, 9800, 45, 15)
    line(img, 200, 4800, 9800, 4800, 45, 15)
    contact_array(img)
    meander_lines(img)
    metal_grid(img)
    add_bridge(img, DEFECT_X, DEFECT_Y)
    return img

# ============================================================
# SEM NOISE
# ============================================================

def reference_sem(img):
    out = cv2.GaussianBlur(img, (3, 3), 0.3).astype(float)
    out += rng.normal(0, 1, img.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def search_sem(img):
    out = cv2.GaussianBlur(img, (5, 5), 1).astype(float)
    charge = cv2.GaussianBlur(rng.normal(0, 15, img.shape), (101, 101), 0)
    out += charge
    out += rng.normal(0, 2, img.shape[0])[:, None]
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
    reference = scene[ry : ry + REF_SIZE, rx : rx + REF_SIZE]
    search = cv2.resize(scene, (SEARCH_SIZE, SEARCH_SIZE), interpolation=cv2.INTER_AREA)
    reference = reference_sem(reference)
    search = search_sem(search)
    cv2.imwrite(str(OUT / "reference_100x.png"), reference)
    cv2.imwrite(str(OUT / "search_10x.png"), search)
    gt = {
        "pair": "dram_51",
        "architecture": "mixed_three_region_dram",
        "structures": [
            "dense_contact_array",
            "meander_bitlines",
            "metal_grid",
        ],
        "defect": "local_bridge",
        "reference_origin": [rx, ry],
        "scale_ratio": 10,
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)
    print("DRAM-51 COMPLETE")


if __name__ == "__main__":
    main()
