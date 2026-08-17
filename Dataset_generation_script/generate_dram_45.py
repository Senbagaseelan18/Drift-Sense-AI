import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-45
# Dual-layer folded bitline DRAM with missing capacitor contact cluster
# ============================================================


SEED = 20260815
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_45"
SIZE = 10000
REF_SIZE = 1000
SEARCH_SIZE = 1000

TARGET = (5300, 4600)
CLUSTER_SIZE = 220


def draw_line(img, x1, y1, x2, y2, value, width):
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), int(value), int(width), lineType=cv2.LINE_AA)


def draw_circle(img, x, y, r, value):
    cv2.circle(img, (int(x), int(y)), int(r), int(value), -1, lineType=cv2.LINE_AA)


def create_scene():
    img = np.full((SIZE, SIZE), 20, dtype=np.uint8)

    for layer in range(2):
        offset = layer * 35
        for x in range(300 + offset, SIZE - 300, 140):
            draw_line(img, x, 200, x, SIZE - 200, 90, 6)
            for y in range(400, SIZE - 400, 120):
                if layer == 1:
                    draw_circle(img, x + 45, y + 30, 16, 230)
                    draw_circle(img, x + 45, y + 30, 4, 255)
                else:
                    draw_circle(img, x - 45, y - 30, 16, 230)
                    draw_circle(img, x - 45, y - 30, 4, 255)

    for y in range(250, SIZE - 250, 500):
        draw_line(img, 250, y, SIZE - 250, y, 100, 8)

    cx, cy = TARGET
    mask_x = range(cx - CLUSTER_SIZE // 2, cx + CLUSTER_SIZE // 2, 140)
    mask_y = range(cy - CLUSTER_SIZE // 2, cy + CLUSTER_SIZE // 2, 120)
    for mx in mask_x:
        for my in mask_y:
            cv2.circle(img, (mx, my), 18, 20, -1)

    return img


def apply_sem_noise(img, mode="search"):
    out = img.astype(float)
    if mode == "reference":
        out = cv2.GaussianBlur(out, (3, 3), 0.3)
        out += rng.normal(0, 0.8, out.shape)
        out *= 1 + rng.normal(0, 0.005, out.shape)
    else:
        out = cv2.GaussianBlur(out, (5, 5), 0.9)
        line_noise = rng.normal(0, 6, (out.shape[0], 1))
        out += line_noise
        out += rng.normal(0, 3.5, out.shape)
        patch = rng.normal(0, 5, (50, 50)).astype(np.float32)
        patch = cv2.resize(patch, (out.shape[1], out.shape[0]), interpolation=cv2.INTER_CUBIC)
        out += patch
        out *= 1 + rng.normal(0, 0.02, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def create_reference(scene):
    rx = TARGET[0] - 400
    ry = TARGET[1] - 400
    ref = scene[ry:ry + REF_SIZE, rx:rx + REF_SIZE]
    return apply_sem_noise(ref, mode="reference"), (rx, ry)


def create_search(scene):
    search = cv2.resize(scene, (SEARCH_SIZE, SEARCH_SIZE), interpolation=cv2.INTER_AREA)
    return apply_sem_noise(search, mode="search")


def save_outputs(reference, search, origin):
    OUT.mkdir(parents=True, exist_ok=True)
    assert reference.shape == (REF_SIZE, REF_SIZE)
    assert search.shape == (SEARCH_SIZE, SEARCH_SIZE)
    cv2.imwrite(str(OUT / "reference_100x.png"), reference)
    cv2.imwrite(str(OUT / "search_10x.png"), search)
    gt = {
        "pair": "dram_45",
        "architecture": "dual_layer_folded_bitline_dram",
        "reference_origin_nm": [origin[0], origin[1]],
        "scale_ratio": 10,
        "defect": "missing_capacitor_cluster"
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)


def main():
    scene = create_scene()
    reference, origin = create_reference(scene)
    search = create_search(scene)
    save_outputs(reference, search, origin)
    print("DRAM-45 COMPLETE")


if __name__ == "__main__":
    main()
