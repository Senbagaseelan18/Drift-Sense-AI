import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-47
# Cross-point DRAM with broken wordline
# ============================================================


SEED = 20260817
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_47"
SIZE = 10000
REF_SIZE = 1000
SEARCH_SIZE = 1000

BROKEN_Y = 5200
BROKEN_X0 = 1200
BROKEN_X1 = 2600


def draw_line(img, x1, y1, x2, y2, value, width):
    cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), int(value), int(width), lineType=cv2.LINE_AA)


def draw_dot(img, x, y, value):
    cv2.circle(img, (int(x), int(y)), 4, int(value), -1, lineType=cv2.LINE_AA)


def create_scene():
    img = np.full((SIZE, SIZE), 20, dtype=np.uint8)
    for y in range(400, SIZE - 400, 200):
        draw_line(img, 300, y, SIZE - 300, y, 90, 5)
    for x in range(300, SIZE - 300, 200):
        draw_line(img, x, 300, x, SIZE - 300, 90, 5)
    for y in range(400, SIZE - 400, 200):
        for x in range(300, SIZE - 300, 200):
            draw_dot(img, x, y, 230)
    draw_line(img, BROKEN_X0, BROKEN_Y, BROKEN_X1, BROKEN_Y, 20, 12)
    return img


def apply_sem_noise(img, mode="search"):
    out = img.astype(float)
    if mode == "reference":
        out = cv2.GaussianBlur(out, (3, 3), 0.4)
        out += rng.normal(0, 1.2, out.shape)
    else:
        out = cv2.GaussianBlur(out, (5, 5), 1.0)
        drift = np.linspace(-2, 2, out.shape[0])
        out += drift[:, None]
        jitter = rng.normal(0, 1.8, (out.shape[0], 1))
        out += jitter
        out += rng.normal(0, 3.2, out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


def create_reference(scene):
    rx = BROKEN_X0 - 300
    ry = BROKEN_Y - 450
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
        "pair": "dram_47",
        "architecture": "cross_point_dram_architecture",
        "reference_origin_nm": [origin[0], origin[1]],
        "scale_ratio": 10,
        "defect": "broken_wordline"
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)


def main():
    scene = create_scene()
    reference, origin = create_reference(scene)
    search = create_search(scene)
    save_outputs(reference, search, origin)
    print("DRAM-47 COMPLETE")


if __name__ == "__main__":
    main()
