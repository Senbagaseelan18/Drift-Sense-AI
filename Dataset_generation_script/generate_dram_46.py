import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-46
# Hexagonal capacitor DRAM array with one shifted contact
# ============================================================


SEED = 20260816
rng = np.random.default_rng(SEED)
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_46"
SIZE = 10000
REF_SIZE = 1000
SEARCH_SIZE = 1000

TARGET = (4800, 5200)
SHIFTED_CONTACT = (TARGET[0] + 40, TARGET[1] + 20)


def draw_hex_circle(img, x, y, r, value):
    cv2.circle(img, (int(x), int(y)), int(r), int(value), -1, lineType=cv2.LINE_AA)


def create_scene():
    img = np.full((SIZE, SIZE), 18, dtype=np.uint8)
    centers = [
        (1800, 1800, 90),
        (5200, 1800, 70),
        (8200, 1800, 50),
        (1800, 5200, 85),
        (5200, 5200, 65),
        (8200, 5200, 55)
    ]
    for cx, cy, pitch in centers:
        for row in range(-8, 9):
            for col in range(-8, 9):
                x = cx + col * pitch + (row % 2) * (pitch // 2)
                y = cy + row * int(pitch * 0.87)
                if 300 < x < SIZE - 300 and 300 < y < SIZE - 300:
                    draw_hex_circle(img, x, y, 18, 220)
                    draw_hex_circle(img, x, y, 5, 250)
    cv2.circle(img, (TARGET[0], TARGET[1]), 22, 18, -1)
    draw_hex_circle(img, SHIFTED_CONTACT[0], SHIFTED_CONTACT[1], 18, 235)
    draw_hex_circle(img, SHIFTED_CONTACT[0], SHIFTED_CONTACT[1], 5, 255)
    return img


def apply_sem_noise(img, mode="search"):
    out = img.astype(float)
    if mode == "reference":
        out = cv2.GaussianBlur(out, (3, 3), 0.35)
        out += rng.normal(0, 1.0, out.shape)
    else:
        out = cv2.GaussianBlur(out, (5, 5), 1.1)
        gain = rng.normal(1.0, 0.03, out.shape)
        out *= gain
        cloud = rng.normal(0, 8, out.shape)
        cloud = cv2.GaussianBlur(cloud.astype(np.float32), (101, 101), 0)
        out += cloud
        line_shift = rng.normal(0, 1.2, (out.shape[0], 1))
        out += line_shift
    return np.clip(out, 0, 255).astype(np.uint8)


def create_reference(scene):
    rx = TARGET[0] - 420
    ry = TARGET[1] - 420
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
        "pair": "dram_46",
        "architecture": "hexagonal_capacitor_dram_array",
        "reference_origin_nm": [origin[0], origin[1]],
        "scale_ratio": 10,
        "defect": "shifted_capacitor_contact"
    }
    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=4)


def main():
    scene = create_scene()
    reference, origin = create_reference(scene)
    search = create_search(scene)
    save_outputs(reference, search, origin)
    print("DRAM-46 COMPLETE")


if __name__ == "__main__":
    main()
