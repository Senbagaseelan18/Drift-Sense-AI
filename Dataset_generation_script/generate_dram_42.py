import cv2
import json
import numpy as np
from pathlib import Path


# ============================================================
# DRAM-42
# Diagonal Capacitor + Hex Contact + Metal Strap DRAM
# ============================================================


SEED = 20260812

rng = np.random.default_rng(SEED)


OUT = Path(__file__).resolve().parents[1] / 'results' / 'generated_dataset_images' / 'dram_42'


SIZE = 10000

REF_SIZE = 1000


# ============================================================
# DRAW HELPERS
# ============================================================


def line(img, x1, y1, x2, y2, val, width):
    cv2.line(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        int(val),
        int(width),
        lineType=cv2.LINE_AA
    )


def circle(img, x, y, r, val):
    cv2.circle(
        img,
        (int(x), int(y)),
        int(r),
        int(val),
        -1,
        lineType=cv2.LINE_AA
    )


# ============================================================
# REGION 1
# DIAGONAL CAPACITOR CHANNELS
# ============================================================


def diagonal_region(img):
    for offset in range(
        -2000,
        SIZE + 2000,
        90
    ):
        line(
            img,
            offset,
            0,
            offset + 3500,
            SIZE,
            90,
            5
        )

        for y in range(
            300,
            SIZE - 300,
            150
        ):
            x = offset + int(y * 0.35)
            if 0 < x < SIZE:
                circle(
                    img,
                    x,
                    y,
                    14,
                    220
                )


# ============================================================
# REGION 2
# HEX CONTACT ARRAY
# ============================================================


def hex_region(img, x0, y0, w, h):
    pitch = 120
    row = 0

    for y in range(
        y0,
        y0 + h,
        pitch
    ):
        shift = 0
        if row % 2:
            shift = pitch // 2

        for x in range(
            x0 + shift,
            x0 + w,
            pitch
        ):
            circle(
                img,
                x,
                y,
                18,
                235
            )
            circle(
                img,
                x,
                y,
                5,
                255
            )

        row += 1


# ============================================================
# METAL STRAPS
# ============================================================


def metal_region(img):
    for y in range(
        700,
        SIZE,
        800
    ):
        line(
            img,
            200,
            y,
            SIZE - 200,
            y,
            60,
            18
        )


# ============================================================
# CREATE SCENE
# ============================================================


def create_scene():
    img = np.full(
        (
            SIZE,
            SIZE
        ),
        18,
        dtype=np.uint8
    )
    diagonal_region(img)
    hex_region(
        img,
        4200,
        1200,
        4500,
        4500
    )
    metal_region(img)
    return img


# ============================================================
# NEW SEM PHYSICS
# ============================================================


def reference_sem(img):
    out = cv2.GaussianBlur(
        img,
        (3, 3),
        0.25
    )
    out = out.astype(float)
    out += rng.normal(
        0,
        0.7,
        img.shape
    )
    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


def search_sem(img):
    out = img.astype(float)
    sigma = rng.uniform(
        0.8,
        1.5
    )
    out = cv2.GaussianBlur(
        out,
        (5, 5),
        sigma
    ).astype(float)
    gain = rng.normal(
        1.0,
        0.015,
        img.shape
    )
    out *= gain
    patch = rng.normal(
        0,
        2,
        (
            20,
            20
        )
    )
    patch = cv2.resize(
        patch.astype(np.float32),
        (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_CUBIC
    )
    out += patch
    out += rng.normal(
        0,
        2,
        img.shape
    )
    return np.clip(
        out,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# MAIN
# ============================================================


def main():
    print("=" * 60)
    print("DRIFT-SENSE DRAM-42")
    print("=" * 60)

    OUT.mkdir(
        parents=True,
        exist_ok=True
    )

    scene = create_scene()

    ref_x = 3900
    ref_y = 2500

    reference = scene[
        ref_y:
        ref_y + REF_SIZE,
        ref_x:
        ref_x + REF_SIZE
    ]

    search = cv2.resize(
        scene,
        (
            1000,
            1000
        ),
        interpolation=cv2.INTER_AREA
    )

    reference = reference_sem(reference)
    search = search_sem(search)

    cv2.imwrite(
        str(
            OUT / "reference_100x.png"
        ),
        reference
    )

    cv2.imwrite(
        str(
            OUT / "search_10x.png"
        ),
        search
    )

    gt = {
        "pair": "dram_42",
        "architecture": "diagonal_capacitor_hex_contact_metal_stack",
        "reference_origin_nm": [
            ref_x,
            ref_y
        ],
        "scale_ratio": 10
    }

    with open(
        OUT / "ground_truth.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            gt,
            f,
            indent=4
        )

    print("DRAM-42 COMPLETE")


if __name__ == "__main__":
    main()
