#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-01 Synthetic Pair Generator
=================================

Generates exactly ONE DRAM reference/search pair.

OUTPUT
------

results/
└── dram_01/
    ├── reference_100x.png
    ├── search_10x.png
    └── ground_truth.json


PHYSICAL MODEL
--------------

Reference:
    1000 x 1000 pixels
    1 nm/pixel
    1 um x 1 um FOV

Search:
    1000 x 1000 pixels
    10 nm/pixel
    10 um x 10 um FOV

The reference and search originate from the SAME
10 um x 10 um physical scene.

The reference is a 1 um crop from the physical scene.

The search is generated from the complete physical scene
using area downsampling.

Therefore:

    100x reference
          |
          | physically contained
          v
    10x search

The target becomes exactly 100 x 100 pixels
inside the 10x search image.
"""

from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 1000

REFERENCE_PIXEL_SIZE_NM = 1
SEARCH_PIXEL_SIZE_NM = 10

SCALE_FACTOR = 10

FINE_SIZE = IMAGE_SIZE * SCALE_FACTOR

SEED = 20260810


# ------------------------------------------------------------
# DRAM-01 target location
# ------------------------------------------------------------
#
# This is the origin of the 1 um x 1 um reference crop
# inside the 10 um x 10 um physical scene.
#
# Because the search is 10 nm/pixel, these coordinates
# must be multiples of 10 nm.

TARGET_X_NM = 2680
TARGET_Y_NM = 4990

REFERENCE_SIZE_NM = 1000


# ------------------------------------------------------------
# Output
# ------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_01"
)


# ============================================================
# DRAWING FUNCTIONS
# ============================================================

def draw_line(
    image,
    x1,
    y1,
    x2,
    y2,
    intensity,
    width,
):
    """
    Draw an anti-aliased semiconductor line.
    """

    cv2.line(
        image,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        int(intensity),
        int(width),
        lineType=cv2.LINE_AA,
    )


def draw_contact(
    image,
    x,
    y,
    radius,
    intensity,
):
    """
    Draw a circular DRAM contact.
    """

    radius = max(
        1,
        int(round(radius)),
    )

    cv2.circle(
        image,
        (
            int(round(x)),
            int(round(y)),
        ),
        radius,
        int(intensity),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Inner contact contrast.
    inner_radius = max(
        1,
        int(round(radius * 0.5)),
    )

    cv2.circle(
        image,
        (
            int(round(x)),
            int(round(y)),
        ),
        inner_radius,
        min(255, int(intensity + 15)),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# DRAM ARRAY
# ============================================================

def draw_dram_array(
    canvas,
    x0,
    y0,
    block_size,
    rng,
):
    """
    Generate one repeated DRAM memory-array block.

    Structure:

        vertical bit lines
        horizontal word lines
        repeated contacts
        array boundary
    """

    # --------------------------------------------------------
    # Array geometry
    # --------------------------------------------------------

    border = 180

    ax0 = x0 + border
    ay0 = y0 + border

    ax1 = x0 + block_size - border
    ay1 = y0 + block_size - border

    # --------------------------------------------------------
    # Array background
    # --------------------------------------------------------

    cv2.rectangle(
        canvas,
        (x0, y0),
        (
            x0 + block_size - 1,
            y0 + block_size - 1,
        ),
        48,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Cell pitch
    # --------------------------------------------------------

    pitch = 140

    x_positions = list(
        range(
            ax0,
            ax1 + 1,
            pitch,
        )
    )

    y_positions = list(
        range(
            ay0,
            ay1 + 1,
            pitch,
        )
    )

    # --------------------------------------------------------
    # Vertical bit lines
    # --------------------------------------------------------

    for x in x_positions:

        width = int(
            np.clip(
                rng.normal(34, 1.5),
                28,
                40,
            )
        )

        intensity = int(
            np.clip(
                rng.normal(182, 4),
                150,
                210,
            )
        )

        draw_line(
            canvas,
            x,
            ay0,
            x,
            ay1,
            intensity,
            width,
        )

    # --------------------------------------------------------
    # Horizontal word lines
    # --------------------------------------------------------

    for y in y_positions:

        width = int(
            np.clip(
                rng.normal(34, 1.5),
                28,
                40,
            )
        )

        intensity = int(
            np.clip(
                rng.normal(155, 4),
                125,
                185,
            )
        )

        draw_line(
            canvas,
            ax0,
            y,
            ax1,
            y,
            intensity,
            width,
        )

    # --------------------------------------------------------
    # Alternating contacts
    # --------------------------------------------------------

    for row, y in enumerate(y_positions):

        for col, x in enumerate(x_positions):

            if (row + col) % 2 == 0:

                radius = int(
                    np.clip(
                        rng.normal(19, 1.2),
                        16,
                        22,
                    )
                )

                draw_contact(
                    canvas,
                    x,
                    y,
                    radius,
                    224,
                )

    # --------------------------------------------------------
    # Array boundary
    # --------------------------------------------------------

    cv2.rectangle(
        canvas,
        (
            ax0 - 48,
            ay0 - 48,
        ),
        (
            ax1 + 48,
            ay1 + 48,
        ),
        72,
        thickness=12,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# COMPLETE PHYSICAL DRAM SCENE
# ============================================================

def generate_physical_scene():
    """
    Generate the complete 10 um x 10 um DRAM scene.

    Resolution:

        10000 x 10000 pixels

    Physical scale:

        1 nm/pixel
    """

    rng = np.random.default_rng(
        SEED
    )

    canvas = np.full(
        (
            FINE_SIZE,
            FINE_SIZE,
        ),
        32,
        dtype=np.uint8,
    )

    # --------------------------------------------------------
    # Repeated DRAM blocks
    # --------------------------------------------------------

    block_size = 2200
    street = 220
    margin = 120

    starts = [
        margin + i * (
            block_size + street
        )
        for i in range(4)
    ]

    for y0 in starts:

        for x0 in starts:

            draw_dram_array(
                canvas,
                x0,
                y0,
                block_size,
                rng,
            )

    # --------------------------------------------------------
    # Vertical isolation/routing streets
    # --------------------------------------------------------

    vertical_streets = [
        450,
        2680,
        4910,
        7140,
        9370,
    ]

    for x in vertical_streets:

        draw_line(
            canvas,
            x,
            0,
            x,
            FINE_SIZE - 1,
            68,
            22,
        )

    # --------------------------------------------------------
    # Horizontal isolation/routing streets
    # --------------------------------------------------------

    horizontal_streets = [
        450,
        2680,
        4910,
        7140,
        9370,
    ]

    for y in horizontal_streets:

        draw_line(
            canvas,
            0,
            y,
            FINE_SIZE - 1,
            y,
            66,
            22,
        )

    # --------------------------------------------------------
    # Very mild physical smoothing
    # --------------------------------------------------------

    canvas = cv2.GaussianBlur(
        canvas,
        (0, 0),
        sigmaX=0.55,
        sigmaY=0.55,
    )

    return canvas


# ============================================================
# SEM ACQUISITION
# ============================================================

def simulate_sem(
    image,
    rng,
    blur_sigma,
    noise_sigma,
    edge_strength,
):
    """
    Apply mild SEM-style image formation.

    This first pair remains relatively clean.
    """

    image_f = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Beam blur
    # --------------------------------------------------------

    image_f = cv2.GaussianBlur(
        image_f,
        (0, 0),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
    )

    # --------------------------------------------------------
    # Edge response
    # --------------------------------------------------------

    smooth = cv2.GaussianBlur(
        image_f,
        (0, 0),
        sigmaX=1.1,
        sigmaY=1.1,
    )

    high_frequency = (
        image_f - smooth
    )

    image_f = (
        image_f
        + edge_strength
        * high_frequency
    )

    # --------------------------------------------------------
    # Detector noise
    # --------------------------------------------------------

    noise = rng.normal(
        0,
        noise_sigma,
        image_f.shape,
    ).astype(np.float32)

    image_f += noise

    return np.clip(
        image_f,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# MAIN GENERATION
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DRIFT-SENSE — DRAM_01 GENERATOR")
    print("=" * 70)

    print()
    print("Generating physical scene...")
    
    # --------------------------------------------------------
    # 1. Generate ONE continuous physical scene
    # --------------------------------------------------------

    physical_scene = (
        generate_physical_scene()
    )

    assert physical_scene.shape == (
        10000,
        10000,
    )

    # --------------------------------------------------------
    # 2. Extract 100x reference
    # --------------------------------------------------------

    print(
        "Generating 100x reference..."
    )

    reference = physical_scene[
        TARGET_Y_NM:
        TARGET_Y_NM + REFERENCE_SIZE_NM,

        TARGET_X_NM:
        TARGET_X_NM + REFERENCE_SIZE_NM,
    ].copy()

    assert reference.shape == (
        1000,
        1000,
    )

    # --------------------------------------------------------
    # 3. Generate 10x search
    # --------------------------------------------------------

    print(
        "Generating 10x search..."
    )

    search = cv2.resize(
        physical_scene,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    assert search.shape == (
        1000,
        1000,
    )

    # --------------------------------------------------------
    # 4. Independent SEM acquisition
    # --------------------------------------------------------

    rng = np.random.default_rng(
        SEED + 100
    )

    reference = simulate_sem(
        reference,
        rng,
        blur_sigma=0.80,
        noise_sigma=1.0,
        edge_strength=1.10,
    )

    search = simulate_sem(
        search,
        rng,
        blur_sigma=0.45,
        noise_sigma=2.0,
        edge_strength=0.70,
    )

    # --------------------------------------------------------
    # 5. Ground truth
    # --------------------------------------------------------

    search_x = (
        TARGET_X_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    search_y = (
        TARGET_Y_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    target_width = (
        REFERENCE_SIZE_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    target_height = target_width

    gt_x = (
        search_x
        + target_width / 2
    )

    gt_y = (
        search_y
        + target_height / 2
    )

    # --------------------------------------------------------
    # 6. Create output directory
    # --------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # 7. Save reference
    # --------------------------------------------------------

    reference_path = (
        OUTPUT_DIR
        / "reference_100x.png"
    )

    cv2.imwrite(
        str(reference_path),
        reference,
    )

    # --------------------------------------------------------
    # 8. Save search
    # --------------------------------------------------------

    search_path = (
        OUTPUT_DIR
        / "search_10x.png"
    )

    cv2.imwrite(
        str(search_path),
        search,
    )

    # --------------------------------------------------------
    # 9. Save ground truth
    # --------------------------------------------------------

    ground_truth = {

        "architecture": "DRAM",

        "pair_id": "dram_01",

        "seed": SEED,

        "reference": {
            "filename": "reference_100x.png",
            "width_px": 1000,
            "height_px": 1000,
            "pixel_size_nm": 1,
            "physical_fov_um": [
                1.0,
                1.0
            ],
            "magnification": "100x",
        },

        "search": {
            "filename": "search_10x.png",
            "width_px": 1000,
            "height_px": 1000,
            "pixel_size_nm": 10,
            "physical_fov_um": [
                10.0,
                10.0
            ],
            "magnification": "10x",
        },

        "target": {

            "physical_origin_nm": [
                TARGET_X_NM,
                TARGET_Y_NM
            ],

            "search_box_xywh": [
                search_x,
                search_y,
                target_width,
                target_height
            ],

            "search_center_xy": [
                gt_x,
                gt_y
            ],
        },

        "coordinate_convention": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
        },

        "generation": {
            "same_physical_scene": True,
            "reference_is_crop": True,
            "search_is_area_downsampled": True,
            "scale_ratio": 10,
        },
    }

    ground_truth_path = (
        OUTPUT_DIR
        / "ground_truth.json"
    )

    with open(
        ground_truth_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            ground_truth,
            f,
            indent=4,
        )

    # --------------------------------------------------------
    # 10. Final report
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("DRAM_01 GENERATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(
        f"Reference : {reference_path}"
    )

    print(
        f"Search    : {search_path}"
    )

    print(
        f"GT box    : "
        f"({search_x:.1f}, "
        f"{search_y:.1f}, "
        f"{target_width:.1f}, "
        f"{target_height:.1f})"
    )

    print(
        f"GT centre : "
        f"({gt_x:.1f}, {gt_y:.1f})"
    )

    print()
    print(
        f"Output    : {OUTPUT_DIR}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()