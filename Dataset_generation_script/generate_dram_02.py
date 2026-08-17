#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-02 Synthetic Pair Generator
=================================

DRAM-02 is designed around a distinctive structural landmark.

The complete search image contains:

    - dense repeated DRAM-like structures
    - diagonal active/cell features
    - repeated contacts
    - periodic array structures
    - ONE unique BOX-SHAPED LANDMARK

The box landmark appears ONLY ONCE in the complete physical
10 um x 10 um scene.

The 100x reference is cropped around that unique structure.

Therefore:

    Reference
        |
        | contains unique box landmark
        v
    Search
        |
        +---- many repeated DRAM patterns
        |
        +---- ONE unique box landmark
                   ^
                   |
                target

Physical relationship:

Reference:
    1000 x 1000 pixels
    1 nm/pixel
    1 um x 1 um

Search:
    1000 x 1000 pixels
    10 nm/pixel
    10 um x 10 um

The reference is extracted from the same physical scene used
to create the search image.

This is important for the localization benchmark.
"""


from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 1000

REFERENCE_SIZE_NM = 1000

REFERENCE_PIXEL_SIZE_NM = 1
SEARCH_PIXEL_SIZE_NM = 10

PHYSICAL_SIZE_NM = 10000

SEED = 20260811


# ============================================================
# UNIQUE TARGET LOCATION
# ============================================================

"""
The unique box structure will be placed around this location.

The reference crop starts here.

Because:

    1000 nm / 10 nm = 100 pixels

the target occupies exactly:

    100 x 100 pixels

in the 10x search image.
"""

TARGET_X_NM = 4850
TARGET_Y_NM = 4650


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_02"
)


# ============================================================
# BASIC DRAWING HELPERS
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
    Draw an anti-aliased line.
    """

    cv2.line(
        image,
        (
            int(round(x1)),
            int(round(y1)),
        ),
        (
            int(round(x2)),
            int(round(y2)),
        ),
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
    Draw a DRAM contact.
    """

    radius = max(
        1,
        int(round(radius)),
    )

    x = int(round(x))
    y = int(round(y))

    cv2.circle(
        image,
        (x, y),
        radius,
        int(intensity),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Inner contrast.
    inner_radius = max(
        1,
        int(round(radius * 0.48)),
    )

    cv2.circle(
        image,
        (x, y),
        inner_radius,
        min(
            255,
            int(intensity + 15),
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# DIAGONAL DRAM CELL
# ============================================================

def draw_diagonal_cell(
    image,
    center_x,
    center_y,
    angle_deg,
    rng,
):
    """
    Draw one elongated diagonal DRAM-like cell.

    The visual structure is intentionally similar to the
    provided DRAM reference:

        elongated diagonal feature
                  +
              circular contacts
    """

    angle = np.deg2rad(
        angle_deg
    )

    dx = np.cos(angle)
    dy = np.sin(angle)

    length = rng.normal(
        185,
        4,
    )

    width = rng.normal(
        58,
        2,
    )

    intensity = int(
        np.clip(
            rng.normal(
                168,
                4,
            ),
            145,
            190,
        )
    )

    # --------------------------------------------------------
    # Main elongated structure
    # --------------------------------------------------------

    half_length = (
        length / 2.0
    )

    x1 = (
        center_x
        - dx * half_length
    )

    y1 = (
        center_y
        - dy * half_length
    )

    x2 = (
        center_x
        + dx * half_length
    )

    y2 = (
        center_y
        + dy * half_length
    )

    draw_line(
        image,
        x1,
        y1,
        x2,
        y2,
        intensity,
        width,
    )

    # Rounded ends.
    radius = int(
        width / 2
    )

    cv2.circle(
        image,
        (
            int(round(x1)),
            int(round(y1)),
        ),
        radius,
        intensity,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    cv2.circle(
        image,
        (
            int(round(x2)),
            int(round(y2)),
        ),
        radius,
        intensity,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Contact at one end
    # --------------------------------------------------------

    contact_distance = 58

    contact_x = (
        center_x
        + dx * contact_distance
    )

    contact_y = (
        center_y
        + dy * contact_distance
    )

    draw_contact(
        image,
        contact_x,
        contact_y,
        rng.normal(
            17,
            1,
        ),
        222,
    )


# ============================================================
# REPEATED DRAM ARRAY
# ============================================================

def draw_dram_array(
    image,
    x0,
    y0,
    width,
    height,
    rng,
):
    """
    Generate a dense repeated DRAM region.

    This creates the repeated background pattern seen
    throughout the search image.
    """

    # --------------------------------------------------------
    # Background
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            x0,
            y0,
        ),
        (
            x0 + width,
            y0 + height,
        ),
        43,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Cell pitch
    # --------------------------------------------------------

    pitch_x = 145
    pitch_y = 145

    start_x = (
        x0 + 105
    )

    start_y = (
        y0 + 105
    )

    end_x = (
        x0 + width - 105
    )

    end_y = (
        y0 + height - 105
    )

    # --------------------------------------------------------
    # Repeated diagonal cells
    # --------------------------------------------------------

    row = 0

    y = start_y

    while y <= end_y:

        col = 0

        x = start_x

        while x <= end_x:

            # Alternating small angle variation.
            if (row + col) % 2 == 0:

                angle = -27.0

            else:

                angle = -25.5

            angle += rng.normal(
                0,
                0.5,
            )

            draw_diagonal_cell(
                image,
                x,
                y,
                angle,
                rng,
            )

            x += pitch_x
            col += 1

        y += pitch_y
        row += 1

    # --------------------------------------------------------
    # Periodic horizontal structures
    # --------------------------------------------------------

    y = y0 + 350

    while y < y0 + height - 200:

        draw_line(
            image,
            x0 + 40,
            y,
            x0 + width - 40,
            y,
            77,
            9,
        )

        y += 480

    # --------------------------------------------------------
    # Periodic vertical structures
    # --------------------------------------------------------

    x = x0 + 350

    while x < x0 + width - 200:

        draw_line(
            image,
            x,
            y0 + 40,
            x,
            y0 + height - 40,
            70,
            8,
        )

        x += 480


# ============================================================
# UNIQUE BOX LANDMARK
# ============================================================

def draw_unique_box_landmark(
    image,
    x0,
    y0,
):
    """
    Draw the ONE unique box-shaped structure.

    IMPORTANT:
        This function is called exactly once.

    The box is intentionally much larger and geometrically
    different from the repeated DRAM cells.

    It provides a strong localization landmark.
    """

    # ========================================================
    # Target box dimensions
    # ========================================================

    box_x0 = x0 + 100
    box_y0 = y0 + 100

    box_x1 = x0 + 900
    box_y1 = y0 + 900

    # --------------------------------------------------------
    # Large dark surrounding region
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            box_x0,
            box_y0,
        ),
        (
            box_x1,
            box_y1,
        ),
        54,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Outer box
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            box_x0,
            box_y0,
        ),
        (
            box_x1,
            box_y1,
        ),
        125,
        thickness=55,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Inner recessed region
    # --------------------------------------------------------

    inner_margin = 150

    inner_x0 = (
        box_x0
        + inner_margin
    )

    inner_y0 = (
        box_y0
        + inner_margin
    )

    inner_x1 = (
        box_x1
        - inner_margin
    )

    inner_y1 = (
        box_y1
        - inner_margin
    )

    cv2.rectangle(
        image,
        (
            inner_x0,
            inner_y0,
        ),
        (
            inner_x1,
            inner_y1,
        ),
        72,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Internal cross structure
    # --------------------------------------------------------

    center_x = (
        x0 + 500
    )

    center_y = (
        y0 + 500
    )

    cv2.rectangle(
        image,
        (
            center_x - 35,
            inner_y0,
        ),
        (
            center_x + 35,
            inner_y1,
        ),
        145,
        thickness=-1,
    )

    cv2.rectangle(
        image,
        (
            inner_x0,
            center_y - 35,
        ),
        (
            inner_x1,
            center_y + 35,
        ),
        145,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Four inner contact features
    # --------------------------------------------------------

    contact_offset = 205

    for sx in (-1, 1):

        for sy in (-1, 1):

            cx = (
                center_x
                + sx * contact_offset
            )

            cy = (
                center_y
                + sy * contact_offset
            )

            draw_contact(
                image,
                cx,
                cy,
                35,
                218,
            )

    # --------------------------------------------------------
    # Small central feature
    # --------------------------------------------------------

    cv2.circle(
        image,
        (
            center_x,
            center_y,
        ),
        52,
        205,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    cv2.circle(
        image,
        (
            center_x,
            center_y,
        ),
        20,
        238,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# COMPLETE PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    """
    Create the complete 10,000 x 10,000 nm physical scene.

    There is exactly ONE box landmark.
    """

    rng = np.random.default_rng(
        SEED
    )

    canvas = np.full(
        (
            PHYSICAL_SIZE_NM,
            PHYSICAL_SIZE_NM,
        ),
        31,
        dtype=np.uint8,
    )

    # ========================================================
    # Repeated DRAM background
    # ========================================================

    block_size = 2200

    street = 220

    margin = 120

    starts = [
        margin
        + i * (
            block_size
            + street
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
                block_size,
                rng,
            )

    # ========================================================
    # Isolation / routing streets
    # ========================================================

    for x in [
        450,
        2680,
        4910,
        7140,
        9370,
    ]:

        draw_line(
            canvas,
            x,
            0,
            x,
            PHYSICAL_SIZE_NM - 1,
            65,
            22,
        )

    for y in [
        450,
        2680,
        4910,
        7140,
        9370,
    ]:

        draw_line(
            canvas,
            0,
            y,
            PHYSICAL_SIZE_NM - 1,
            y,
            64,
            22,
        )

    # ========================================================
    # ONE AND ONLY ONE UNIQUE BOX
    # ========================================================

    draw_unique_box_landmark(
        canvas,
        TARGET_X_NM,
        TARGET_Y_NM,
    )

    # ========================================================
    # Physical smoothing
    # ========================================================

    canvas = cv2.GaussianBlur(
        canvas,
        (
            0,
            0,
        ),
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
    Apply mild SEM-like acquisition differences.
    """

    image_f = image.astype(
        np.float32
    )

    # Beam blur.
    image_f = cv2.GaussianBlur(
        image_f,
        (
            0,
            0,
        ),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
    )

    # Edge response.
    smooth = cv2.GaussianBlur(
        image_f,
        (
            0,
            0,
        ),
        sigmaX=1.1,
        sigmaY=1.1,
    )

    image_f += (
        edge_strength
        * (image_f - smooth)
    )

    # Detector noise.
    noise = rng.normal(
        0,
        noise_sigma,
        image_f.shape,
    )

    image_f += noise.astype(
        np.float32
    )

    return np.clip(
        image_f,
        0,
        255,
    ).astype(np.uint8)


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("DRIFT-SENSE — DRAM_02 GENERATOR")
    print("=" * 70)

    print()
    print("Generating complete physical scene...")

    # ========================================================
    # 1. Generate physical scene
    # ========================================================

    physical_scene = (
        generate_physical_scene()
    )

    assert physical_scene.shape == (
        10000,
        10000,
    )

    # ========================================================
    # 2. Extract reference
    # ========================================================

    print(
        "Extracting unique-landmark reference..."
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

    # ========================================================
    # 3. Create search
    # ========================================================

    print(
        "Creating 10x search image..."
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

    # ========================================================
    # 4. SEM acquisition
    # ========================================================

    rng = np.random.default_rng(
        SEED + 1000
    )

    reference = simulate_sem(
        reference,
        rng,
        blur_sigma=0.70,
        noise_sigma=1.0,
        edge_strength=1.05,
    )

    search = simulate_sem(
        search,
        rng,
        blur_sigma=0.45,
        noise_sigma=1.8,
        edge_strength=0.75,
    )

    # ========================================================
    # 5. Ground truth
    # ========================================================

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
        + target_width / 2.0
    )

    gt_y = (
        search_y
        + target_height / 2.0
    )

    # ========================================================
    # 6. Create output directory
    # ========================================================

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ========================================================
    # 7. Save reference
    # ========================================================

    reference_path = (
        OUTPUT_DIR
        / "reference_100x.png"
    )

    if not cv2.imwrite(
        str(reference_path),
        reference,
    ):
        raise RuntimeError(
            "Failed to save reference image."
        )

    # ========================================================
    # 8. Save search
    # ========================================================

    search_path = (
        OUTPUT_DIR
        / "search_10x.png"
    )

    if not cv2.imwrite(
        str(search_path),
        search,
    ):
        raise RuntimeError(
            "Failed to save search image."
        )

    # ========================================================
    # 9. Save ground truth
    # ========================================================

    ground_truth = {

        "pair_id": "dram_02",

        "architecture": "DRAM",

        "seed": SEED,

        "reference": {

            "filename":
                "reference_100x.png",

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

            "filename":
                "search_10x.png",

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
                TARGET_Y_NM,
            ],

            "search_box_xywh": [
                search_x,
                search_y,
                target_width,
                target_height,
            ],

            "search_center_xy": [
                gt_x,
                gt_y,
            ],

            "unique_landmark":
                "box_structure",
        },

        "coordinate_convention": {

            "origin":
                "top_left",

            "x_direction":
                "right",

            "y_direction":
                "down",
        },

        "generation": {

            "same_physical_scene":
                True,

            "reference_is_crop":
                True,

            "search_is_area_downsampled":
                True,

            "physical_scale_ratio":
                10,

            "unique_box_count":
                1,
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

    # ========================================================
    # 10. Completion report
    # ========================================================

    print()
    print("=" * 70)
    print("DRAM_02 GENERATED SUCCESSFULLY")
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
        "Unique landmark: ONE box structure"
    )

    print()
    print(
        f"Output: {OUTPUT_DIR}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()