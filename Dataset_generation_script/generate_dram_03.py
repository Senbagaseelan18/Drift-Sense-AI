#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-03 Synthetic Pair Generator
=================================

DRAM-03 contains a SMALL UNIQUE BOX-LIKE STRUCTURE
embedded naturally inside a dense DRAM-style array.

The objective is different from DRAM-02:

DRAM-02:
    Large obvious box landmark.

DRAM-03:
    Small integrated box landmark surrounded by
    normal DRAM structures.

The complete physical scene is generated at:

    10,000 x 10,000 nm
    1 nm / pixel

Reference:

    1,000 x 1,000 pixels
    1 nm / pixel
    1 um x 1 um

Search:

    1,000 x 1,000 pixels
    10 nm / pixel
    10 um x 10 um

The reference is an exact crop from the same physical
scene used to generate the search.

Therefore the target appears at exactly 10:1 scale.
"""


from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = 1000

PHYSICAL_SIZE_NM = 10000

REFERENCE_SIZE_NM = 1000

REFERENCE_PIXEL_SIZE_NM = 1

SEARCH_PIXEL_SIZE_NM = 10

SEED = 20260812


# ============================================================
# DRAM-03 TARGET LOCATION
# ============================================================

# New location, different from DRAM-01 and DRAM-02.

TARGET_X_NM = 6250
TARGET_Y_NM = 5850


# ============================================================
# SMALL UNIQUE LANDMARK
# ============================================================

# IMPORTANT:
#
# This is deliberately much smaller than DRAM-02.
#
# Physical size approximately:
#
#     260 nm x 220 nm
#
# At 10 nm/pixel this becomes approximately:
#
#     26 x 22 pixels
#
# in the search image.
#
# The landmark is therefore visible but not dominant.

BOX_WIDTH_NM = 260
BOX_HEIGHT_NM = 220


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_03"
)


# ============================================================
# BASIC DRAWING
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
    Draw an anti-aliased semiconductor feature.
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
    Draw a small DRAM contact/via.
    """

    x = int(round(x))
    y = int(round(y))

    radius = max(
        1,
        int(round(radius)),
    )

    cv2.circle(
        image,
        (x, y),
        radius,
        int(intensity),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    inner_radius = max(
        1,
        int(round(radius * 0.45)),
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
# DRAM DIAGONAL CELL
# ============================================================

def draw_dram_cell(
    image,
    center_x,
    center_y,
    angle_deg,
    rng,
):
    """
    Draw one elongated diagonal DRAM-like feature.

    Each cell contains:

        elongated active structure
        +
        contact/via
    """

    angle = np.deg2rad(
        angle_deg
    )

    dx = np.cos(angle)
    dy = np.sin(angle)

    # --------------------------------------------------------
    # Main active structure
    # --------------------------------------------------------

    length = rng.normal(
        185,
        4,
    )

    width = rng.normal(
        56,
        2,
    )

    intensity = int(
        np.clip(
            rng.normal(
                166,
                4,
            ),
            145,
            190,
        )
    )

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

    # Central elongated region.

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
    # Contact
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
        220,
    )


# ============================================================
# DRAM ARRAY
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
    Generate a dense DRAM array.

    This is the dominant visual structure of DRAM-03.
    """

    # --------------------------------------------------------
    # Array background
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
    # Repeated cell grid
    # --------------------------------------------------------

    pitch_x = 145
    pitch_y = 145

    start_x = x0 + 95
    start_y = y0 + 95

    end_x = x0 + width - 95
    end_y = y0 + height - 95

    row = 0

    y = start_y

    while y <= end_y:

        col = 0

        x = start_x

        while x <= end_x:

            # Predominantly consistent diagonal direction,
            # with tiny physical variation.

            if (
                (row + col) % 2
                == 0
            ):
                angle = -27.0
            else:
                angle = -25.5

            angle += rng.normal(
                0,
                0.45,
            )

            draw_dram_cell(
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
    # Word-line structures
    # --------------------------------------------------------

    y = y0 + 340

    while y < y0 + height - 180:

        draw_line(
            image,
            x0 + 35,
            y,
            x0 + width - 35,
            y,
            76,
            8,
        )

        y += 480

    # --------------------------------------------------------
    # Bit-line structures
    # --------------------------------------------------------

    x = x0 + 340

    while x < x0 + width - 180:

        draw_line(
            image,
            x,
            y0 + 35,
            x,
            y0 + height - 35,
            69,
            8,
        )

        x += 480


# ============================================================
# SMALL UNIQUE BOX STRUCTURE
# ============================================================

def draw_small_unique_box(
    image,
    center_x,
    center_y,
):
    """
    Draw the small unique structure for DRAM-03.

    Unlike DRAM-02, this is NOT a huge isolated landmark.

    It is intentionally compact and integrated into the
    surrounding DRAM pattern.

    Physical size:

        ~260 nm x 220 nm
    """

    half_w = BOX_WIDTH_NM / 2.0
    half_h = BOX_HEIGHT_NM / 2.0

    x0 = int(
        round(center_x - half_w)
    )

    y0 = int(
        round(center_y - half_h)
    )

    x1 = int(
        round(center_x + half_w)
    )

    y1 = int(
        round(center_y + half_h)
    )

    # --------------------------------------------------------
    # Slightly brighter rectangular enclosure
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            x0,
            y0,
        ),
        (
            x1,
            y1,
        ),
        112,
        thickness=18,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Inner recessed region
    # --------------------------------------------------------

    inner_margin_x = 48
    inner_margin_y = 40

    ix0 = x0 + inner_margin_x
    iy0 = y0 + inner_margin_y

    ix1 = x1 - inner_margin_x
    iy1 = y1 - inner_margin_y

    cv2.rectangle(
        image,
        (
            ix0,
            iy0,
        ),
        (
            ix1,
            iy1,
        ),
        63,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Small internal gate/bar
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            int(center_x - 18),
            iy0,
        ),
        (
            int(center_x + 18),
            iy1,
        ),
        132,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Two small contacts
    # --------------------------------------------------------

    draw_contact(
        image,
        center_x - 72,
        center_y,
        18,
        218,
    )

    draw_contact(
        image,
        center_x + 72,
        center_y,
        18,
        218,
    )


# ============================================================
# PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    """
    Generate the complete 10 um x 10 um physical DRAM scene.

    Exactly ONE small unique box is inserted.
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

    # --------------------------------------------------------
    # Repeated DRAM arrays
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Isolation/routing streets
    # --------------------------------------------------------

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
            64,
            20,
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
            63,
            20,
        )

    # ========================================================
    # UNIQUE SMALL LANDMARK
    # ========================================================

    draw_small_unique_box(
        canvas,
        TARGET_X_NM + 500,
        TARGET_Y_NM + 500,
    )

    # --------------------------------------------------------
    # Physical smoothing
    # --------------------------------------------------------

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
    Apply mild SEM-style image formation.
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
        * (
            image_f
            - smooth
        )
    )

    # Detector noise.

    noise = rng.normal(
        0,
        noise_sigma,
        image_f.shape,
    ).astype(
        np.float32
    )

    image_f += noise

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
    print("DRIFT-SENSE — DRAM_03 GENERATOR")
    print("=" * 70)

    print()
    print("Structure:")
    print(
        "Dense DRAM array + "
        "ONE small unique box landmark"
    )

    print()
    print("Generating physical scene...")

    # ========================================================
    # 1. Generate complete physical scene
    # ========================================================

    physical_scene = (
        generate_physical_scene()
    )

    assert physical_scene.shape == (
        10000,
        10000,
    )

    # ========================================================
    # 2. Reference crop
    # ========================================================

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

    # ========================================================
    # 3. Search image
    # ========================================================

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

    # ========================================================
    # 4. SEM acquisition
    # ========================================================

    rng = np.random.default_rng(
        SEED + 500
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
    # 6. Output directory
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
            "Failed to save reference."
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
            "Failed to save search."
        )

    # ========================================================
    # 9. Save ground truth
    # ========================================================

    ground_truth = {

        "pair_id": "dram_03",

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
                1.0,
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
                10.0,
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

            "landmark_type":
                "small_unique_box",

            "landmark_physical_size_nm": [
                BOX_WIDTH_NM,
                BOX_HEIGHT_NM,
            ],

            "landmark_count":
                1,
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
    # 10. Final report
    # ========================================================

    print()
    print("=" * 70)
    print("DRAM_03 GENERATED SUCCESSFULLY")
    print("=" * 70)

    print()
    print(
        f"Reference : {reference_path}"
    )

    print(
        f"Search    : {search_path}"
    )

    print()
    print(
        "Small landmark:"
    )

    print(
        f"  Physical size = "
        f"{BOX_WIDTH_NM} x "
        f"{BOX_HEIGHT_NM} nm"
    )

    print(
        f"  Search size = "
        f"{BOX_WIDTH_NM / 10:.1f} x "
        f"{BOX_HEIGHT_NM / 10:.1f} px"
    )

    print()
    print(
        "Ground-truth box:"
    )

    print(
        f"  x = {search_x:.1f}"
    )

    print(
        f"  y = {search_y:.1f}"
    )

    print(
        f"  width = {target_width:.1f}"
    )

    print(
        f"  height = {target_height:.1f}"
    )

    print()
    print(
        "Ground-truth centre:"
    )

    print(
        f"  ({gt_x:.1f}, {gt_y:.1f})"
    )

    print()
    print(
        "Unique landmark count: 1"
    )

    print()
    print(
        f"Output: {OUTPUT_DIR}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()