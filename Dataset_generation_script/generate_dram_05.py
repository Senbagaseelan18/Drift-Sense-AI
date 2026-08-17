#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-05 Synthetic Pair Generator
=================================

DESIGN
------

This pair is based on a highly periodic DRAM/contact-array
appearance.

The structure contains:

    * dense circular contacts
    * regular row/column pitch
    * small surrounding cell structures
    * subtle local process variation
    * ONE unique diagonal rectangular landmark

The landmark is intentionally small.

The complete physical scene is generated first:

    10,000 x 10,000 nm
    1 nm / pixel

Then:

    Reference:
        1,000 x 1,000 px
        1 nm / px

    Search:
        1,000 x 1,000 px
        10 nm / px

The reference is an exact crop from the physical scene.

Therefore the reference is physically contained inside the
search image.

OUTPUT
------

results/
└── dram_05/
    ├── reference_100x.png
    ├── search_10x.png
    └── ground_truth.json
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

SEED = 20260814


# ============================================================
# TARGET LOCATION
# ============================================================

"""
Different location from the previous DRAM pairs.

The reference crop begins here.
"""

TARGET_X_NM = 6180
TARGET_Y_NM = 3640


# ============================================================
# PERIODIC DRAM ARRAY
# ============================================================

CONTACT_PITCH_NM = 135

CONTACT_RADIUS_NM = 30

CELL_LINE_WIDTH_NM = 12

CELL_HALF_LENGTH_NM = 52


# ============================================================
# UNIQUE LANDMARK
# ============================================================

"""
Small rectangular feature with a diagonal internal line.

This reproduces the visual idea of the uploaded example:

        ┌──────────┐
        │       /  │
        │      /   │
        └──────────┘

It appears exactly once in the entire physical scene.
"""

LANDMARK_WIDTH_NM = 230

LANDMARK_HEIGHT_NM = 170

LANDMARK_LINE_WIDTH_NM = 14


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_05"
)


# ============================================================
# DRAWING HELPERS
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
        (
            int(round(x1)),
            int(round(y1)),
        ),
        (
            int(round(x2)),
            int(round(y2)),
        ),
        int(np.clip(intensity, 0, 255)),
        max(
            1,
            int(round(width)),
        ),
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

    The contact has a soft bright core to create a more
    SEM-like appearance instead of a perfectly flat circle.
    """

    x = int(round(x))
    y = int(round(y))

    radius = max(
        2,
        int(round(radius)),
    )

    # Outer contact.

    cv2.circle(
        image,
        (
            x,
            y,
        ),
        radius,
        int(
            np.clip(
                intensity,
                0,
                255,
            )
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Slightly brighter center.

    inner_radius = max(
        1,
        int(
            round(
                radius * 0.48
            )
        ),
    )

    cv2.circle(
        image,
        (
            x,
            y,
        ),
        inner_radius,
        int(
            np.clip(
                intensity + 10,
                0,
                255,
            )
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# ONE NORMAL DRAM CELL
# ============================================================

def draw_dram_cell(
    image,
    cx,
    cy,
    row,
    col,
    rng,
):
    """
    Draw one periodic DRAM contact/cell.

    Structure:

              |
              |
        ------●------
              |
              |

    The contact remains the dominant visual feature.
    """

    # Small fabrication variation.

    local_x = (
        cx
        + rng.normal(
            0,
            1.5,
        )
    )

    local_y = (
        cy
        + rng.normal(
            0,
            1.5,
        )
    )

    # --------------------------------------------------------
    # Horizontal supporting feature
    # --------------------------------------------------------

    horizontal_length = (
        2
        * CELL_HALF_LENGTH_NM
    )

    draw_line(
        image,
        local_x - horizontal_length / 2,
        local_y,
        local_x + horizontal_length / 2,
        local_y,
        rng.normal(
            82,
            3,
        ),
        CELL_LINE_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Vertical supporting feature
    # --------------------------------------------------------

    vertical_length = (
        2
        * CELL_HALF_LENGTH_NM
    )

    draw_line(
        image,
        local_x,
        local_y - vertical_length / 2,
        local_x,
        local_y + vertical_length / 2,
        rng.normal(
            78,
            3,
        ),
        CELL_LINE_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Main circular contact
    # --------------------------------------------------------

    draw_contact(
        image,
        local_x,
        local_y,
        rng.normal(
            CONTACT_RADIUS_NM,
            1.2,
        ),
        rng.normal(
            194,
            5,
        ),
    )


# ============================================================
# MEMORY ARRAY REGION
# ============================================================

def draw_memory_array(
    image,
    x0,
    y0,
    width,
    height,
    rng,
):
    """
    Draw a dense regular DRAM/contact array.
    """

    # Dark semiconductor background.

    cv2.rectangle(
        image,
        (
            int(x0),
            int(y0),
        ),
        (
            int(x0 + width),
            int(y0 + height),
        ),
        47,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Periodic contacts
    # --------------------------------------------------------

    start_x = (
        x0
        + CONTACT_PITCH_NM / 2
    )

    start_y = (
        y0
        + CONTACT_PITCH_NM / 2
    )

    end_x = (
        x0
        + width
        - CONTACT_PITCH_NM / 2
    )

    end_y = (
        y0
        + height
        - CONTACT_PITCH_NM / 2
    )

    row = 0

    y = start_y

    while y <= end_y:

        col = 0

        x = start_x

        while x <= end_x:

            draw_dram_cell(
                image,
                x,
                y,
                row,
                col,
                rng,
            )

            x += CONTACT_PITCH_NM

            col += 1

        y += CONTACT_PITCH_NM

        row += 1


# ============================================================
# UNIQUE DIAGONAL LANDMARK
# ============================================================

def draw_unique_landmark(
    image,
    center_x,
    center_y,
):
    """
    Draw ONE small rectangular/diagonal landmark.

    This is the only occurrence in the entire physical scene.

    The landmark is intentionally small and does not dominate
    the reference.
    """

    half_w = (
        LANDMARK_WIDTH_NM
        / 2
    )

    half_h = (
        LANDMARK_HEIGHT_NM
        / 2
    )

    x0 = int(
        round(
            center_x
            - half_w
        )
    )

    y0 = int(
        round(
            center_y
            - half_h
        )
    )

    x1 = int(
        round(
            center_x
            + half_w
        )
    )

    y1 = int(
        round(
            center_y
            + half_h
        )
    )

    # --------------------------------------------------------
    # Slightly bright rectangular structure
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
        142,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Darker interior
    # --------------------------------------------------------

    inner_margin_x = 18
    inner_margin_y = 16

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
        78,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Diagonal line
    # --------------------------------------------------------

    draw_line(
        image,
        ix0 + 18,
        iy1 - 18,
        ix1 - 18,
        iy0 + 18,
        220,
        LANDMARK_LINE_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Slightly dark border
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
        116,
        thickness=8,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# COMPLETE PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    """
    Generate the complete 10 um x 10 um scene.

    The scene contains several dense DRAM regions separated
    by dark streets.

    Exactly ONE landmark is inserted.
    """

    rng = np.random.default_rng(
        SEED
    )

    canvas = np.full(
        (
            PHYSICAL_SIZE_NM,
            PHYSICAL_SIZE_NM,
        ),
        34,
        dtype=np.uint8,
    )

    # ========================================================
    # ARRAY BLOCKS
    # ========================================================

    block_size = 2150

    street = 260

    margin = 100

    block_starts = [
        margin
        + i
        * (
            block_size
            + street
        )
        for i in range(4)
    ]

    for y0 in block_starts:

        for x0 in block_starts:

            draw_memory_array(
                canvas,
                x0,
                y0,
                block_size,
                block_size,
                rng,
            )

    # ========================================================
    # DARK INTER-BLOCK STREETS
    # ========================================================

    # Vertical streets.

    for x in [
        2310,
        4620,
        6930,
        9240,
    ]:

        draw_line(
            canvas,
            x,
            0,
            x,
            PHYSICAL_SIZE_NM,
            34,
            street,
        )

    # Horizontal streets.

    for y in [
        2310,
        4620,
        6930,
        9240,
    ]:

        draw_line(
            canvas,
            0,
            y,
            PHYSICAL_SIZE_NM,
            y,
            34,
            street,
        )

    # ========================================================
    # SUBTLE BLOCK BORDER
    # ========================================================

    for y0 in block_starts:

        for x0 in block_starts:

            cv2.rectangle(
                canvas,
                (
                    int(x0),
                    int(y0),
                ),
                (
                    int(
                        x0
                        + block_size
                    ),
                    int(
                        y0
                        + block_size
                    ),
                ),
                60,
                thickness=8,
                lineType=cv2.LINE_AA,
            )

    # ========================================================
    # UNIQUE LANDMARK
    # ========================================================

    landmark_center_x = (
        TARGET_X_NM
        + REFERENCE_SIZE_NM / 2
    )

    landmark_center_y = (
        TARGET_Y_NM
        + REFERENCE_SIZE_NM / 2
    )

    draw_unique_landmark(
        canvas,
        landmark_center_x,
        landmark_center_y,
    )

    # ========================================================
    # PHYSICAL IMAGE FORMATION
    # ========================================================

    canvas = cv2.GaussianBlur(
        canvas,
        (
            0,
            0,
        ),
        sigmaX=0.60,
        sigmaY=0.60,
    )

    return canvas


# ============================================================
# SEM NOISE
# ============================================================

def add_poisson_noise(
    image,
    rng,
    photon_level,
):
    """
    Approximate SEM shot noise.
    """

    image_f = np.clip(
        image.astype(
            np.float32
        ),
        0,
        255,
    )

    normalized = (
        image_f
        / 255.0
    )

    photons = (
        normalized
        * photon_level
    )

    photons = np.maximum(
        photons,
        0.01,
    )

    noisy = rng.poisson(
        photons
    ).astype(
        np.float32
    )

    noisy /= photon_level

    noisy *= 255.0

    return noisy


def add_detector_noise(
    image,
    rng,
    sigma,
):
    """
    SEM detector/readout noise.
    """

    noise = rng.normal(
        0,
        sigma,
        image.shape,
    ).astype(
        np.float32
    )

    return image + noise


def add_low_frequency_variation(
    image,
    rng,
    amplitude,
):
    """
    Smooth background variation.
    """

    h, w = image.shape

    low_res = rng.normal(
        0,
        1,
        (
            24,
            24,
        ),
    ).astype(
        np.float32
    )

    field = cv2.resize(
        low_res,
        (
            w,
            h,
        ),
        interpolation=cv2.INTER_CUBIC,
    )

    field -= field.mean()

    std = field.std()

    if std > 1e-6:

        field /= std

    return (
        image
        + amplitude
        * field
    )


def add_scan_variation(
    image,
    rng,
    amplitude,
):
    """
    Mild line-by-line SEM acquisition variation.
    """

    h, _ = image.shape

    row_noise = rng.normal(
        0,
        amplitude,
        h,
    ).astype(
        np.float32
    )

    return (
        image
        + row_noise[:, None]
    )


def simulate_sem(
    image,
    rng,
    blur_sigma,
    photon_level,
    detector_sigma,
    low_frequency_amplitude,
    scan_amplitude,
):
    """
    Complete SEM-like acquisition.
    """

    image_f = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Beam blur
    # --------------------------------------------------------

    image_f = cv2.GaussianBlur(
        image_f,
        (
            0,
            0,
        ),
        sigmaX=blur_sigma,
        sigmaY=blur_sigma,
    )

    # --------------------------------------------------------
    # Mild edge enhancement
    # --------------------------------------------------------

    smooth = cv2.GaussianBlur(
        image_f,
        (
            0,
            0,
        ),
        sigmaX=1.15,
        sigmaY=1.15,
    )

    image_f += (
        0.35
        * (
            image_f
            - smooth
        )
    )

    # --------------------------------------------------------
    # Shot noise
    # --------------------------------------------------------

    image_f = add_poisson_noise(
        image_f,
        rng,
        photon_level,
    )

    # --------------------------------------------------------
    # Detector noise
    # --------------------------------------------------------

    image_f = add_detector_noise(
        image_f,
        rng,
        detector_sigma,
    )

    # --------------------------------------------------------
    # Background variation
    # --------------------------------------------------------

    image_f = add_low_frequency_variation(
        image_f,
        rng,
        low_frequency_amplitude,
    )

    # --------------------------------------------------------
    # Scan-line variation
    # --------------------------------------------------------

    image_f = add_scan_variation(
        image_f,
        rng,
        scan_amplitude,
    )

    return np.clip(
        image_f,
        0,
        255,
    ).astype(
        np.uint8
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 72)
    print("DRIFT-SENSE — DRAM_05 GENERATOR")
    print("=" * 72)

    print()
    print("Pattern:")
    print(
        "Dense periodic circular-contact DRAM array"
    )

    print()
    print("Unique structure:")
    print(
        "ONE small rectangle + diagonal landmark"
    )

    print()
    print("Scale:")
    print(
        "Reference = 1 nm/px"
    )
    print(
        "Search    = 10 nm/px"
    )

    # ========================================================
    # 1. Physical scene
    # ========================================================

    print()
    print(
        "[1/6] Generating physical scene..."
    )

    physical_scene = (
        generate_physical_scene()
    )

    if physical_scene.shape != (
        10000,
        10000,
    ):

        raise RuntimeError(
            "Invalid physical scene dimensions."
        )

    # ========================================================
    # 2. Reference
    # ========================================================

    print(
        "[2/6] Extracting reference..."
    )

    reference = physical_scene[
        TARGET_Y_NM:
        TARGET_Y_NM + REFERENCE_SIZE_NM,

        TARGET_X_NM:
        TARGET_X_NM + REFERENCE_SIZE_NM,
    ].copy()

    if reference.shape != (
        1000,
        1000,
    ):

        raise RuntimeError(
            "Invalid reference dimensions."
        )

    # ========================================================
    # 3. Search
    # ========================================================

    print(
        "[3/6] Creating search..."
    )

    search = cv2.resize(
        physical_scene,
        (
            IMAGE_SIZE,
            IMAGE_SIZE,
        ),
        interpolation=cv2.INTER_AREA,
    )

    if search.shape != (
        1000,
        1000,
    ):

        raise RuntimeError(
            "Invalid search dimensions."
        )

    # ========================================================
    # 4. SEM degradation
    # ========================================================

    print(
        "[4/6] Applying SEM degradation..."
    )

    reference_rng = np.random.default_rng(
        SEED + 100
    )

    search_rng = np.random.default_rng(
        SEED + 200
    )

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = simulate_sem(
        reference,
        reference_rng,
        blur_sigma=0.65,
        photon_level=950,
        detector_sigma=1.2,
        low_frequency_amplitude=2.0,
        scan_amplitude=0.55,
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search = simulate_sem(
        search,
        search_rng,
        blur_sigma=0.50,
        photon_level=560,
        detector_sigma=2.0,
        low_frequency_amplitude=3.5,
        scan_amplitude=0.9,
    )

    # ========================================================
    # 5. Ground truth
    # ========================================================

    print(
        "[5/6] Creating ground truth..."
    )

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

    # ========================================================
    # 6. Save
    # ========================================================

    print(
        "[6/6] Saving..."
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_path = (
        OUTPUT_DIR
        / "reference_100x.png"
    )

    search_path = (
        OUTPUT_DIR
        / "search_10x.png"
    )

    ground_truth_path = (
        OUTPUT_DIR
        / "ground_truth.json"
    )

    cv2.imwrite(
        str(reference_path),
        reference,
    )

    cv2.imwrite(
        str(search_path),
        search,
    )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "pair_id":
            "dram_05",

        "architecture":
            "DRAM",

        "seed":
            SEED,

        "reference": {

            "filename":
                "reference_100x.png",

            "width_px":
                1000,

            "height_px":
                1000,

            "pixel_size_nm":
                1,

            "physical_fov_um":
                [
                    1.0,
                    1.0,
                ],

            "magnification":
                "100x",
        },

        "search": {

            "filename":
                "search_10x.png",

            "width_px":
                1000,

            "height_px":
                1000,

            "pixel_size_nm":
                10,

            "physical_fov_um":
                [
                    10.0,
                    10.0,
                ],

            "magnification":
                "10x",
        },

        "target": {

            "physical_origin_nm":
                [
                    TARGET_X_NM,
                    TARGET_Y_NM,
                ],

            "search_box_xywh":
                [
                    search_x,
                    search_y,
                    target_width,
                    target_height,
                ],

            "search_center_xy":
                [
                    gt_x,
                    gt_y,
                ],

            "landmark_type":
                "small_diagonal_rectangle",

            "landmark_physical_size_nm":
                [
                    LANDMARK_WIDTH_NM,
                    LANDMARK_HEIGHT_NM,
                ],

            "landmark_count":
                1,
        },

        "pattern": {

            "contact_pitch_nm":
                CONTACT_PITCH_NM,

            "contact_radius_nm":
                CONTACT_RADIUS_NM,

            "cell_line_width_nm":
                CELL_LINE_WIDTH_NM,

            "periodic":
                True,
        },

        "noise_model": {

            "poisson_shot_noise":
                True,

            "detector_noise":
                True,

            "low_frequency_variation":
                True,

            "scan_line_variation":
                True,

            "reference_photon_level":
                950,

            "search_photon_level":
                560,

            "reference_detector_sigma":
                1.2,

            "search_detector_sigma":
                2.0,
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

        "coordinate_convention": {

            "origin":
                "top_left",

            "x_direction":
                "right",

            "y_direction":
                "down",
        },
    }

    with open(
        ground_truth_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4,
        )

    # ========================================================
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 72)
    print("DRAM_05 GENERATED SUCCESSFULLY")
    print("=" * 72)

    print()
    print(
        f"Reference : {reference_path}"
    )

    print(
        f"Search    : {search_path}"
    )

    print()
    print(
        "Target search box:"
    )

    print(
        f"  x      = {search_x:.1f} px"
    )

    print(
        f"  y      = {search_y:.1f} px"
    )

    print(
        f"  width  = {target_width:.1f} px"
    )

    print(
        f"  height = {target_height:.1f} px"
    )

    print()
    print(
        "Target center:"
    )

    print(
        f"  ({gt_x:.1f}, {gt_y:.1f}) px"
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
    print("=" * 72)


if __name__ == "__main__":
    main()