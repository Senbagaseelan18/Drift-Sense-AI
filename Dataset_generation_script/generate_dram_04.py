#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-04 Synthetic Pair Generator
================================

DRAM-04 is a dense DRAM-style memory-array scene.

DESIGN GOAL
-----------

The generated image is intentionally based on a dense,
highly repetitive semiconductor memory-array appearance:

    * large repeated memory blocks
    * dark isolation streets
    * dense horizontal/vertical cell lines
    * small bright contact features
    * repeated cross-like cell geometry
    * small local process variation

There is exactly ONE unique local structural modification.

The landmark is NOT a large box.

Instead, one small group of cells contains a modified
contact/line arrangement that is different from all other
locations.

This makes the localization problem more realistic.


PHYSICAL RELATIONSHIP
---------------------

Complete physical scene:

    10000 x 10000 nm
    1 nm / pixel

Reference:

    1000 x 1000 pixels
    1 nm / pixel
    1 um x 1 um

Search:

    1000 x 1000 pixels
    10 nm / pixel
    10 um x 10 um

The reference is an exact crop from the same physical
scene used to create the search.


NOISE
-----

Moderate SEM-style degradation:

    * Poisson / shot noise
    * detector noise
    * low-frequency intensity variation
    * row/line variation
    * small charging-like background variation

The search is slightly more degraded than the reference.
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

SEED = 20260813


# ============================================================
# TARGET LOCATION
# ============================================================

"""
The 1 um x 1 um reference starts here in the physical scene.

The unique modified cell is located inside this crop.
"""

TARGET_X_NM = 7350
TARGET_Y_NM = 2450


# ============================================================
# ARRAY CONFIGURATION
# ============================================================

BLOCK_SIZE_NM = 2200

STREET_WIDTH_NM = 220

ARRAY_MARGIN_NM = 120


# ============================================================
# CELL CONFIGURATION
# ============================================================

CELL_PITCH_NM = 105

CELL_LINE_WIDTH_NM = 24

CONTACT_SIZE_NM = 24


# ============================================================
# UNIQUE LOCAL STRUCTURE
# ============================================================

"""
The landmark is a small modified 3x3 cell neighborhood.

Normal structure:

    + + +
    + + +
    + + +

Modified structure:

    + + +
    + - +
    + + +

plus a local shifted line/contact arrangement.

At 10 nm/pixel the landmark remains small.
"""

LANDMARK_SIZE_NM = 315


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_04"
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
    Draw a semiconductor line with anti-aliasing.
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
        max(1, int(round(width))),
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
    Draw a small bright contact / via.
    """

    x = int(round(x))
    y = int(round(y))

    radius = max(
        2,
        int(round(radius)),
    )

    cv2.circle(
        image,
        (x, y),
        radius,
        int(np.clip(intensity, 0, 255)),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Slightly brighter center.

    inner_radius = max(
        1,
        int(round(radius * 0.42)),
    )

    cv2.circle(
        image,
        (x, y),
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
# NORMAL DRAM CELL
# ============================================================

def draw_normal_cell(
    image,
    center_x,
    center_y,
    row,
    col,
    rng,
):
    """
    Draw one normal DRAM cell.

    Visual structure:

        horizontal feature
             +
        vertical feature
             +
          contact

    The result creates a dense repeated cross/grid appearance.
    """

    # --------------------------------------------------------
    # Small physical variation
    # --------------------------------------------------------

    local_x = (
        center_x
        + rng.normal(0, 2.0)
    )

    local_y = (
        center_y
        + rng.normal(0, 2.0)
    )

    # --------------------------------------------------------
    # Vertical cell feature
    # --------------------------------------------------------

    vertical_height = rng.normal(
        70,
        2,
    )

    vertical_width = rng.normal(
        18,
        1,
    )

    vertical_intensity = rng.normal(
        178,
        5,
    )

    draw_line(
        image,
        local_x,
        local_y - vertical_height / 2,
        local_x,
        local_y + vertical_height / 2,
        vertical_intensity,
        vertical_width,
    )

    # --------------------------------------------------------
    # Horizontal cell feature
    # --------------------------------------------------------

    horizontal_width = rng.normal(
        70,
        2,
    )

    horizontal_height = rng.normal(
        18,
        1,
    )

    horizontal_intensity = rng.normal(
        156,
        5,
    )

    draw_line(
        image,
        local_x - horizontal_width / 2,
        local_y,
        local_x + horizontal_width / 2,
        local_y,
        horizontal_intensity,
        horizontal_height,
    )

    # --------------------------------------------------------
    # Bright contact
    # --------------------------------------------------------

    contact_radius = rng.normal(
        10.5,
        0.7,
    )

    # Slight alternating contact placement.
    if (row + col) % 2 == 0:

        contact_dx = 22

    else:

        contact_dx = -22

    contact_dy = -22

    draw_contact(
        image,
        local_x + contact_dx,
        local_y + contact_dy,
        contact_radius,
        rng.normal(
            218,
            5,
        ),
    )


# ============================================================
# DENSE MEMORY BLOCK
# ============================================================

def draw_memory_block(
    image,
    x0,
    y0,
    width,
    height,
    rng,
):
    """
    Generate one dense DRAM memory block.
    """

    # --------------------------------------------------------
    # Dark block background
    # --------------------------------------------------------

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
        43,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Cell positions
    # --------------------------------------------------------

    start_x = (
        x0 + 75
    )

    start_y = (
        y0 + 75
    )

    end_x = (
        x0 + width - 75
    )

    end_y = (
        y0 + height - 75
    )

    row = 0

    y = start_y

    while y <= end_y:

        col = 0

        x = start_x

        while x <= end_x:

            draw_normal_cell(
                image,
                x,
                y,
                row,
                col,
                rng,
            )

            x += CELL_PITCH_NM
            col += 1

        y += CELL_PITCH_NM
        row += 1

    # --------------------------------------------------------
    # Fine continuous word/bit structures
    # --------------------------------------------------------

    # Horizontal fine lines.

    y = y0 + 48

    while y < y0 + height - 48:

        draw_line(
            image,
            x0 + 20,
            y,
            x0 + width - 20,
            y,
            rng.normal(
                72,
                2,
            ),
            5,
        )

        y += CELL_PITCH_NM

    # Vertical fine lines.

    x = x0 + 48

    while x < x0 + width - 48:

        draw_line(
            image,
            x,
            y0 + 20,
            x,
            y0 + height - 20,
            rng.normal(
                68,
                2,
            ),
            5,
        )

        x += CELL_PITCH_NM


# ============================================================
# UNIQUE MODIFIED CELL CLUSTER
# ============================================================

def draw_unique_modified_cluster(
    image,
    center_x,
    center_y,
    rng,
):
    """
    Create ONE subtle but detectable structural anomaly.

    Normal cells have:

        cross + contact

    This 3x3 region contains:

        * one missing contact
        * one shifted contact
        * one reinforced vertical line
        * one asymmetric bright feature

    The result is still semiconductor-like and embedded
    in the surrounding repeated DRAM pattern.
    """

    pitch = CELL_PITCH_NM

    # 3 x 3 local neighborhood.

    offsets = [
        (-1, -1),
        ( 0, -1),
        ( 1, -1),

        (-1,  0),
        ( 0,  0),
        ( 1,  0),

        (-1,  1),
        ( 0,  1),
        ( 1,  1),
    ]

    # --------------------------------------------------------
    # Redraw local region with slightly stronger contrast.
    # --------------------------------------------------------

    for dx, dy in offsets:

        cx = (
            center_x
            + dx * pitch
        )

        cy = (
            center_y
            + dy * pitch
        )

        # Normal local cross.

        draw_line(
            image,
            cx,
            cy - 36,
            cx,
            cy + 36,
            185,
            20,
        )

        draw_line(
            image,
            cx - 36,
            cy,
            cx + 36,
            cy,
            162,
            20,
        )

    # --------------------------------------------------------
    # 1. Remove the normal centre contact.
    #
    # Create a slightly dark recessed feature.
    # --------------------------------------------------------

    cv2.circle(
        image,
        (
            int(center_x),
            int(center_y),
        ),
        17,
        52,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # 2. Shifted bright contact.
    # --------------------------------------------------------

    draw_contact(
        image,
        center_x + 31,
        center_y - 27,
        12,
        235,
    )

    # --------------------------------------------------------
    # 3. Reinforced vertical structure.
    # --------------------------------------------------------

    draw_line(
        image,
        center_x,
        center_y - 67,
        center_x,
        center_y + 67,
        205,
        29,
    )

    # --------------------------------------------------------
    # 4. Short asymmetric extension.
    # --------------------------------------------------------

    draw_line(
        image,
        center_x + 10,
        center_y + 8,
        center_x + 78,
        center_y + 8,
        194,
        24,
    )

    # --------------------------------------------------------
    # 5. Small bright endpoint.
    # --------------------------------------------------------

    draw_contact(
        image,
        center_x + 79,
        center_y + 8,
        11,
        228,
    )

    # --------------------------------------------------------
    # 6. Subtle local material halo.
    # --------------------------------------------------------

    cv2.circle(
        image,
        (
            int(center_x),
            int(center_y),
        ),
        82,
        58,
        thickness=3,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# COMPLETE PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    """
    Generate the complete 10,000 x 10,000 nm DRAM scene.

    There is exactly ONE modified cell cluster.
    """

    rng = np.random.default_rng(
        SEED
    )

    canvas = np.full(
        (
            PHYSICAL_SIZE_NM,
            PHYSICAL_SIZE_NM,
        ),
        29,
        dtype=np.uint8,
    )

    # ========================================================
    # 4 x 4 MEMORY BLOCK ARRAY
    # ========================================================

    starts = [
        ARRAY_MARGIN_NM
        + i * (
            BLOCK_SIZE_NM
            + STREET_WIDTH_NM
        )
        for i in range(4)
    ]

    for y0 in starts:

        for x0 in starts:

            draw_memory_block(
                canvas,
                x0,
                y0,
                BLOCK_SIZE_NM,
                BLOCK_SIZE_NM,
                rng,
            )

    # ========================================================
    # DARK ISOLATION STREETS
    # ========================================================

    # Vertical streets.

    for x in [
        2320,
        4740,
        7160,
        9580,
    ]:

        draw_line(
            canvas,
            x,
            0,
            x,
            PHYSICAL_SIZE_NM,
            34,
            STREET_WIDTH_NM,
        )

    # Horizontal streets.

    for y in [
        2320,
        4740,
        7160,
        9580,
    ]:

        draw_line(
            canvas,
            0,
            y,
            PHYSICAL_SIZE_NM,
            y,
            34,
            STREET_WIDTH_NM,
        )

    # ========================================================
    # BLOCK EDGE DETAILS
    # ========================================================

    # Add subtle perimeter lines around blocks.

    for y0 in starts:

        for x0 in starts:

            cv2.rectangle(
                canvas,
                (
                    int(x0),
                    int(y0),
                ),
                (
                    int(
                        x0
                        + BLOCK_SIZE_NM
                    ),
                    int(
                        y0
                        + BLOCK_SIZE_NM
                    ),
                ),
                62,
                thickness=8,
                lineType=cv2.LINE_AA,
            )

    # ========================================================
    # UNIQUE LANDMARK
    # ========================================================

    landmark_center_x = (
        TARGET_X_NM
        + REFERENCE_SIZE_NM // 2
    )

    landmark_center_y = (
        TARGET_Y_NM
        + REFERENCE_SIZE_NM // 2
    )

    draw_unique_modified_cluster(
        canvas,
        landmark_center_x,
        landmark_center_y,
        rng,
    )

    # ========================================================
    # VERY MILD PHYSICAL BLUR
    # ========================================================

    canvas = cv2.GaussianBlur(
        canvas,
        (
            0,
            0,
        ),
        sigmaX=0.65,
        sigmaY=0.65,
    )

    return canvas


# ============================================================
# NOISE MODELS
# ============================================================

def add_poisson_noise(
    image,
    rng,
    photon_level,
):
    """
    Shot-noise approximation.
    """

    image_f = np.clip(
        image.astype(np.float32),
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
    Detector/readout noise.
    """

    noise = rng.normal(
        0,
        sigma,
        image.shape,
    ).astype(
        np.float32
    )

    return (
        image
        + noise
    )


def add_low_frequency_variation(
    image,
    rng,
    amplitude,
):
    """
    Smooth spatial intensity variation.
    """

    h, w = image.shape

    small = rng.normal(
        0,
        1,
        (
            20,
            20,
        ),
    ).astype(
        np.float32
    )

    field = cv2.resize(
        small,
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
        + amplitude * field
    )


def add_row_variation(
    image,
    rng,
    amplitude,
):
    """
    SEM scan-line variation.
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
    row_amplitude,
):
    """
    Moderate SEM acquisition model.
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
    # Edge response
    # --------------------------------------------------------

    smooth = cv2.GaussianBlur(
        image_f,
        (
            0,
            0,
        ),
        sigmaX=1.25,
        sigmaY=1.25,
    )

    image_f += (
        0.55
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
    # Low-frequency variation
    # --------------------------------------------------------

    image_f = add_low_frequency_variation(
        image_f,
        rng,
        low_frequency_amplitude,
    )

    # --------------------------------------------------------
    # Row/scan variation
    # --------------------------------------------------------

    image_f = add_row_variation(
        image_f,
        rng,
        row_amplitude,
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
    print("=" * 72)
    print("DRIFT-SENSE — DRAM_04 GENERATOR")
    print("=" * 72)

    print()
    print("Architecture:")
    print(
        "Dense DRAM memory-cell array"
    )

    print()
    print("Unique landmark:")
    print(
        "One modified 3x3 cell neighborhood"
    )

    print()
    print("Noise:")
    print(
        "Moderate SEM shot + detector + "
        "low-frequency + scan-line variation"
    )

    # ========================================================
    # 1. Generate physical scene
    # ========================================================

    print()
    print(
        "1/6  Generating physical scene..."
    )

    physical_scene = (
        generate_physical_scene()
    )

    if physical_scene.shape != (
        10000,
        10000,
    ):

        raise RuntimeError(
            "Physical scene has incorrect dimensions."
        )

    # ========================================================
    # 2. Reference
    # ========================================================

    print(
        "2/6  Extracting 100x reference..."
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
            "Reference image has incorrect dimensions."
        )

    # ========================================================
    # 3. Search
    # ========================================================

    print(
        "3/6  Creating 10x search..."
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
            "Search image has incorrect dimensions."
        )

    # ========================================================
    # 4. SEM acquisition
    # ========================================================

    print(
        "4/6  Applying SEM degradation..."
    )

    reference_rng = np.random.default_rng(
        SEED + 100
    )

    search_rng = np.random.default_rng(
        SEED + 200
    )

    # --------------------------------------------------------
    # Reference:
    # cleaner high-magnification image
    # --------------------------------------------------------

    reference = simulate_sem(
        reference,
        reference_rng,
        blur_sigma=0.65,
        photon_level=850,
        detector_sigma=1.5,
        low_frequency_amplitude=2.8,
        row_amplitude=0.7,
    )

    # --------------------------------------------------------
    # Search:
    # stronger degradation
    # --------------------------------------------------------

    search = simulate_sem(
        search,
        search_rng,
        blur_sigma=0.55,
        photon_level=430,
        detector_sigma=2.6,
        low_frequency_amplitude=4.5,
        row_amplitude=1.3,
    )

    # ========================================================
    # 5. Ground truth
    # ========================================================

    print(
        "5/6  Creating ground truth..."
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
        + target_width / 2.0
    )

    gt_y = (
        search_y
        + target_height / 2.0
    )

    # ========================================================
    # 6. Save
    # ========================================================

    print(
        "6/6  Saving dataset..."
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

    if not cv2.imwrite(
        str(reference_path),
        reference,
    ):

        raise RuntimeError(
            "Could not save reference image."
        )

    if not cv2.imwrite(
        str(search_path),
        search,
    ):

        raise RuntimeError(
            "Could not save search image."
        )

    # ========================================================
    # Ground-truth manifest
    # ========================================================

    ground_truth = {

        "pair_id":
            "dram_04",

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
                "modified_3x3_cell_cluster",

            "landmark_size_nm":
                LANDMARK_SIZE_NM,

            "landmark_count":
                1,
        },

        "noise_model": {

            "shot_noise":
                True,

            "detector_noise":
                True,

            "low_frequency_variation":
                True,

            "scan_line_variation":
                True,

            "reference_photon_level":
                850,

            "search_photon_level":
                430,

            "reference_detector_sigma":
                1.5,

            "search_detector_sigma":
                2.6,
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
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 72)
    print("DRAM_04 GENERATED SUCCESSFULLY")
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
        "Target centre:"
    )

    print(
        f"  ({gt_x:.1f}, {gt_y:.1f}) px"
    )

    print()
    print(
        "Unique modified cluster count: 1"
    )

    print()
    print(
        f"Output: {OUTPUT_DIR}"
    )

    print()
    print("=" * 72)


if __name__ == "__main__":
    main()