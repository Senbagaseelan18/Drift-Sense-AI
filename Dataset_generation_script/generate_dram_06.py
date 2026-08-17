#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-06 Synthetic Pair Generator
=================================

DRAM-06 keeps the dense circular-contact DRAM background
from DRAM-05 but introduces a new type of unique local
structural landmark.

BACKGROUND
----------

    * dense circular contacts
    * highly periodic rows and columns
    * 4 x 4 memory-array blocks
    * dark isolation streets
    * subtle contact-size variation
    * mild placement variation

UNIQUE LANDMARK
---------------

Exactly ONE local structural modification exists:

    * one missing circular contact
    * one shifted contact
    * one additional contact
    * one short diagonal connector
    * one small bridge structure

The landmark is intentionally compact.

This means the reference is still dominated by normal DRAM
circular-contact structure.

PHYSICAL RELATIONSHIP
---------------------

Physical scene:

    10000 x 10000 nm
    1 nm / pixel

Reference:

    1000 x 1000 px
    1 nm / pixel

Search:

    1000 x 1000 px
    10 nm / pixel

The reference is extracted from the same physical scene
used to generate the search.

Therefore the pair has an exact 10:1 physical scale
relationship.

NOISE
-----

Moderate SEM-like degradation:

    * Poisson shot noise
    * detector/readout noise
    * low-frequency background variation
    * scan-line variation
    * mild charging-like intensity variation
    * different degradation levels for reference/search
"""


from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# GLOBAL CONFIGURATION
# ============================================================

IMAGE_SIZE = 1000

PHYSICAL_SIZE_NM = 10000

REFERENCE_SIZE_NM = 1000

REFERENCE_PIXEL_SIZE_NM = 1

SEARCH_PIXEL_SIZE_NM = 10

SEED = 20260815


# ============================================================
# TARGET LOCATION
# ============================================================

# Different target location from previous DRAM pairs.

TARGET_X_NM = 2940
TARGET_Y_NM = 6120


# ============================================================
# PERIODIC DRAM CONTACT ARRAY
# ============================================================

CONTACT_PITCH_NM = 135

CONTACT_RADIUS_NM = 29

SUPPORT_LINE_WIDTH_NM = 10

SUPPORT_HALF_LENGTH_NM = 48


# ============================================================
# UNIQUE LANDMARK SIZE
# ============================================================

LANDMARK_WIDTH_NM = 320

LANDMARK_HEIGHT_NM = 300


# ============================================================
# ARRAY GEOMETRY
# ============================================================

BLOCK_SIZE_NM = 2150

STREET_WIDTH_NM = 260

ARRAY_MARGIN_NM = 100


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_06"
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
        int(
            np.clip(
                intensity,
                0,
                255,
            )
        ),
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
    Draw one circular DRAM contact.

    The outer circle and slightly brighter inner region
    produce a more realistic SEM-like contact appearance.
    """

    x = int(round(x))
    y = int(round(y))

    radius = max(
        2,
        int(round(radius)),
    )

    intensity = int(
        np.clip(
            intensity,
            0,
            255,
        )
    )

    # Outer contact.

    cv2.circle(
        image,
        (
            x,
            y,
        ),
        radius,
        intensity,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # Soft bright center.

    inner_radius = max(
        1,
        int(
            round(
                radius * 0.45
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
                intensity + 8,
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
    cx,
    cy,
    row,
    col,
    rng,
):
    """
    Draw one normal circular-contact DRAM cell.

    The circles are the dominant visual structure.
    """

    # --------------------------------------------------------
    # Tiny fabrication variation
    # --------------------------------------------------------

    local_x = (
        cx
        + rng.normal(
            0,
            1.4,
        )
    )

    local_y = (
        cy
        + rng.normal(
            0,
            1.4,
        )
    )

    # --------------------------------------------------------
    # Subtle horizontal support
    # --------------------------------------------------------

    draw_line(
        image,
        local_x
        - SUPPORT_HALF_LENGTH_NM,
        local_y,
        local_x
        + SUPPORT_HALF_LENGTH_NM,
        local_y,
        rng.normal(
            75,
            2.5,
        ),
        SUPPORT_LINE_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Subtle vertical support
    # --------------------------------------------------------

    draw_line(
        image,
        local_x,
        local_y
        - SUPPORT_HALF_LENGTH_NM,
        local_x,
        local_y
        + SUPPORT_HALF_LENGTH_NM,
        rng.normal(
            73,
            2.5,
        ),
        SUPPORT_LINE_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Main circular contact
    # --------------------------------------------------------

    radius = rng.normal(
        CONTACT_RADIUS_NM,
        1.1,
    )

    intensity = rng.normal(
        195,
        4.5,
    )

    draw_contact(
        image,
        local_x,
        local_y,
        radius,
        intensity,
    )


# ============================================================
# MEMORY ARRAY BLOCK
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
    Generate one dense circular-contact memory block.
    """

    # --------------------------------------------------------
    # Dark substrate/background
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
        45,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Contact array
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

            draw_normal_cell(
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
# UNIQUE LOCAL LANDMARK
# ============================================================

def draw_unique_local_pattern(
    image,
    center_x,
    center_y,
):
    """
    Draw exactly ONE unique structural pattern.

    Normal array:

        O O O O
        O O O O
        O O O O
        O O O O

    Unique region:

        O O O O
        O   O O
        O  /  O
        O O--O

    The landmark is deliberately compact and asymmetric.
    """

    pitch = CONTACT_PITCH_NM

    # ========================================================
    # 3 x 3 local reference pattern
    # ========================================================

    positions = [
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
    # Draw surrounding normal contacts first.
    # --------------------------------------------------------

    for dx, dy in positions:

        if (
            dx == 0
            and dy == 0
        ):
            continue

        px = (
            center_x
            + dx * pitch
        )

        py = (
            center_y
            + dy * pitch
        )

        draw_contact(
            image,
            px,
            py,
            CONTACT_RADIUS_NM,
            196,
        )

    # ========================================================
    # UNIQUE MODIFICATION 1
    # Missing central contact
    # ========================================================

    # Instead of a bright circle, create a subtle recessed
    # circular region.

    cv2.circle(
        image,
        (
            int(center_x),
            int(center_y),
        ),
        int(
            CONTACT_RADIUS_NM
            * 0.95
        ),
        54,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # ========================================================
    # UNIQUE MODIFICATION 2
    # Shifted contact
    # ========================================================

    shifted_x = (
        center_x
        + 34
    )

    shifted_y = (
        center_y
        - 31
    )

    draw_contact(
        image,
        shifted_x,
        shifted_y,
        CONTACT_RADIUS_NM * 0.82,
        218,
    )

    # ========================================================
    # UNIQUE MODIFICATION 3
    # Additional contact
    # ========================================================

    extra_x = (
        center_x
        + 33
    )

    extra_y = (
        center_y
        + 70
    )

    draw_contact(
        image,
        extra_x,
        extra_y,
        CONTACT_RADIUS_NM * 0.70,
        210,
    )

    # ========================================================
    # UNIQUE MODIFICATION 4
    # Short diagonal connector
    # ========================================================

    draw_line(
        image,
        center_x - 38,
        center_y + 40,
        center_x + 40,
        center_y - 37,
        145,
        15,
    )

    # ========================================================
    # UNIQUE MODIFICATION 5
    # Small horizontal bridge
    # ========================================================

    draw_line(
        image,
        center_x + 20,
        center_y + 70,
        center_x + 75,
        center_y + 70,
        126,
        13,
    )

    # ========================================================
    # Small recessed halo
    # ========================================================

    cv2.circle(
        image,
        (
            int(center_x),
            int(center_y),
        ),
        112,
        65,
        thickness=3,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# COMPLETE PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    """
    Generate the complete 10,000 x 10,000 nm physical scene.
    """

    rng = np.random.default_rng(
        SEED
    )

    canvas = np.full(
        (
            PHYSICAL_SIZE_NM,
            PHYSICAL_SIZE_NM,
        ),
        33,
        dtype=np.uint8,
    )

    # ========================================================
    # 4 x 4 MEMORY ARRAY
    # ========================================================

    block_starts = [
        ARRAY_MARGIN_NM
        + i
        * (
            BLOCK_SIZE_NM
            + STREET_WIDTH_NM
        )
        for i in range(4)
    ]

    for y0 in block_starts:

        for x0 in block_starts:

            draw_memory_block(
                canvas,
                x0,
                y0,
                BLOCK_SIZE_NM,
                BLOCK_SIZE_NM,
                rng,
            )

    # ========================================================
    # DARK STREETS
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
            35,
            STREET_WIDTH_NM,
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
            35,
            STREET_WIDTH_NM,
        )

    # ========================================================
    # SUBTLE ARRAY BORDERS
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
                        + BLOCK_SIZE_NM
                    ),
                    int(
                        y0
                        + BLOCK_SIZE_NM
                    ),
                ),
                57,
                thickness=7,
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

    draw_unique_local_pattern(
        canvas,
        landmark_center_x,
        landmark_center_y,
    )

    # ========================================================
    # PHYSICAL BLUR
    # ========================================================

    canvas = cv2.GaussianBlur(
        canvas,
        (
            0,
            0,
        ),
        sigmaX=0.58,
        sigmaY=0.58,
    )

    return canvas


# ============================================================
# NOISE FUNCTIONS
# ============================================================

def add_poisson_noise(
    image,
    rng,
    photon_level,
):
    """
    SEM-like shot noise.
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
    Smooth illumination / charging variation.
    """

    h, w = image.shape

    low_res = rng.normal(
        0,
        1,
        (
            22,
            22,
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


def add_horizontal_scan_variation(
    image,
    rng,
    amplitude,
):
    """
    Mild SEM scan-line variation.
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


def add_mild_charging(
    image,
    rng,
    amplitude,
):
    """
    Smooth localized charging-like variation.

    This is intentionally weak so the periodic DRAM
    structure remains readable.
    """

    h, w = image.shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    # Slightly off-center charging region.

    cx = (
        w
        * rng.uniform(
            0.35,
            0.65,
        )
    )

    cy = (
        h
        * rng.uniform(
            0.35,
            0.65,
        )
    )

    sx = (
        w
        * rng.uniform(
            0.30,
            0.48,
        )
    )

    sy = (
        h
        * rng.uniform(
            0.30,
            0.48,
        )
    )

    field = np.exp(
        -(
            (
                (xx - cx) ** 2
                / (2 * sx * sx)
            )
            +
            (
                (yy - cy) ** 2
                / (2 * sy * sy)
            )
        )
    )

    # Center around zero.

    field -= field.mean()

    return (
        image
        + amplitude
        * field
    )


# ============================================================
# SEM SIMULATION
# ============================================================

def simulate_sem(
    image,
    rng,
    blur_sigma,
    photon_level,
    detector_sigma,
    low_frequency_amplitude,
    scan_amplitude,
    charging_amplitude,
):
    """
    Complete SEM-like acquisition model.
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
    # Very mild edge response
    # --------------------------------------------------------

    smooth = cv2.GaussianBlur(
        image_f,
        (
            0,
            0,
        ),
        sigmaX=1.2,
        sigmaY=1.2,
    )

    image_f += (
        0.28
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
    # Scan-line variation
    # --------------------------------------------------------

    image_f = add_horizontal_scan_variation(
        image_f,
        rng,
        scan_amplitude,
    )

    # --------------------------------------------------------
    # Charging-like variation
    # --------------------------------------------------------

    image_f = add_mild_charging(
        image_f,
        rng,
        charging_amplitude,
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
    print("=" * 74)
    print("DRIFT-SENSE — DRAM_06 GENERATOR")
    print("=" * 74)

    print()
    print("Background:")
    print(
        "Dense circular-contact DRAM array"
    )

    print()
    print("Unique landmark:")
    print(
        "Small asymmetric contact/connector pattern"
    )

    print()
    print("Noise:")
    print(
        "Shot + detector + background + "
        "scan-line + charging variation"
    )

    # ========================================================
    # STEP 1
    # ========================================================

    print()
    print(
        "[1/6] Generating physical scene..."
    )

    physical_scene = (
        generate_physical_scene()
    )

    if physical_scene.shape != (
        PHYSICAL_SIZE_NM,
        PHYSICAL_SIZE_NM,
    ):

        raise RuntimeError(
            "Physical scene dimensions are incorrect."
        )

    # ========================================================
    # STEP 2
    # ========================================================

    print(
        "[2/6] Extracting 100x reference..."
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
            "Reference dimensions are incorrect."
        )

    # ========================================================
    # STEP 3
    # ========================================================

    print(
        "[3/6] Creating 10x search..."
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
            "Search dimensions are incorrect."
        )

    # ========================================================
    # STEP 4
    # ========================================================

    print(
        "[4/6] Applying SEM degradation..."
    )

    reference_rng = np.random.default_rng(
        SEED + 101
    )

    search_rng = np.random.default_rng(
        SEED + 202
    )

    # --------------------------------------------------------
    # Reference
    # --------------------------------------------------------

    reference = simulate_sem(
        reference,
        reference_rng,
        blur_sigma=0.62,
        photon_level=900,
        detector_sigma=1.3,
        low_frequency_amplitude=2.2,
        scan_amplitude=0.55,
        charging_amplitude=1.8,
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search = simulate_sem(
        search,
        search_rng,
        blur_sigma=0.52,
        photon_level=480,
        detector_sigma=2.2,
        low_frequency_amplitude=4.0,
        scan_amplitude=1.0,
        charging_amplitude=3.0,
    )

    # ========================================================
    # STEP 5
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

    gt_center_x = (
        search_x
        + target_width / 2
    )

    gt_center_y = (
        search_y
        + target_height / 2
    )

    # ========================================================
    # STEP 6
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

    # --------------------------------------------------------
    # Save images
    # --------------------------------------------------------

    if not cv2.imwrite(
        str(reference_path),
        reference,
    ):

        raise RuntimeError(
            "Failed to save reference image."
        )

    if not cv2.imwrite(
        str(search_path),
        search,
    ):

        raise RuntimeError(
            "Failed to save search image."
        )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "pair_id":
            "dram_06",

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
                    gt_center_x,
                    gt_center_y,
                ],

            "landmark_type":
                "asymmetric_contact_cluster",

            "landmark_physical_size_nm":
                [
                    LANDMARK_WIDTH_NM,
                    LANDMARK_HEIGHT_NM,
                ],

            "landmark_count":
                1,
        },

        "pattern": {

            "background":
                "periodic_circular_contacts",

            "contact_pitch_nm":
                CONTACT_PITCH_NM,

            "contact_radius_nm":
                CONTACT_RADIUS_NM,

            "block_size_nm":
                BLOCK_SIZE_NM,

            "street_width_nm":
                STREET_WIDTH_NM,
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

            "charging_like_variation":
                True,

            "reference_photon_level":
                900,

            "search_photon_level":
                480,

            "reference_detector_sigma":
                1.3,

            "search_detector_sigma":
                2.2,
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
    print("=" * 74)
    print("DRAM_06 GENERATED SUCCESSFULLY")
    print("=" * 74)

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
        f"  ({gt_center_x:.1f}, "
        f"{gt_center_y:.1f}) px"
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
    print("=" * 74)


if __name__ == "__main__":
    main()