#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-08 Synthetic Pair Generator
=================================

DRAM-08 introduces a different DRAM-like structural family.

Unlike DRAM-05/06/07, this pair does NOT use circular contacts
as the dominant feature.

STRUCTURE
---------

The physical scene contains:

    * rectangular storage-node structures
    * alternating horizontal / vertical support bars
    * periodic cell arrangement
    * 4 x 4 macro-array organization
    * dark separator streets

UNIQUE FEATURE
--------------

Exactly ONE small asymmetric L-shaped structural modification
is inserted into the physical scene.

The modification contains:

    * one missing rectangular element
    * one short extension
    * one small connecting bridge

The feature is deliberately SMALL.

Landmark:

    160 nm x 180 nm

At 10 nm / pixel:

    16 x 18 search pixels

IMAGE RELATIONSHIP
------------------

Physical scene:

    10000 x 10000 nm

Reference:

    1000 x 1000 px
    1 nm / pixel

Search:

    1000 x 1000 px
    10 nm / pixel

The reference is an exact crop of the same physical scene
used to create the search.

SEM SETTINGS
------------

These settings are based on the provided SEM controls.

Reference:

    beam spot       ~5 nm
    dose            2000
    drift           0
    row jitter      0
    low noise

Search:

    beam spot       ~6 nm
    dose            200
    raster drift    1.5 px
    row jitter      0.5 px
    mild charging
    mild speckle
    mild salt/pepper
    slight blur

The search is intentionally noisier than the reference,
but the structural pattern remains detectable.
"""

from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# GLOBAL SETTINGS
# ============================================================

IMAGE_SIZE = 1000

PHYSICAL_SIZE_NM = 10000

REFERENCE_SIZE_NM = 1000

REFERENCE_PIXEL_SIZE_NM = 1

SEARCH_PIXEL_SIZE_NM = 10

SEED = 20260817


# ============================================================
# TARGET LOCATION
# ============================================================

# Exact physical origin of the 1 um reference crop.

TARGET_X_NM = 6120

TARGET_Y_NM = 2780


# ============================================================
# STRUCTURE SCALE
# ============================================================

FEATURE_SCALE = 1.00


# ============================================================
# ARRAY BLOCK GEOMETRY
# ============================================================

ARRAY_BLOCK_SIZE_NM = 2600

SEPARATOR_WIDTH_NM = 320

ARRAY_MARGIN_NM = 100


# ============================================================
# CELL GEOMETRY
# ============================================================

CELL_PITCH_NM = 145

CELL_WIDTH_NM = 72

CELL_HEIGHT_NM = 72

BAR_WIDTH_NM = 11

BAR_LENGTH_NM = 105


# ============================================================
# UNIQUE LANDMARK
# ============================================================

LANDMARK_WIDTH_NM = 160

LANDMARK_HEIGHT_NM = 180

LANDMARK_EDGE_NM = 9


# ============================================================
# SEM PHYSICS
# ============================================================

# Beam spot

REFERENCE_BEAM_SPOT_NM = 5.0

SEARCH_BEAM_SPOT_NM = 6.0


# Pattern collapse threshold

PATTERN_COLLAPSE_THRESHOLD_NM = 10.0


# Acquisition dose

REFERENCE_DOSE = 2000.0

SEARCH_DOSE = 200.0


# Search geometric acquisition variation

SEARCH_RASTER_DRIFT_PX = 1.50

SEARCH_ROW_JITTER_PX = 0.50


# ============================================================
# DISTORTION
# ============================================================

REFERENCE_CD_BIAS_NM = 0.0

SEARCH_CD_BIAS_NM = 0.8

REFERENCE_CORNER_ROUNDING_PX = 0.5

SEARCH_CORNER_ROUNDING_PX = 1.1

REFERENCE_ASTIGMATISM = 1.00

SEARCH_ASTIGMATISM = 1.03

REFERENCE_BARREL = 0.0

SEARCH_BARREL = 0.008

REFERENCE_VIGNETTE = 0.01

SEARCH_VIGNETTE = 0.08

REFERENCE_GAMMA = 1.00

SEARCH_GAMMA = 1.05


# ============================================================
# NOISE
# ============================================================

REFERENCE_CHARGING_STREAKS = 0.0

SEARCH_CHARGING_STREAKS = 0.8

REFERENCE_CHARGING_INTENSITY = 0.0

SEARCH_CHARGING_INTENSITY = 0.7

REFERENCE_SPECKLE_SIGMA = 0.015

SEARCH_SPECKLE_SIGMA = 0.045

REFERENCE_SALT_PEPPER = 0.0

SEARCH_SALT_PEPPER = 0.0015


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_08"
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


# ============================================================
# RECTANGULAR CELL
# ============================================================

def draw_cell(
    image,
    cx,
    cy,
    rng,
):
    """
    Draw one rectangular DRAM-like storage structure.

             │
        ─────┼─────
             │
    """

    cx += rng.normal(
        0,
        1.0,
    )

    cy += rng.normal(
        0,
        1.0,
    )

    width = (
        CELL_WIDTH_NM
        + rng.normal(
            0,
            1.5,
        )
    )

    height = (
        CELL_HEIGHT_NM
        + rng.normal(
            0,
            1.5,
        )
    )

    x0 = int(
        round(
            cx - width / 2
        )
    )

    y0 = int(
        round(
            cy - height / 2
        )
    )

    x1 = int(
        round(
            cx + width / 2
        )
    )

    y1 = int(
        round(
            cy + height / 2
        )
    )

    # --------------------------------------------------------
    # Main rectangular storage node
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
        int(
            np.clip(
                rng.normal(
                    185,
                    4,
                ),
                0,
                255,
            )
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Horizontal support
    # --------------------------------------------------------

    draw_line(
        image,
        cx - BAR_LENGTH_NM / 2,
        cy,
        cx + BAR_LENGTH_NM / 2,
        cy,
        76,
        BAR_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Vertical support
    # --------------------------------------------------------

    draw_line(
        image,
        cx,
        cy - BAR_LENGTH_NM / 2,
        cx,
        cy + BAR_LENGTH_NM / 2,
        73,
        BAR_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Slight darker inner edge
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            x0 + 8,
            y0 + 8,
        ),
        (
            x1 - 8,
            y1 - 8,
        ),
        170,
        thickness=3,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# ARRAY BLOCK
# ============================================================

def draw_array_block(
    image,
    x0,
    y0,
    width,
    height,
    rng,
):
    """
    Generate one periodic DRAM macro block.
    """

    # --------------------------------------------------------
    # Block substrate
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
        42,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Periodic cells
    # --------------------------------------------------------

    start_x = (
        x0
        + CELL_PITCH_NM / 2
    )

    start_y = (
        y0
        + CELL_PITCH_NM / 2
    )

    end_x = (
        x0
        + width
        - CELL_PITCH_NM / 2
    )

    end_y = (
        y0
        + height
        - CELL_PITCH_NM / 2
    )

    y = start_y

    while y <= end_y:

        x = start_x

        while x <= end_x:

            draw_cell(
                image,
                x,
                y,
                rng,
            )

            x += CELL_PITCH_NM

        y += CELL_PITCH_NM


# ============================================================
# UNIQUE L-SHAPED STRUCTURE
# ============================================================

def draw_unique_pattern(
    image,
    center_x,
    center_y,
):
    """
    Draw exactly ONE small asymmetric L-shaped feature.

    The surrounding DRAM cells remain present.

    The landmark is intentionally compact.
    """

    w = LANDMARK_WIDTH_NM

    h = LANDMARK_HEIGHT_NM

    x0 = int(
        round(
            center_x - w / 2
        )
    )

    y0 = int(
        round(
            center_y - h / 2
        )
    )

    x1 = int(
        round(
            center_x + w / 2
        )
    )

    y1 = int(
        round(
            center_y + h / 2
        )
    )

    # ========================================================
    # First remove a small local region
    # ========================================================

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
        43,
        thickness=-1,
    )

    # ========================================================
    # L-shaped bright structure
    # ========================================================

    vertical_x = (
        center_x
        - 32
    )

    horizontal_y = (
        center_y
        + 25
    )

    # Vertical arm

    draw_line(
        image,
        vertical_x,
        y0 + 25,
        vertical_x,
        y1 - 20,
        194,
        25,
    )

    # Horizontal arm

    draw_line(
        image,
        vertical_x,
        horizontal_y,
        x1 - 22,
        horizontal_y,
        194,
        25,
    )

    # ========================================================
    # Small secondary bridge
    # ========================================================

    draw_line(
        image,
        center_x + 22,
        center_y - 50,
        center_x + 55,
        center_y - 18,
        155,
        12,
    )

    # ========================================================
    # Small dark notch
    # ========================================================

    cv2.rectangle(
        image,
        (
            int(center_x + 15),
            int(center_y - 38),
        ),
        (
            int(center_x + 38),
            int(center_y - 10),
        ),
        27,
        thickness=-1,
    )

    # ========================================================
    # Subtle perimeter
    # ========================================================

    draw_line(
        image,
        x0,
        y0,
        x1,
        y0,
        58,
        LANDMARK_EDGE_NM,
    )

    draw_line(
        image,
        x0,
        y1,
        x1,
        y1,
        56,
        LANDMARK_EDGE_NM,
    )

    draw_line(
        image,
        x0,
        y0,
        x0,
        y1,
        57,
        LANDMARK_EDGE_NM,
    )

    draw_line(
        image,
        x1,
        y0,
        x1,
        y1,
        55,
        LANDMARK_EDGE_NM,
    )


# ============================================================
# PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    """
    Create the complete 10 um x 10 um physical DRAM scene.
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
    # 4 x 4 ARRAY
    # ========================================================

    step = (
        ARRAY_BLOCK_SIZE_NM
        + SEPARATOR_WIDTH_NM
    )

    block_starts = [
        ARRAY_MARGIN_NM
        + i * step
        for i in range(4)
    ]

    for y0 in block_starts:

        for x0 in block_starts:

            draw_array_block(
                canvas,
                x0,
                y0,
                ARRAY_BLOCK_SIZE_NM,
                ARRAY_BLOCK_SIZE_NM,
                rng,
            )

    # ========================================================
    # SEPARATOR STREETS
    # ========================================================

    for x in [
        2700,
        5620,
        8540,
    ]:

        draw_line(
            canvas,
            x,
            0,
            x,
            PHYSICAL_SIZE_NM,
            34,
            SEPARATOR_WIDTH_NM,
        )

    for y in [
        2700,
        5620,
        8540,
    ]:

        draw_line(
            canvas,
            0,
            y,
            PHYSICAL_SIZE_NM,
            y,
            34,
            SEPARATOR_WIDTH_NM,
        )

    # ========================================================
    # BLOCK BORDERS
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
                        + ARRAY_BLOCK_SIZE_NM
                    ),
                    int(
                        y0
                        + ARRAY_BLOCK_SIZE_NM
                    ),
                ),
                55,
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

    draw_unique_pattern(
        canvas,
        landmark_center_x,
        landmark_center_y,
    )

    # ========================================================
    # PHYSICAL BEAM BLUR
    # ========================================================

    physical_sigma = (
        REFERENCE_BEAM_SPOT_NM
        / 10.0
    )

    canvas = cv2.GaussianBlur(
        canvas,
        (
            0,
            0,
        ),
        sigmaX=physical_sigma,
        sigmaY=physical_sigma,
    )

    return canvas


# ============================================================
# DOSE / SHOT NOISE
# ============================================================

def add_poisson_noise(
    image,
    rng,
    dose,
):
    """
    Dose-dependent shot noise.

    Higher dose -> cleaner image.
    """

    image_f = np.clip(
        image.astype(
            np.float32
        ),
        0,
        255,
    )

    normalized = (
        image_f / 255.0
    )

    # Scale dose to a useful photon/electron count.

    effective_counts = (
        normalized
        * dose
    )

    effective_counts = np.maximum(
        effective_counts,
        0.01,
    )

    noisy = rng.poisson(
        effective_counts
    ).astype(
        np.float32
    )

    noisy /= dose

    noisy *= 255.0

    return noisy


# ============================================================
# SPECKLE
# ============================================================

def add_speckle(
    image,
    rng,
    sigma,
):
    """
    Mild multiplicative SEM speckle.
    """

    if sigma <= 0:

        return image

    noise = rng.normal(
        1.0,
        sigma,
        image.shape,
    ).astype(
        np.float32
    )

    return image * noise


# ============================================================
# SALT AND PEPPER
# ============================================================

def add_salt_pepper(
    image,
    rng,
    probability,
):
    """
    Very small impulse-noise probability.
    """

    if probability <= 0:

        return image

    output = image.copy()

    random = rng.random(
        image.shape
    )

    salt = (
        random
        < probability / 2
    )

    pepper = (
        (
            random
            >= probability / 2
        )
        &
        (
            random
            < probability
        )
    )

    output[salt] = 255

    output[pepper] = 0

    return output


# ============================================================
# LOW-FREQUENCY BACKGROUND
# ============================================================

def add_low_frequency_variation(
    image,
    rng,
    amplitude,
):
    """
    Mild broad SEM illumination variation.
    """

    h, w = image.shape

    low = rng.normal(
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
        low,
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


# ============================================================
# ROW JITTER
# ============================================================

def apply_row_jitter(
    image,
    rng,
    jitter_px,
):
    """
    Shift each row slightly.

    Used mainly on the search image.
    """

    if jitter_px <= 0:

        return image

    h, w = image.shape

    output = np.empty_like(
        image
    )

    for y in range(h):

        shift = rng.normal(
            0,
            jitter_px,
        )

        x_coords = (
            np.arange(w)
            - shift
        )

        output[y] = np.interp(
            x_coords,
            np.arange(w),
            image[y].astype(
                np.float32
            ),
            left=float(
                image[y, 0]
            ),
            right=float(
                image[y, -1]
            ),
        )

    return output


# ============================================================
# RASTER DRIFT
# ============================================================

def apply_raster_drift(
    image,
    rng,
    drift_px,
):
    """
    Apply a small global affine drift/shear.
    """

    if drift_px <= 0:

        return image

    h, w = image.shape

    dx = rng.uniform(
        -drift_px,
        drift_px,
    )

    dy = rng.uniform(
        -drift_px,
        drift_px,
    )

    shear = rng.uniform(
        -0.0008,
        0.0008,
    )

    matrix = np.array(
        [
            [
                1.0,
                shear,
                dx,
            ],
            [
                0.0,
                1.0,
                dy,
            ],
        ],
        dtype=np.float32,
    )

    return cv2.warpAffine(
        image,
        matrix,
        (
            w,
            h,
        ),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


# ============================================================
# ASTIGMATISM
# ============================================================

def apply_astigmatism(
    image,
    ratio,
):
    """
    Slight anisotropic blur.
    """

    if abs(
        ratio - 1.0
    ) < 1e-6:

        return image

    sigma_x = 0.7

    sigma_y = (
        sigma_x * ratio
    )

    return cv2.GaussianBlur(
        image,
        (
            0,
            0,
        ),
        sigmaX=sigma_x,
        sigmaY=sigma_y,
    )


# ============================================================
# CORNER ROUNDING / FINAL BLUR
# ============================================================

def apply_corner_rounding(
    image,
    amount,
):
    """
    Small final smoothing representing CD/corner rounding.
    """

    if amount <= 0:

        return image

    return cv2.GaussianBlur(
        image,
        (
            0,
            0,
        ),
        sigmaX=amount,
        sigmaY=amount,
    )


# ============================================================
# VIGNETTE
# ============================================================

def apply_vignette(
    image,
    strength,
):
    """
    Very mild field-of-view intensity falloff.
    """

    if strength <= 0:

        return image

    h, w = image.shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    nx = (
        xx
        - w / 2
    ) / (
        w / 2
    )

    ny = (
        yy
        - h / 2
    ) / (
        h / 2
    )

    radius2 = (
        nx * nx
        + ny * ny
    )

    factor = (
        1.0
        - strength
        * np.clip(
            radius2 / 2.0,
            0,
            1,
        )
    )

    return (
        image
        * factor
    )


# ============================================================
# CHARGING STREAKS
# ============================================================

def add_charging_streaks(
    image,
    rng,
    intensity,
):
    """
    Add very weak broad horizontal charging streaks.
    """

    if intensity <= 0:

        return image

    output = image.astype(
        np.float32
    ).copy()

    h, w = image.shape

    number = max(
        1,
        int(
            h
            / 100
            * 0.8
        ),
    )

    for _ in range(number):

        y = rng.integers(
            0,
            h,
        )

        width = rng.integers(
            2,
            6,
        )

        amount = rng.uniform(
            -intensity,
            intensity,
        )

        y0 = max(
            0,
            y - width,
        )

        y1 = min(
            h,
            y + width + 1,
        )

        output[
            y0:y1,
            :
        ] += amount

    return output


# ============================================================
# GAMMA
# ============================================================

def apply_gamma(
    image,
    gamma,
):
    """
    Contrast curve.
    """

    if abs(
        gamma - 1.0
    ) < 1e-6:

        return image

    normalized = (
        np.clip(
            image,
            0,
            255,
        )
        / 255.0
    )

    corrected = (
        normalized
        ** (1.0 / gamma)
    )

    return (
        corrected
        * 255.0
    )


# ============================================================
# SEM ACQUISITION
# ============================================================

def simulate_sem(
    image,
    rng,
    *,
    beam_spot_nm,
    dose,
    detector_sigma,
    drift_px,
    row_jitter_px,
    corner_rounding,
    astigmatism,
    vignette,
    gamma,
    charging_intensity,
    speckle_sigma,
    salt_pepper_probability,
    background_amplitude,
):
    """
    Complete synthetic SEM acquisition model.
    """

    image_f = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Beam spot blur
    # --------------------------------------------------------

    sigma = (
        beam_spot_nm
        / 10.0
    )

    image_f = cv2.GaussianBlur(
        image_f,
        (
            0,
            0,
        ),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    # --------------------------------------------------------
    # Astigmatism
    # --------------------------------------------------------

    image_f = apply_astigmatism(
        image_f,
        astigmatism,
    )

    # --------------------------------------------------------
    # Corner rounding / CD smoothing
    # --------------------------------------------------------

    image_f = apply_corner_rounding(
        image_f,
        corner_rounding,
    )

    # --------------------------------------------------------
    # Raster drift
    # --------------------------------------------------------

    image_f = apply_raster_drift(
        image_f,
        rng,
        drift_px,
    )

    # --------------------------------------------------------
    # Row jitter
    # --------------------------------------------------------

    image_f = apply_row_jitter(
        image_f,
        rng,
        row_jitter_px,
    )

    # --------------------------------------------------------
    # Dose-dependent shot noise
    # --------------------------------------------------------

    image_f = add_poisson_noise(
        image_f,
        rng,
        dose,
    )

    # --------------------------------------------------------
    # Detector noise
    # --------------------------------------------------------

    image_f += rng.normal(
        0,
        detector_sigma,
        image_f.shape,
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # Low-frequency background
    # --------------------------------------------------------

    image_f = add_low_frequency_variation(
        image_f,
        rng,
        background_amplitude,
    )

    # --------------------------------------------------------
    # Charging
    # --------------------------------------------------------

    image_f = add_charging_streaks(
        image_f,
        rng,
        charging_intensity,
    )

    # --------------------------------------------------------
    # Speckle
    # --------------------------------------------------------

    image_f = add_speckle(
        image_f,
        rng,
        speckle_sigma,
    )

    # --------------------------------------------------------
    # Vignette
    # --------------------------------------------------------

    image_f = apply_vignette(
        image_f,
        vignette,
    )

    # --------------------------------------------------------
    # Gamma
    # --------------------------------------------------------

    image_f = apply_gamma(
        image_f,
        gamma,
    )

    # --------------------------------------------------------
    # Salt and pepper
    # --------------------------------------------------------

    image_f = add_salt_pepper(
        image_f,
        rng,
        salt_pepper_probability,
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
    print("=" * 78)
    print("DRIFT-SENSE — DRAM_08 GENERATOR")
    print("=" * 78)

    print()
    print("Architecture:")
    print(
        "Rectangular storage-node / support-bar DRAM"
    )

    print()
    print("Unique pattern:")
    print(
        "ONE small asymmetric L-shaped structure"
    )

    print()
    print("Landmark:")
    print(
        f"{LANDMARK_WIDTH_NM} x "
        f"{LANDMARK_HEIGHT_NM} nm"
    )

    print()
    print("Search-scale landmark:")
    print(
        f"{LANDMARK_WIDTH_NM / SEARCH_PIXEL_SIZE_NM:.1f} x "
        f"{LANDMARK_HEIGHT_NM / SEARCH_PIXEL_SIZE_NM:.1f} px"
    )

    print()
    print("Reference dose:")
    print(
        REFERENCE_DOSE
    )

    print()
    print("Search dose:")
    print(
        SEARCH_DOSE
    )

    # ========================================================
    # 1. PHYSICAL SCENE
    # ========================================================

    print()
    print(
        "[1/7] Generating physical DRAM scene..."
    )

    physical_scene = (
        generate_physical_scene()
    )

    if physical_scene.shape != (
        PHYSICAL_SIZE_NM,
        PHYSICAL_SIZE_NM,
    ):

        raise RuntimeError(
            "Invalid physical scene dimensions."
        )

    # ========================================================
    # 2. REFERENCE CROP
    # ========================================================

    print(
        "[2/7] Extracting 100x reference..."
    )

    reference = physical_scene[
        TARGET_Y_NM:
        TARGET_Y_NM
        + REFERENCE_SIZE_NM,

        TARGET_X_NM:
        TARGET_X_NM
        + REFERENCE_SIZE_NM,
    ].copy()

    if reference.shape != (
        1000,
        1000,
    ):

        raise RuntimeError(
            "Invalid reference dimensions."
        )

    # ========================================================
    # 3. WIDE SEARCH
    # ========================================================

    print(
        "[3/7] Creating 10x wide search..."
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
    # 4. REFERENCE SEM
    # ========================================================

    print(
        "[4/7] Simulating reference SEM..."
    )

    reference_rng = np.random.default_rng(
        SEED + 100
    )

    reference = simulate_sem(
        reference,
        reference_rng,
        beam_spot_nm=REFERENCE_BEAM_SPOT_NM,
        dose=REFERENCE_DOSE,
        detector_sigma=0.55,
        drift_px=0.0,
        row_jitter_px=0.0,
        corner_rounding=REFERENCE_CORNER_ROUNDING_PX,
        astigmatism=REFERENCE_ASTIGMATISM,
        vignette=REFERENCE_VIGNETTE,
        gamma=REFERENCE_GAMMA,
        charging_intensity=REFERENCE_CHARGING_INTENSITY,
        speckle_sigma=REFERENCE_SPECKLE_SIGMA,
        salt_pepper_probability=REFERENCE_SALT_PEPPER,
        background_amplitude=0.8,
    )

    # ========================================================
    # 5. SEARCH SEM
    # ========================================================

    print(
        "[5/7] Simulating noisy search SEM..."
    )

    search_rng = np.random.default_rng(
        SEED + 200
    )

    search = simulate_sem(
        search,
        search_rng,
        beam_spot_nm=SEARCH_BEAM_SPOT_NM,
        dose=SEARCH_DOSE,
        detector_sigma=1.45,
        drift_px=SEARCH_RASTER_DRIFT_PX,
        row_jitter_px=SEARCH_ROW_JITTER_PX,
        corner_rounding=SEARCH_CORNER_ROUNDING_PX,
        astigmatism=SEARCH_ASTIGMATISM,
        vignette=SEARCH_VIGNETTE,
        gamma=SEARCH_GAMMA,
        charging_intensity=SEARCH_CHARGING_INTENSITY,
        speckle_sigma=SEARCH_SPECKLE_SIGMA,
        salt_pepper_probability=SEARCH_SALT_PEPPER,
        background_amplitude=2.5,
    )

    # ========================================================
    # 6. GROUND TRUTH
    # ========================================================

    print(
        "[6/7] Creating ground truth..."
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

    center_x = (
        search_x
        + target_width / 2
    )

    center_y = (
        search_y
        + target_height / 2
    )

    # Actual landmark size at search scale.

    landmark_width_px = (
        LANDMARK_WIDTH_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    landmark_height_px = (
        LANDMARK_HEIGHT_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    # ========================================================
    # 7. SAVE
    # ========================================================

    print(
        "[7/7] Saving DRAM-08..."
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
    # Images
    # --------------------------------------------------------

    if not cv2.imwrite(
        str(reference_path),
        reference,
    ):

        raise RuntimeError(
            "Failed to save reference."
        )

    if not cv2.imwrite(
        str(search_path),
        search,
    ):

        raise RuntimeError(
            "Failed to save search."
        )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "pair_id":
            "dram_08",

        "architecture":
            "DRAM",

        "seed":
            SEED,

        "structure_family":
            "rectangular_storage_node_array",

        "reference": {

            "filename":
                "reference_100x.png",

            "width_px":
                1000,

            "height_px":
                1000,

            "pixel_size_nm":
                1,

            "magnification":
                "100x",

            "physical_fov_um":
                [
                    1.0,
                    1.0,
                ],
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

            "magnification":
                "10x",

            "physical_fov_um":
                [
                    10.0,
                    10.0,
                ],
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
                    center_x,
                    center_y,
                ],

            "landmark_type":
                "small_asymmetric_L_structure",

            "landmark_physical_size_nm":
                [
                    LANDMARK_WIDTH_NM,
                    LANDMARK_HEIGHT_NM,
                ],

            "landmark_search_size_px":
                [
                    landmark_width_px,
                    landmark_height_px,
                ],

            "landmark_count":
                1,
        },

        "structure": {

            "feature_size_scale":
                FEATURE_SCALE,

            "array_block_size_nm":
                ARRAY_BLOCK_SIZE_NM,

            "separator_width_nm":
                SEPARATOR_WIDTH_NM,

            "cell_pitch_nm":
                CELL_PITCH_NM,

            "cell_width_nm":
                CELL_WIDTH_NM,

            "cell_height_nm":
                CELL_HEIGHT_NM,

            "support_bar_width_nm":
                BAR_WIDTH_NM,
        },

        "sem_physics": {

            "pattern_collapse_threshold_nm":
                PATTERN_COLLAPSE_THRESHOLD_NM,

            "reference_beam_spot_nm":
                REFERENCE_BEAM_SPOT_NM,

            "search_beam_spot_nm":
                SEARCH_BEAM_SPOT_NM,

            "reference_dose":
                REFERENCE_DOSE,

            "search_dose":
                SEARCH_DOSE,

            "search_raster_drift_px":
                SEARCH_RASTER_DRIFT_PX,

            "search_row_jitter_px":
                SEARCH_ROW_JITTER_PX,
        },

        "distortion": {

            "reference_cd_bias_nm":
                REFERENCE_CD_BIAS_NM,

            "search_cd_bias_nm":
                SEARCH_CD_BIAS_NM,

            "reference_corner_rounding_px":
                REFERENCE_CORNER_ROUNDING_PX,

            "search_corner_rounding_px":
                SEARCH_CORNER_ROUNDING_PX,

            "reference_astigmatism_ratio":
                REFERENCE_ASTIGMATISM,

            "search_astigmatism_ratio":
                SEARCH_ASTIGMATISM,

            "reference_barrel_distortion":
                REFERENCE_BARREL,

            "search_barrel_distortion":
                SEARCH_BARREL,

            "reference_vignette":
                REFERENCE_VIGNETTE,

            "search_vignette":
                SEARCH_VIGNETTE,

            "reference_gamma":
                REFERENCE_GAMMA,

            "search_gamma":
                SEARCH_GAMMA,
        },

        "noise": {

            "reference_charging_intensity":
                REFERENCE_CHARGING_INTENSITY,

            "search_charging_intensity":
                SEARCH_CHARGING_INTENSITY,

            "reference_speckle_sigma":
                REFERENCE_SPECKLE_SIGMA,

            "search_speckle_sigma":
                SEARCH_SPECKLE_SIGMA,

            "reference_salt_pepper":
                REFERENCE_SALT_PEPPER,

            "search_salt_pepper":
                SEARCH_SALT_PEPPER,
        },

        "generation": {

            "same_physical_scene":
                True,

            "reference_is_exact_crop":
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
    print("=" * 78)
    print("DRAM_08 GENERATED SUCCESSFULLY")
    print("=" * 78)

    print()
    print(
        f"Reference : {reference_path}"
    )

    print(
        f"Search    : {search_path}"
    )

    print()
    print("Reference physical origin:")
    print(
        f"  ({TARGET_X_NM}, {TARGET_Y_NM}) nm"
    )

    print()
    print("Search target region:")
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
    print("Target center:")
    print(
        f"  ({center_x:.1f}, {center_y:.1f}) px"
    )

    print()
    print("Actual unique pattern:")
    print(
        f"  {LANDMARK_WIDTH_NM} x "
        f"{LANDMARK_HEIGHT_NM} nm"
    )

    print(
        f"  {landmark_width_px:.1f} x "
        f"{landmark_height_px:.1f} search px"
    )

    print()
    print("Reference dose:")
    print(
        f"  {REFERENCE_DOSE}"
    )

    print("Search dose:")
    print(
        f"  {SEARCH_DOSE}"
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
    print("=" * 78)


if __name__ == "__main__":
    main()