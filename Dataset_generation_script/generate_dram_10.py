#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-10 Synthetic Pair Generator
================================

DRAM-10 uses a new structural family and a new SEM noise
model compared with DRAM-07 / DRAM-08 / DRAM-09.

STRUCTURE
---------

    * elongated recessed storage trenches
    * alternating vertical bitline-like rails
    * short horizontal coupling bridges
    * staggered trench rows
    * repeated macro blocks
    * dark macro separator streets

UNIQUE PATTERN
--------------

Exactly ONE small asymmetric T-shaped trench interruption.

Physical size:

    150 nm x 170 nm

Search scale:

    15 x 17 pixels

The feature is embedded into a normal cell and is NOT
represented as a giant artificial box.

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

The reference is an exact crop from the same physical scene.

NOISE MODEL
-----------

DRAM-10 intentionally uses a different noise family.

REFERENCE:

    high dose
    weak fixed-pattern response
    very low readout noise
    almost no acquisition distortion

SEARCH:

    lower dose
    detector fixed-pattern variation
    vertical banding
    periodic scan modulation
    low-frequency illumination field
    multiplicative gain variation
    correlated grain
    readout noise
    mild charging gradient
    anisotropic blur
    sparse hot/dead pixels

The search noise is intentionally different from DRAM-08
and DRAM-09.
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

SEARCH_PIXEL_SIZE_NM = 10

REFERENCE_PIXEL_SIZE_NM = 1

SEED = 20260819


# ============================================================
# TARGET
# ============================================================

TARGET_X_NM = 3760

TARGET_Y_NM = 6180


# ============================================================
# MACRO ARRAY
# ============================================================

ARRAY_BLOCK_SIZE_NM = 2400

SEPARATOR_WIDTH_NM = 360

ARRAY_MARGIN_NM = 120


# ============================================================
# DRAM-10 TRENCH GEOMETRY
# ============================================================

TRENCH_PITCH_X_NM = 125

TRENCH_PITCH_Y_NM = 145

TRENCH_WIDTH_NM = 38

TRENCH_HEIGHT_NM = 82

RAIL_WIDTH_NM = 9

BRIDGE_WIDTH_NM = 8

BRIDGE_LENGTH_NM = 70

ROW_STAGGER_NM = 34


# ============================================================
# UNIQUE FEATURE
# ============================================================

LANDMARK_WIDTH_NM = 150

LANDMARK_HEIGHT_NM = 170

LANDMARK_EDGE_NM = 6


# ============================================================
# SEM PARAMETERS
# ============================================================

REFERENCE_BEAM_SPOT_NM = 4.0

SEARCH_BEAM_SPOT_NM = 6.0

PATTERN_COLLAPSE_THRESHOLD_NM = 10.0


# ============================================================
# DOSE
# ============================================================

REFERENCE_DOSE = 2000.0

SEARCH_DOSE = 200.0


# ============================================================
# SEARCH DISTORTION
# ============================================================

SEARCH_RASTER_DRIFT_PX = 1.25

SEARCH_ROW_JITTER_PX = 0.35

SEARCH_ASTIGMATISM = 1.06

SEARCH_GAMMA = 1.06

SEARCH_VIGNETTE = 0.09


# ============================================================
# NEW DRAM-10 NOISE MODEL
# ============================================================

REFERENCE_READOUT_SIGMA = 0.35

SEARCH_READOUT_SIGMA = 1.30

REFERENCE_FIXED_PATTERN = 0.25

SEARCH_FIXED_PATTERN = 2.2

REFERENCE_VERTICAL_BANDING = 0.10

SEARCH_VERTICAL_BANDING = 2.0

REFERENCE_SCAN_MODULATION = 0.10

SEARCH_SCAN_MODULATION = 1.8

REFERENCE_GRAIN = 0.008

SEARCH_GRAIN = 0.035

REFERENCE_ILLUMINATION = 0.5

SEARCH_ILLUMINATION = 3.0

REFERENCE_CHARGING = 0.0

SEARCH_CHARGING = 2.0

SEARCH_HOT_PIXEL_PROBABILITY = 0.0007

SEARCH_DEAD_PIXEL_PROBABILITY = 0.0004


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_10"
)


# ============================================================
# DRAW LINE
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
# DRAW RECESSED TRENCH
# ============================================================

def draw_trench(
    image,
    cx,
    cy,
    rng,
):
    """
    Draw one elongated recessed DRAM storage trench.

    The dominant pattern is intentionally elongated rather
    than circular or pillar-like.
    """

    cx += rng.normal(
        0,
        1.1,
    )

    cy += rng.normal(
        0,
        1.1,
    )

    width = (
        TRENCH_WIDTH_NM
        + rng.normal(
            0,
            1.2,
        )
    )

    height = (
        TRENCH_HEIGHT_NM
        + rng.normal(
            0,
            2.0,
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
    # Outer raised response
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            x0 - 5,
            y0 - 5,
        ),
        (
            x1 + 5,
            y1 + 5,
        ),
        76,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Dark recessed trench
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
            rng.normal(
                64,
                3,
            )
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Slightly brighter inner bottom
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            x0 + 5,
            y0 + 7,
        ),
        (
            x1 - 5,
            y1 - 7,
        ),
        int(
            rng.normal(
                72,
                2,
            )
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# NORMAL DRAM CELL
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
    Draw a normal DRAM-10 cell.

    Layout:

          │
          │
       ┌─────┐
       │     │
       │     │
       └─────┘
          │
          │

    with a short lateral bridge.
    """

    # --------------------------------------------------------
    # Main recessed trench
    # --------------------------------------------------------

    draw_trench(
        image,
        cx,
        cy,
        rng,
    )

    # --------------------------------------------------------
    # Vertical rail
    # --------------------------------------------------------

    draw_line(
        image,
        cx,
        cy - TRENCH_HEIGHT_NM / 2 - 28,
        cx,
        cy + TRENCH_HEIGHT_NM / 2 + 28,
        rng.normal(
            82,
            2,
        ),
        RAIL_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Horizontal coupling bridge
    # --------------------------------------------------------

    bridge_y = (
        cy
        + rng.normal(
            0,
            2,
        )
    )

    direction = (
        -1
        if col % 2
        else 1
    )

    draw_line(
        image,
        cx,
        bridge_y,
        cx
        + direction
        * BRIDGE_LENGTH_NM,
        bridge_y,
        rng.normal(
            73,
            2,
        ),
        BRIDGE_WIDTH_NM,
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
    Draw one staggered DRAM-10 memory block.
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
        38,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Staggered rows
    # --------------------------------------------------------

    row = 0

    y = (
        y0
        + TRENCH_PITCH_Y_NM / 2
    )

    while y < (
        y0
        + height
        - TRENCH_PITCH_Y_NM / 2
    ):

        offset = (
            ROW_STAGGER_NM
            if row % 2
            else 0
        )

        x = (
            x0
            + TRENCH_PITCH_X_NM / 2
            + offset
        )

        col = 0

        while x < (
            x0
            + width
        ):

            draw_dram_cell(
                image,
                x,
                y,
                row,
                col,
                rng,
            )

            x += TRENCH_PITCH_X_NM

            col += 1

        y += TRENCH_PITCH_Y_NM

        row += 1


# ============================================================
# UNIQUE T-SHAPED INTERRUPTION
# ============================================================

def draw_unique_feature(
    image,
    center_x,
    center_y,
):
    """
    Insert ONE small T-shaped asymmetric trench structure.

    The feature is deliberately compact.

             │
             │
        ─────┼─────
             │
          ███
             │

    It replaces a small normal-cell region rather than
    appearing as an isolated giant black box.
    """

    w = LANDMARK_WIDTH_NM

    h = LANDMARK_HEIGHT_NM

    x0 = (
        center_x
        - w / 2
    )

    y0 = (
        center_y
        - h / 2
    )

    x1 = (
        center_x
        + w / 2
    )

    y1 = (
        center_y
        + h / 2
    )

    # ========================================================
    # Clear local normal cell
    # ========================================================

    cv2.rectangle(
        image,
        (
            int(x0),
            int(y0),
        ),
        (
            int(x1),
            int(y1),
        ),
        40,
        thickness=-1,
    )

    # ========================================================
    # Central vertical recessed trench
    # ========================================================

    draw_line(
        image,
        center_x,
        y0 + 15,
        center_x,
        y1 - 18,
        67,
        24,
    )

    # ========================================================
    # Horizontal T-head
    # ========================================================

    draw_line(
        image,
        center_x - 55,
        center_y - 5,
        center_x + 48,
        center_y - 5,
        69,
        22,
    )

    # ========================================================
    # Short offset secondary arm
    # ========================================================

    draw_line(
        image,
        center_x + 24,
        center_y + 15,
        center_x + 54,
        center_y + 34,
        82,
        10,
    )

    # ========================================================
    # Bright asymmetric edge
    # ========================================================

    draw_line(
        image,
        center_x - 42,
        center_y - 17,
        center_x + 39,
        center_y - 17,
        105,
        5,
    )

    # ========================================================
    # Small dark notch
    # ========================================================

    cv2.rectangle(
        image,
        (
            int(center_x - 13),
            int(center_y + 13),
        ),
        (
            int(center_x + 13),
            int(center_y + 31),
        ),
        29,
        thickness=-1,
    )

    # ========================================================
    # Subtle local boundary
    # ========================================================

    draw_line(
        image,
        x0,
        y0,
        x1,
        y0,
        54,
        LANDMARK_EDGE_NM,
    )

    draw_line(
        image,
        x0,
        y1,
        x1,
        y1,
        52,
        LANDMARK_EDGE_NM,
    )

    draw_line(
        image,
        x0,
        y0,
        x0,
        y1,
        53,
        LANDMARK_EDGE_NM,
    )

    draw_line(
        image,
        x1,
        y0,
        x1,
        y1,
        51,
        LANDMARK_EDGE_NM,
    )


# ============================================================
# PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    """
    Generate the complete 10 um x 10 um physical scene.
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
    # 4 x 4 MACRO ARRAY
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
        2880,
        5640,
        8400,
    ]:

        draw_line(
            canvas,
            x,
            0,
            x,
            PHYSICAL_SIZE_NM,
            32,
            SEPARATOR_WIDTH_NM,
        )

    for y in [
        2880,
        5640,
        8400,
    ]:

        draw_line(
            canvas,
            0,
            y,
            PHYSICAL_SIZE_NM,
            y,
            32,
            SEPARATOR_WIDTH_NM,
        )

    # ========================================================
    # MACRO BOUNDARIES
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
                51,
                thickness=6,
                lineType=cv2.LINE_AA,
            )

    # ========================================================
    # UNIQUE SMALL FEATURE
    # ========================================================

    landmark_center_x = (
        TARGET_X_NM
        + REFERENCE_SIZE_NM / 2
    )

    landmark_center_y = (
        TARGET_Y_NM
        + REFERENCE_SIZE_NM / 2
    )

    draw_unique_feature(
        canvas,
        landmark_center_x,
        landmark_center_y,
    )

    # ========================================================
    # VERY SMALL PHYSICAL BLUR
    # ========================================================

    sigma = (
        REFERENCE_BEAM_SPOT_NM
        / 11.0
    )

    canvas = cv2.GaussianBlur(
        canvas,
        (
            0,
            0,
        ),
        sigmaX=sigma,
        sigmaY=sigma,
    )

    return canvas


# ============================================================
# POISSON / DOSE NOISE
# ============================================================

def add_poisson_noise(
    image,
    rng,
    dose,
):
    """
    Dose-dependent shot noise.
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

    counts = (
        normalized
        * dose
    )

    counts = np.maximum(
        counts,
        0.01,
    )

    noisy = rng.poisson(
        counts
    ).astype(
        np.float32
    )

    noisy /= dose

    noisy *= 255.0

    return noisy


# ============================================================
# FIXED-PATTERN DETECTOR RESPONSE
# ============================================================

def add_fixed_pattern(
    image,
    rng,
    strength,
):
    """
    Detector pixel-response nonuniformity.

    This is different from ordinary Gaussian/speckle noise:
    the pattern is spatially correlated and detector-specific.
    """

    h, w = image.shape

    field = rng.normal(
        0,
        1,
        (
            h,
            w,
        ),
    ).astype(
        np.float32
    )

    field = cv2.GaussianBlur(
        field,
        (
            0,
            0,
        ),
        sigmaX=2.2,
        sigmaY=2.2,
    )

    std = field.std()

    if std > 1e-6:

        field /= std

    gain = (
        1.0
        + (
            strength
            / 100.0
        )
        * field
    )

    return (
        image
        * gain
    )


# ============================================================
# VERTICAL DETECTOR BANDING
# ============================================================

def add_vertical_banding(
    image,
    rng,
    strength,
):
    """
    Creates slowly varying vertical detector bands.
    """

    h, w = image.shape

    columns = rng.normal(
        0,
        1,
        w,
    ).astype(
        np.float32
    )

    columns = cv2.GaussianBlur(
        columns.reshape(
            1,
            -1,
        ),
        (
            0,
            0,
        ),
        sigmaX=10,
        sigmaY=1,
    ).reshape(
        -1
    )

    std = columns.std()

    if std > 1e-6:

        columns /= std

    return (
        image
        + columns[None, :]
        * strength
    )


# ============================================================
# PERIODIC SCAN MODULATION
# ============================================================

def add_scan_modulation(
    image,
    rng,
    strength,
):
    """
    Adds weak periodic acquisition modulation.

    It is intentionally subtle and not aligned to the
    target feature.
    """

    h, w = image.shape

    yy = np.arange(
        h,
        dtype=np.float32,
    )

    period = rng.uniform(
        17,
        31,
    )

    phase = rng.uniform(
        0,
        2 * np.pi,
    )

    modulation = (
        np.sin(
            2
            * np.pi
            * yy
            / period
            + phase
        )
    )

    modulation *= strength

    return (
        image
        + modulation[:, None]
    )


# ============================================================
# LOW FREQUENCY ILLUMINATION
# ============================================================

def add_illumination_field(
    image,
    rng,
    strength,
):
    """
    Broad illumination gradient.
    """

    h, w = image.shape

    low = rng.normal(
        0,
        1,
        (
            12,
            12,
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
        + field
        * strength
    )


# ============================================================
# CORRELATED GRAIN
# ============================================================

def add_correlated_grain(
    image,
    rng,
    strength,
):
    """
    Fine detector grain.
    """

    grain = rng.normal(
        0,
        1,
        image.shape,
    ).astype(
        np.float32
    )

    grain = cv2.GaussianBlur(
        grain,
        (
            0,
            0,
        ),
        sigmaX=0.45,
        sigmaY=0.45,
    )

    std = grain.std()

    if std > 1e-6:

        grain /= std

    return (
        image
        * (
            1.0
            + strength
            * grain
        )
    )


# ============================================================
# CHARGING GRADIENT
# ============================================================

def add_charging_gradient(
    image,
    rng,
    strength,
):
    """
    Broad directional charging variation.

    This is deliberately different from horizontal
    charging streaks used in previous pairs.
    """

    h, w = image.shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    angle = rng.uniform(
        -0.6,
        0.6,
    )

    direction = (
        np.cos(angle)
        * xx
        + np.sin(angle)
        * yy
    )

    direction -= direction.min()

    maximum = direction.max()

    if maximum > 1e-6:

        direction /= maximum

    direction -= 0.5

    return (
        image
        + direction
        * strength
    )


# ============================================================
# HOT / DEAD PIXELS
# ============================================================

def add_sparse_detector_events(
    image,
    rng,
    hot_probability,
    dead_probability,
):
    """
    Sparse detector defects.

    Very low probability so they do not dominate localization.
    """

    output = image.copy()

    random = rng.random(
        image.shape
    )

    hot = (
        random
        < hot_probability
    )

    dead = (
        (
            random
            >= hot_probability
        )
        &
        (
            random
            < (
                hot_probability
                + dead_probability
            )
        )
    )

    output[hot] = np.minimum(
        255,
        output[hot].astype(
            np.int16
        )
        + 45,
    ).astype(
        np.uint8
    )

    output[dead] = (
        output[dead]
        * 0.35
    ).astype(
        np.uint8
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
    Mild global shear/translation.
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
# ROW JITTER
# ============================================================

def apply_row_jitter(
    image,
    rng,
    jitter_px,
):
    """
    Very small row-wise spatial displacement.
    """

    if jitter_px <= 0:

        return image

    h, w = image.shape

    output = np.empty_like(
        image
    )

    base = np.arange(
        w,
        dtype=np.float32,
    )

    for y in range(h):

        shift = rng.normal(
            0,
            jitter_px,
        )

        coords = (
            base
            - shift
        )

        output[y] = np.interp(
            coords,
            base,
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
# ASTIGMATIC BLUR
# ============================================================

def apply_astigmatism(
    image,
    ratio,
):
    """
    Slight directional beam blur.
    """

    sigma_x = 0.55

    sigma_y = (
        sigma_x
        * ratio
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
# VIGNETTE
# ============================================================

def apply_vignette(
    image,
    strength,
):
    """
    Mild field-of-view attenuation.
    """

    h, w = image.shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

    nx = (
        xx - w / 2
    ) / (
        w / 2
    )

    ny = (
        yy - h / 2
    ) / (
        h / 2
    )

    r2 = (
        nx * nx
        + ny * ny
    )

    factor = (
        1
        - strength
        * np.clip(
            r2 / 2,
            0,
            1,
        )
    )

    return (
        image
        * factor
    )


# ============================================================
# GAMMA
# ============================================================

def apply_gamma(
    image,
    gamma,
):
    """
    Mild SEM contrast curve.
    """

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
        ** (
            1.0
            / gamma
        )
    )

    return (
        corrected
        * 255.0
    )


# ============================================================
# COMPLETE SEM ACQUISITION
# ============================================================

def simulate_reference(
    image,
    rng,
):
    """
    Clean high-dose reference acquisition.
    """

    image_f = (
        image.astype(
            np.float32
        )
    )

    image_f = cv2.GaussianBlur(
        image_f,
        (
            0,
            0,
        ),
        sigmaX=0.45,
        sigmaY=0.45,
    )

    image_f = add_poisson_noise(
        image_f,
        rng,
        REFERENCE_DOSE,
    )

    image_f += rng.normal(
        0,
        REFERENCE_READOUT_SIGMA,
        image_f.shape,
    )

    image_f = add_fixed_pattern(
        image_f,
        rng,
        REFERENCE_FIXED_PATTERN,
    )

    image_f = add_vertical_banding(
        image_f,
        rng,
        REFERENCE_VERTICAL_BANDING,
    )

    image_f = add_illumination_field(
        image_f,
        rng,
        REFERENCE_ILLUMINATION,
    )

    image_f = add_correlated_grain(
        image_f,
        rng,
        REFERENCE_GRAIN,
    )

    return np.clip(
        image_f,
        0,
        255,
    ).astype(
        np.uint8
    )


def simulate_search(
    image,
    rng,
):
    """
    DRAM-10 degraded search acquisition.
    """

    image_f = (
        image.astype(
            np.float32
        )
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
        sigmaX=0.72,
        sigmaY=0.82,
    )

    # --------------------------------------------------------
    # Slight anisotropic response
    # --------------------------------------------------------

    image_f = apply_astigmatism(
        image_f,
        SEARCH_ASTIGMATISM,
    )

    # --------------------------------------------------------
    # Raster movement
    # --------------------------------------------------------

    image_f = apply_raster_drift(
        image_f,
        rng,
        SEARCH_RASTER_DRIFT_PX,
    )

    image_f = apply_row_jitter(
        image_f,
        rng,
        SEARCH_ROW_JITTER_PX,
    )

    # --------------------------------------------------------
    # Lower dose
    # --------------------------------------------------------

    image_f = add_poisson_noise(
        image_f,
        rng,
        SEARCH_DOSE,
    )

    # --------------------------------------------------------
    # NEW DRAM-10 detector effects
    # --------------------------------------------------------

    image_f += rng.normal(
        0,
        SEARCH_READOUT_SIGMA,
        image_f.shape,
    )

    image_f = add_fixed_pattern(
        image_f,
        rng,
        SEARCH_FIXED_PATTERN,
    )

    image_f = add_vertical_banding(
        image_f,
        rng,
        SEARCH_VERTICAL_BANDING,
    )

    image_f = add_scan_modulation(
        image_f,
        rng,
        SEARCH_SCAN_MODULATION,
    )

    image_f = add_illumination_field(
        image_f,
        rng,
        SEARCH_ILLUMINATION,
    )

    image_f = add_correlated_grain(
        image_f,
        rng,
        SEARCH_GRAIN,
    )

    # --------------------------------------------------------
    # Directional charging
    # --------------------------------------------------------

    image_f = add_charging_gradient(
        image_f,
        rng,
        SEARCH_CHARGING,
    )

    # --------------------------------------------------------
    # Vignette
    # --------------------------------------------------------

    image_f = apply_vignette(
        image_f,
        SEARCH_VIGNETTE,
    )

    # --------------------------------------------------------
    # Contrast curve
    # --------------------------------------------------------

    image_f = apply_gamma(
        image_f,
        SEARCH_GAMMA,
    )

    # --------------------------------------------------------
    # Sparse detector defects
    # --------------------------------------------------------

    image_f = add_sparse_detector_events(
        image_f,
        rng,
        SEARCH_HOT_PIXEL_PROBABILITY,
        SEARCH_DEAD_PIXEL_PROBABILITY,
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
    print("=" * 80)
    print("DRIFT-SENSE — DRAM_10 GENERATOR")
    print("=" * 80)

    print()
    print("Structure:")
    print(
        "Elongated recessed trench / rail DRAM"
    )

    print()
    print("Unique feature:")
    print(
        "ONE small asymmetric T-shaped interruption"
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

    # ========================================================
    # PHYSICAL SCENE
    # ========================================================

    print()
    print(
        "[1/6] Generating physical DRAM scene..."
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
    # REFERENCE
    # ========================================================

    print(
        "[2/6] Extracting reference..."
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
            "Reference dimensions are incorrect."
        )

    # ========================================================
    # SEARCH
    # ========================================================

    print(
        "[3/6] Creating wide search..."
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
    # REFERENCE NOISE
    # ========================================================

    print(
        "[4/6] Applying clean reference SEM..."
    )

    reference_rng = np.random.default_rng(
        SEED + 101
    )

    reference = simulate_reference(
        reference,
        reference_rng,
    )

    # ========================================================
    # SEARCH NOISE
    # ========================================================

    print(
        "[5/6] Applying DRAM-10 search noise..."
    )

    search_rng = np.random.default_rng(
        SEED + 202
    )

    search = simulate_search(
        search,
        search_rng,
    )

    # ========================================================
    # GROUND TRUTH
    # ========================================================

    print(
        "[6/6] Saving images and manifest..."
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

    landmark_width_px = (
        LANDMARK_WIDTH_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    landmark_height_px = (
        LANDMARK_HEIGHT_NM
        / SEARCH_PIXEL_SIZE_NM
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

    manifest_path = (
        OUTPUT_DIR
        / "ground_truth.json"
    )

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
            "dram_10",

        "architecture":
            "DRAM",

        "seed":
            SEED,

        "structure_family":
            "elongated_recessed_trench_array",

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
                "small_asymmetric_T_trench",

            "landmark_size_nm":
                [
                    LANDMARK_WIDTH_NM,
                    LANDMARK_HEIGHT_NM,
                ],

            "landmark_size_search_px":
                [
                    landmark_width_px,
                    landmark_height_px,
                ],

            "landmark_count":
                1,
        },

        "structure": {

            "array_block_size_nm":
                ARRAY_BLOCK_SIZE_NM,

            "separator_width_nm":
                SEPARATOR_WIDTH_NM,

            "trench_pitch_x_nm":
                TRENCH_PITCH_X_NM,

            "trench_pitch_y_nm":
                TRENCH_PITCH_Y_NM,

            "trench_width_nm":
                TRENCH_WIDTH_NM,

            "trench_height_nm":
                TRENCH_HEIGHT_NM,

            "rail_width_nm":
                RAIL_WIDTH_NM,

            "bridge_width_nm":
                BRIDGE_WIDTH_NM,

            "row_stagger_nm":
                ROW_STAGGER_NM,
        },

        "sem_physics": {

            "reference_beam_spot_nm":
                REFERENCE_BEAM_SPOT_NM,

            "search_beam_spot_nm":
                SEARCH_BEAM_SPOT_NM,

            "pattern_collapse_threshold_nm":
                PATTERN_COLLAPSE_THRESHOLD_NM,

            "reference_dose":
                REFERENCE_DOSE,

            "search_dose":
                SEARCH_DOSE,

            "search_raster_drift_px":
                SEARCH_RASTER_DRIFT_PX,

            "search_row_jitter_px":
                SEARCH_ROW_JITTER_PX,

            "search_astigmatism":
                SEARCH_ASTIGMATISM,

            "search_gamma":
                SEARCH_GAMMA,

            "search_vignette":
                SEARCH_VIGNETTE,
        },

        "noise_model": {

            "type":
                "DRAM-10 detector-pattern noise",

            "reference_readout_sigma":
                REFERENCE_READOUT_SIGMA,

            "search_readout_sigma":
                SEARCH_READOUT_SIGMA,

            "reference_fixed_pattern":
                REFERENCE_FIXED_PATTERN,

            "search_fixed_pattern":
                SEARCH_FIXED_PATTERN,

            "reference_vertical_banding":
                REFERENCE_VERTICAL_BANDING,

            "search_vertical_banding":
                SEARCH_VERTICAL_BANDING,

            "reference_scan_modulation":
                REFERENCE_SCAN_MODULATION,

            "search_scan_modulation":
                SEARCH_SCAN_MODULATION,

            "reference_grain":
                REFERENCE_GRAIN,

            "search_grain":
                SEARCH_GRAIN,

            "reference_illumination":
                REFERENCE_ILLUMINATION,

            "search_illumination":
                SEARCH_ILLUMINATION,

            "search_charging":
                SEARCH_CHARGING,

            "search_hot_pixel_probability":
                SEARCH_HOT_PIXEL_PROBABILITY,

            "search_dead_pixel_probability":
                SEARCH_DEAD_PIXEL_PROBABILITY,
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
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            manifest,
            f,
            indent=4,
        )

    # ========================================================
    # REPORT
    # ========================================================

    print()
    print("=" * 80)
    print("DRAM_10 GENERATED SUCCESSFULLY")
    print("=" * 80)

    print()
    print(
        f"Reference : {reference_path}"
    )

    print(
        f"Search    : {search_path}"
    )

    print()
    print("Reference crop origin:")
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
    print("Unique feature:")
    print(
        f"  {LANDMARK_WIDTH_NM} x "
        f"{LANDMARK_HEIGHT_NM} nm"
    )

    print(
        f"  {landmark_width_px:.1f} x "
        f"{landmark_height_px:.1f} search px"
    )

    print()
    print("NEW DRAM-10 noise:")
    print(
        "  detector fixed-pattern variation"
    )

    print(
        "  vertical detector banding"
    )

    print(
        "  periodic scan modulation"
    )

    print(
        "  low-frequency illumination"
    )

    print(
        "  multiplicative correlated grain"
    )

    print(
        "  directional charging gradient"
    )

    print(
        "  sparse detector hot/dead pixels"
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
    print("=" * 80)


if __name__ == "__main__":
    main()