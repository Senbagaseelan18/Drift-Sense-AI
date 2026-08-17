#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-09 Synthetic Pair Generator
=================================

DRAM-09 uses a different structural family from DRAM-08.

STRUCTURE
---------

The scene contains:

    * staggered rectangular storage/contact pillars
    * horizontal bus-like structures
    * narrow vertical traces
    * alternating row offsets
    * repeated DRAM macro blocks
    * dark separator streets
    * subtle block boundary structures

UNIQUE FEATURE
--------------

Exactly ONE small asymmetric double-notch / bridge feature
is embedded into one normal DRAM cell.

It is intentionally small.

Landmark physical size:

    170 nm x 160 nm

At search scale:

    17 x 16 pixels

The feature is surrounded by normal DRAM structures so
that localization depends on the actual structural pattern,
rather than a giant artificial black rectangle.

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

SEM DEGRADATION
---------------

Reference:
    high dose
    low detector noise
    very small blur
    no raster drift
    no row jitter

Search:
    lower dose
    slightly stronger blur
    raster drift
    row jitter
    correlated detector grain
    scan-line variation
    mild charging
    mild vignette
    very small impulse noise

The search is degraded, but the structural information
remains usable for localization.
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

SEED = 20260818


# ============================================================
# TARGET LOCATION
# ============================================================

# 1 um x 1 um reference crop origin.

TARGET_X_NM = 6040

TARGET_Y_NM = 3640


# ============================================================
# MACRO ARRAY
# ============================================================

ARRAY_BLOCK_SIZE_NM = 2350

SEPARATOR_WIDTH_NM = 300

ARRAY_MARGIN_NM = 120


# ============================================================
# DRAM CELL GEOMETRY
# ============================================================

CELL_PITCH_X_NM = 135

CELL_PITCH_Y_NM = 145

ROW_OFFSET_NM = 38

PILLAR_WIDTH_NM = 52

PILLAR_HEIGHT_NM = 68

HORIZONTAL_TRACE_WIDTH_NM = 10

VERTICAL_TRACE_WIDTH_NM = 8

TRACE_LENGTH_NM = 112


# ============================================================
# UNIQUE FEATURE
# ============================================================

LANDMARK_WIDTH_NM = 170

LANDMARK_HEIGHT_NM = 160

LANDMARK_EDGE_NM = 7


# ============================================================
# SEM / IMAGING
# ============================================================

REFERENCE_BEAM_SPOT_NM = 4.5

SEARCH_BEAM_SPOT_NM = 6.5

PATTERN_COLLAPSE_THRESHOLD_NM = 10.0


# ============================================================
# ACQUISITION
# ============================================================

REFERENCE_DOSE = 2000.0

SEARCH_DOSE = 200.0

SEARCH_RASTER_DRIFT_PX = 1.50

SEARCH_ROW_JITTER_PX = 0.50


# ============================================================
# DISTORTION
# ============================================================

REFERENCE_CD_BIAS_NM = 0.0

SEARCH_CD_BIAS_NM = 0.8

REFERENCE_CORNER_ROUNDING_PX = 0.35

SEARCH_CORNER_ROUNDING_PX = 1.0

REFERENCE_ASTIGMATISM = 1.00

SEARCH_ASTIGMATISM = 1.04

REFERENCE_VIGNETTE = 0.01

SEARCH_VIGNETTE = 0.07

REFERENCE_GAMMA = 1.00

SEARCH_GAMMA = 1.04


# ============================================================
# NOISE
# ============================================================

REFERENCE_SPECKLE_SIGMA = 0.012

SEARCH_SPECKLE_SIGMA = 0.055

REFERENCE_SALT_PEPPER = 0.0

SEARCH_SALT_PEPPER = 0.0012

REFERENCE_SCAN_NOISE = 0.20

SEARCH_SCAN_NOISE = 0.85

REFERENCE_CHARGING = 0.0

SEARCH_CHARGING = 0.65


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_09"
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
# DRAM PILLAR
# ============================================================

def draw_pillar(
    image,
    cx,
    cy,
    rng,
):
    """
    Draw one rectangular DRAM storage/contact pillar.

    The rows are intentionally slightly asymmetric and
    staggered to avoid a perfectly computer-generated grid.
    """

    cx += rng.normal(
        0,
        1.2,
    )

    cy += rng.normal(
        0,
        1.2,
    )

    width = (
        PILLAR_WIDTH_NM
        + rng.normal(
            0,
            1.8,
        )
    )

    height = (
        PILLAR_HEIGHT_NM
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
    # Main pillar
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
                    181,
                    5,
                ),
                0,
                255,
            )
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Slightly darker inner body
    # --------------------------------------------------------

    inner = 6

    if (
        x1 - x0 > 2 * inner
        and y1 - y0 > 2 * inner
    ):

        cv2.rectangle(
            image,
            (
                x0 + inner,
                y0 + inner,
            ),
            (
                x1 - inner,
                y1 - inner,
            ),
            int(
                np.clip(
                    rng.normal(
                        170,
                        3,
                    ),
                    0,
                    255,
                )
            ),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    # --------------------------------------------------------
    # Small bright top response
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            x0 + 7,
            y0 + 5,
        ),
        (
            x1 - 7,
            y0 + 12,
        ),
        196,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# NORMAL DRAM CELL
# ============================================================

def draw_cell(
    image,
    cx,
    cy,
    rng,
):
    """
    Draw one normal DRAM cell:

       ─────────
           │
         [██]
           │
       ─────────

    The rectangular pillar is the dominant local feature.
    """

    # --------------------------------------------------------
    # Vertical trace
    # --------------------------------------------------------

    draw_line(
        image,
        cx,
        cy - TRACE_LENGTH_NM / 2,
        cx,
        cy + TRACE_LENGTH_NM / 2,
        rng.normal(
            67,
            2,
        ),
        VERTICAL_TRACE_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Horizontal trace
    # --------------------------------------------------------

    draw_line(
        image,
        cx - TRACE_LENGTH_NM / 2,
        cy,
        cx + TRACE_LENGTH_NM / 2,
        cy,
        rng.normal(
            69,
            2,
        ),
        HORIZONTAL_TRACE_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Central pillar
    # --------------------------------------------------------

    draw_pillar(
        image,
        cx,
        cy,
        rng,
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
    Draw one staggered DRAM macro block.
    """

    # --------------------------------------------------------
    # Block background
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
        40,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Staggered rows
    # --------------------------------------------------------

    row = 0

    y = (
        y0
        + CELL_PITCH_Y_NM / 2
    )

    while y < (
        y0
        + height
        - CELL_PITCH_Y_NM / 2
    ):

        offset = (
            ROW_OFFSET_NM
            if row % 2
            else 0
        )

        x = (
            x0
            + CELL_PITCH_X_NM / 2
            + offset
        )

        while x < (
            x0
            + width
        ):

            draw_cell(
                image,
                x,
                y,
                rng,
            )

            x += CELL_PITCH_X_NM

        y += CELL_PITCH_Y_NM

        row += 1


# ============================================================
# UNIQUE DOUBLE-NOTCH / BRIDGE FEATURE
# ============================================================

def draw_unique_feature(
    image,
    center_x,
    center_y,
):
    """
    Create exactly ONE small asymmetric structure.

    Normal surrounding cell:
        ─── [██] ───

    Unique cell:

        ─── [██] ──┐
                  │
             [██]─┘

    It contains:

        * split pillar
        * narrow bridge
        * asymmetric notch

    This creates a structural cue without using a giant
    artificial box.
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
    # Clear the local normal cell
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
        41,
        thickness=-1,
    )

    # ========================================================
    # Upper pillar
    # ========================================================

    cv2.rectangle(
        image,
        (
            int(center_x - 27),
            int(y0 + 19),
        ),
        (
            int(center_x + 24),
            int(center_y - 8),
        ),
        185,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # ========================================================
    # Lower offset pillar
    # ========================================================

    cv2.rectangle(
        image,
        (
            int(center_x + 4),
            int(center_y + 15),
        ),
        (
            int(center_x + 36),
            int(y1 - 18),
        ),
        179,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # ========================================================
    # Horizontal bridge
    # ========================================================

    draw_line(
        image,
        center_x - 21,
        center_y - 3,
        center_x + 44,
        center_y - 3,
        165,
        15,
    )

    # ========================================================
    # Small diagonal connection
    # ========================================================

    draw_line(
        image,
        center_x + 20,
        center_y + 1,
        center_x + 40,
        center_y + 22,
        150,
        9,
    )

    # ========================================================
    # Two small dark notches
    # ========================================================

    cv2.rectangle(
        image,
        (
            int(center_x - 13),
            int(center_y - 27),
        ),
        (
            int(center_x + 10),
            int(center_y - 9),
        ),
        27,
        thickness=-1,
    )

    cv2.rectangle(
        image,
        (
            int(center_x + 8),
            int(center_y + 17),
        ),
        (
            int(center_x + 24),
            int(center_y + 33),
        ),
        29,
        thickness=-1,
    )

    # ========================================================
    # Very subtle local perimeter
    # ========================================================

    draw_line(
        image,
        x0,
        y0,
        x1,
        y0,
        55,
        LANDMARK_EDGE_NM,
    )

    draw_line(
        image,
        x0,
        y1,
        x1,
        y1,
        53,
        LANDMARK_EDGE_NM,
    )

    draw_line(
        image,
        x0,
        y0,
        x0,
        y1,
        54,
        LANDMARK_EDGE_NM,
    )

    draw_line(
        image,
        x1,
        y0,
        x1,
        y1,
        52,
        LANDMARK_EDGE_NM,
    )


# ============================================================
# COMPLETE PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    """
    Generate the complete 10 um x 10 um DRAM scene.
    """

    rng = np.random.default_rng(
        SEED
    )

    canvas = np.full(
        (
            PHYSICAL_SIZE_NM,
            PHYSICAL_SIZE_NM,
        ),
        30,
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
    # DARK VERTICAL SEPARATORS
    # ========================================================

    for x in [
        2470,
        5120,
        7770,
    ]:

        draw_line(
            canvas,
            x,
            0,
            x,
            PHYSICAL_SIZE_NM,
            33,
            SEPARATOR_WIDTH_NM,
        )

    # ========================================================
    # DARK HORIZONTAL SEPARATORS
    # ========================================================

    for y in [
        2470,
        5120,
        7770,
    ]:

        draw_line(
            canvas,
            0,
            y,
            PHYSICAL_SIZE_NM,
            y,
            33,
            SEPARATOR_WIDTH_NM,
        )

    # ========================================================
    # BLOCK EDGE
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
                54,
                thickness=6,
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

    draw_unique_feature(
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
# POISSON SHOT NOISE
# ============================================================

def add_poisson_noise(
    image,
    rng,
    dose,
):
    """
    Lower dose -> stronger shot noise.
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
# CORRELATED GRAIN
# ============================================================

def add_correlated_grain(
    image,
    rng,
    sigma,
):
    """
    SEM-like correlated grain.

    Unlike pure pixel-independent Gaussian noise, this
    generates a spatially correlated texture.
    """

    h, w = image.shape

    grain = rng.normal(
        0,
        1,
        (
            h,
            w,
        ),
    ).astype(
        np.float32
    )

    grain = cv2.GaussianBlur(
        grain,
        (
            0,
            0,
        ),
        sigmaX=0.65,
        sigmaY=0.65,
    )

    std = grain.std()

    if std > 1e-6:

        grain /= std

    # Multiplicative component.

    output = (
        image
        * (
            1.0
            + sigma * grain
        )
    )

    return output


# ============================================================
# LOW FREQUENCY BACKGROUND
# ============================================================

def add_low_frequency_variation(
    image,
    rng,
    amplitude,
):
    """
    Broad background intensity variation.
    """

    h, w = image.shape

    low = rng.normal(
        0,
        1,
        (
            18,
            18,
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
# SCAN-LINE NOISE
# ============================================================

def add_scanline_noise(
    image,
    rng,
    amplitude,
):
    """
    Correlated horizontal scan-line intensity variation.
    """

    h, w = image.shape

    row_noise = rng.normal(
        0,
        amplitude,
        h,
    ).astype(
        np.float32
    )

    # Correlate neighboring rows.

    row_noise = cv2.GaussianBlur(
        row_noise.reshape(
            -1,
            1,
        ),
        (
            1,
            0,
        ),
        sigmaX=0,
        sigmaY=1.1,
    ).reshape(
        -1
    )

    return (
        image
        + row_noise[:, None]
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
    Slight row-to-row spatial displacement.
    """

    if jitter_px <= 0:

        return image

    h, w = image.shape

    output = np.empty_like(
        image
    )

    xbase = np.arange(
        w,
        dtype=np.float32,
    )

    for y in range(h):

        shift = rng.normal(
            0,
            jitter_px,
        )

        coords = (
            xbase
            - shift
        )

        output[y] = np.interp(
            coords,
            xbase,
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
    Small global acquisition drift/shear.
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
        -0.0007,
        0.0007,
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
    Mild anisotropic beam response.
    """

    if abs(
        ratio - 1.0
    ) < 1e-6:

        return image

    sigma_x = 0.65

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
    Mild field-of-view brightness falloff.
    """

    if strength <= 0:

        return image

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
# CHARGING
# ============================================================

def add_charging(
    image,
    rng,
    intensity,
):
    """
    Very mild broad charging bands.

    Deliberately weak so that charging itself does not
    become the localization feature.
    """

    if intensity <= 0:

        return image

    output = image.astype(
        np.float32
    ).copy()

    h, w = image.shape

    # One or two weak broad bands.

    number = rng.integers(
        1,
        3,
    )

    for _ in range(number):

        y = rng.integers(
            50,
            max(
                51,
                h - 50,
            ),
        )

        sigma = rng.uniform(
            4,
            10,
        )

        yy = np.arange(
            h,
            dtype=np.float32,
        )

        band = np.exp(
            -(
                (
                    yy - y
                ) ** 2
                / (
                    2
                    * sigma
                    * sigma
                )
            )
        )

        amount = rng.uniform(
            -intensity,
            intensity,
        )

        output += (
            band[:, None]
            * amount
        )

    return output


# ============================================================
# GAMMA
# ============================================================

def apply_gamma(
    image,
    gamma,
):
    """
    Mild contrast curve.
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
            1.0 / gamma
        )
    )

    return (
        corrected
        * 255.0
    )


# ============================================================
# SALT / PEPPER
# ============================================================

def add_salt_pepper(
    image,
    rng,
    probability,
):
    """
    Very sparse impulse noise.
    """

    if probability <= 0:

        return image

    output = image.copy()

    random = rng.random(
        image.shape
    )

    pepper = (
        random
        < probability / 2
    )

    salt = (
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

    output[pepper] = 0

    output[salt] = 255

    return output


# ============================================================
# COMPLETE SEM ACQUISITION
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
    astigmatism,
    vignette,
    gamma,
    speckle_sigma,
    scan_noise,
    charging,
    salt_pepper,
    background_amplitude,
):
    """
    Full acquisition model.
    """

    image_f = (
        image.astype(
            np.float32
        )
    )

    # --------------------------------------------------------
    # Beam spot
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
    # Shot noise
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
    # Correlated grain
    # --------------------------------------------------------

    image_f = add_correlated_grain(
        image_f,
        rng,
        speckle_sigma,
    )

    # --------------------------------------------------------
    # Low frequency variation
    # --------------------------------------------------------

    image_f = add_low_frequency_variation(
        image_f,
        rng,
        background_amplitude,
    )

    # --------------------------------------------------------
    # Scan-line noise
    # --------------------------------------------------------

    image_f = add_scanline_noise(
        image_f,
        rng,
        scan_noise,
    )

    # --------------------------------------------------------
    # Charging
    # --------------------------------------------------------

    image_f = add_charging(
        image_f,
        rng,
        charging,
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
    # Sparse impulse noise
    # --------------------------------------------------------

    image_f = add_salt_pepper(
        image_f,
        rng,
        salt_pepper,
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
    print("DRIFT-SENSE — DRAM_09 GENERATOR")
    print("=" * 78)

    print()
    print("Structure:")
    print(
        "Staggered rectangular pillar / trace DRAM"
    )

    print()
    print("Unique feature:")
    print(
        "ONE small asymmetric double-notch bridge"
    )

    print()
    print("Landmark:")
    print(
        f"{LANDMARK_WIDTH_NM} x "
        f"{LANDMARK_HEIGHT_NM} nm"
    )

    print()
    print("Search landmark:")
    print(
        f"{LANDMARK_WIDTH_NM / SEARCH_PIXEL_SIZE_NM:.1f} x "
        f"{LANDMARK_HEIGHT_NM / SEARCH_PIXEL_SIZE_NM:.1f} px"
    )

    # ========================================================
    # 1
    # ========================================================

    print()
    print(
        "[1/7] Generating physical DRAM scene..."
    )

    physical_scene = (
        generate_physical_scene()
    )

    # ========================================================
    # 2
    # ========================================================

    print(
        "[2/7] Extracting reference..."
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
            "Reference is not 1000x1000."
        )

    # ========================================================
    # 3
    # ========================================================

    print(
        "[3/7] Creating wide search..."
    )

    search = cv2.resize(
        physical_scene,
        (
            1000,
            1000,
        ),
        interpolation=cv2.INTER_AREA,
    )

    if search.shape != (
        1000,
        1000,
    ):

        raise RuntimeError(
            "Search is not 1000x1000."
        )

    # ========================================================
    # 4
    # ========================================================

    print(
        "[4/7] Applying reference SEM..."
    )

    reference_rng = np.random.default_rng(
        SEED + 101
    )

    reference = simulate_sem(
        reference,
        reference_rng,
        beam_spot_nm=REFERENCE_BEAM_SPOT_NM,
        dose=REFERENCE_DOSE,
        detector_sigma=0.55,
        drift_px=0.0,
        row_jitter_px=0.0,
        astigmatism=REFERENCE_ASTIGMATISM,
        vignette=REFERENCE_VIGNETTE,
        gamma=REFERENCE_GAMMA,
        speckle_sigma=REFERENCE_SPECKLE_SIGMA,
        scan_noise=REFERENCE_SCAN_NOISE,
        charging=REFERENCE_CHARGING,
        salt_pepper=REFERENCE_SALT_PEPPER,
        background_amplitude=0.8,
    )

    # ========================================================
    # 5
    # ========================================================

    print(
        "[5/7] Applying noisy search SEM..."
    )

    search_rng = np.random.default_rng(
        SEED + 202
    )

    search = simulate_sem(
        search,
        search_rng,
        beam_spot_nm=SEARCH_BEAM_SPOT_NM,
        dose=SEARCH_DOSE,
        detector_sigma=1.40,
        drift_px=SEARCH_RASTER_DRIFT_PX,
        row_jitter_px=SEARCH_ROW_JITTER_PX,
        astigmatism=SEARCH_ASTIGMATISM,
        vignette=SEARCH_VIGNETTE,
        gamma=SEARCH_GAMMA,
        speckle_sigma=SEARCH_SPECKLE_SIGMA,
        scan_noise=SEARCH_SCAN_NOISE,
        charging=SEARCH_CHARGING,
        salt_pepper=SEARCH_SALT_PEPPER,
        background_amplitude=2.4,
    )

    # ========================================================
    # 6
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

    landmark_width_px = (
        LANDMARK_WIDTH_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    landmark_height_px = (
        LANDMARK_HEIGHT_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    # ========================================================
    # 7
    # ========================================================

    print(
        "[7/7] Saving..."
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
            "Could not save reference."
        )

    if not cv2.imwrite(
        str(search_path),
        search,
    ):

        raise RuntimeError(
            "Could not save search."
        )

    # ========================================================
    # MANIFEST
    # ========================================================

    manifest = {

        "pair_id":
            "dram_09",

        "architecture":
            "DRAM",

        "seed":
            SEED,

        "structure_family":
            "staggered_rectangular_pillar_array",

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
                "small_asymmetric_double_notch_bridge",

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

            "cell_pitch_x_nm":
                CELL_PITCH_X_NM,

            "cell_pitch_y_nm":
                CELL_PITCH_Y_NM,

            "row_offset_nm":
                ROW_OFFSET_NM,

            "pillar_width_nm":
                PILLAR_WIDTH_NM,

            "pillar_height_nm":
                PILLAR_HEIGHT_NM,

            "horizontal_trace_width_nm":
                HORIZONTAL_TRACE_WIDTH_NM,

            "vertical_trace_width_nm":
                VERTICAL_TRACE_WIDTH_NM,
        },

        "sem_imaging": {

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

            "reference_speckle_sigma":
                REFERENCE_SPECKLE_SIGMA,

            "search_speckle_sigma":
                SEARCH_SPECKLE_SIGMA,

            "reference_scan_noise":
                REFERENCE_SCAN_NOISE,

            "search_scan_noise":
                SEARCH_SCAN_NOISE,

            "reference_charging":
                REFERENCE_CHARGING,

            "search_charging":
                SEARCH_CHARGING,

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

            "scale_ratio":
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
    # FINAL REPORT
    # ========================================================

    print()
    print("=" * 78)
    print("DRAM_09 GENERATED SUCCESSFULLY")
    print("=" * 78)

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
    print("Target center:")
    print(
        f"  ({center_x:.1f}, {center_y:.1f}) px"
    )

    print()
    print("Unique landmark:")
    print(
        f"  {LANDMARK_WIDTH_NM} x "
        f"{LANDMARK_HEIGHT_NM} nm"
    )

    print(
        f"  {landmark_width_px:.1f} x "
        f"{landmark_height_px:.1f} search px"
    )

    print()
    print("Search acquisition:")
    print(
        f"  dose          = {SEARCH_DOSE}"
    )

    print(
        f"  beam spot     = "
        f"{SEARCH_BEAM_SPOT_NM} nm"
    )

    print(
        f"  raster drift  = "
        f"{SEARCH_RASTER_DRIFT_PX} px"
    )

    print(
        f"  row jitter    = "
        f"{SEARCH_ROW_JITTER_PX} px"
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