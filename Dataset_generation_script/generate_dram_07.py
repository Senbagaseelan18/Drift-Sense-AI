#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-07 Synthetic Pair Generator
=================================

DRAM-07
-------

Dense circular-contact DRAM background with ONE small
dark recessed rectangular structure.

The dark structure is a real part of the synthetic
physical scene. It is NOT a bounding-box annotation.

The complete physical scene is generated first.

Physical scene:
    10000 x 10000 nm
    1 nm / pixel

Reference:
    1000 x 1000 px
    1 nm / pixel
    1 um x 1 um

Search:
    1000 x 1000 px
    10 nm / pixel
    10 um x 10 um

The reference is an exact crop from the same physical
scene used to create the search.

IMPORTANT
---------

The dark landmark is intentionally SMALL.

Landmark physical size:

    180 nm x 240 nm

Therefore at the 10 nm/pixel search scale:

    18 px x 24 px

This keeps the feature detectable but prevents it from
dominating the reference image.


NOISE
-----

Reference:
    cleaner high-resolution acquisition

Search:
    stronger degradation

Included:

    * Poisson shot noise
    * detector/readout noise
    * low-frequency variation
    * scan-line variation
    * mild charging-like variation
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

SEED = 20260816


# ============================================================
# TARGET LOCATION
# ============================================================

"""
The reference crop starts at this physical coordinate.

The target is deliberately placed inside a normal DRAM
memory block, away from the major inter-block streets.
"""

TARGET_X_NM = 5280

TARGET_Y_NM = 3380


# ============================================================
# DRAM CIRCULAR-CONTACT ARRAY
# ============================================================

CONTACT_PITCH_NM = 125

CONTACT_RADIUS_NM = 28

SUPPORT_LINE_WIDTH_NM = 9

SUPPORT_HALF_LENGTH_NM = 47


# ============================================================
# MEMORY BLOCK GEOMETRY
# ============================================================

BLOCK_SIZE_NM = 2150

STREET_WIDTH_NM = 260

ARRAY_MARGIN_NM = 100


# ============================================================
# SMALL DARK LANDMARK
# ============================================================

"""
IMPORTANT:

Previous version:
    300 x 440 nm

Updated version:
    180 x 240 nm

At search scale:

    18 x 24 pixels

This is intentionally much smaller.
"""

LANDMARK_WIDTH_NM = 180

LANDMARK_HEIGHT_NM = 240

LANDMARK_DARK_LEVEL = 24

LANDMARK_EDGE_LEVEL = 47

LANDMARK_EDGE_WIDTH_NM = 6


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_07"
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
    Draw an anti-aliased semiconductor structure.
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
    Draw one circular contact.

    A slightly brighter center provides a more natural
    SEM-like contact appearance.
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

    # --------------------------------------------------------
    # Outer contact
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Slight central brightness
    # --------------------------------------------------------

    inner_radius = max(
        1,
        int(
            round(
                radius * 0.42
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
                intensity + 7,
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

    Structure:

             |
             |
         ----O----
             |
             |

    The circular contact is the dominant feature.
    """

    # --------------------------------------------------------
    # Small placement variation
    # --------------------------------------------------------

    local_x = (
        cx
        + rng.normal(
            0,
            1.3,
        )
    )

    local_y = (
        cy
        + rng.normal(
            0,
            1.3,
        )
    )

    # --------------------------------------------------------
    # Horizontal support
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
            70,
            2,
        ),
        SUPPORT_LINE_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Vertical support
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
            68,
            2,
        ),
        SUPPORT_LINE_WIDTH_NM,
    )

    # --------------------------------------------------------
    # Circular contact
    # --------------------------------------------------------

    draw_contact(
        image,
        local_x,
        local_y,
        rng.normal(
            CONTACT_RADIUS_NM,
            1.0,
        ),
        rng.normal(
            192,
            4,
        ),
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
    Draw one dense circular-contact DRAM block.
    """

    # --------------------------------------------------------
    # Dark semiconductor background
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
        44,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Contact grid
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
# SMALL DARK RECESSED LANDMARK
# ============================================================

def draw_dark_landmark(
    image,
    center_x,
    center_y,
):
    """
    Draw ONE SMALL dark recessed rectangular structure.

    This version is deliberately much smaller than the
    previous DRAM-07.

    Approximate appearance:

          normal contacts

        O  O  O  O  O
        O  O [██] O  O
        O  O [██] O  O
        O  O [██] O  O
        O  O  O  O  O

    The structure is slightly irregular so that it does
    not look like a perfect computer-drawn rectangle.
    """

    half_w = (
        LANDMARK_WIDTH_NM
        / 2.0
    )

    half_h = (
        LANDMARK_HEIGHT_NM
        / 2.0
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

    # ========================================================
    # Slightly bright outer edge
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
        LANDMARK_EDGE_LEVEL,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # ========================================================
    # Dark inner recessed region
    # ========================================================

    inset_x = 13

    inset_y = 14

    ix0 = x0 + inset_x

    iy0 = y0 + inset_y

    ix1 = x1 - inset_x

    iy1 = y1 - inset_y

    # Slightly irregular polygon.

    polygon = np.array(
        [
            [
                ix0,
                iy0 + 2,
            ],
            [
                ix1 - 2,
                iy0,
            ],
            [
                ix1,
                iy1 - 3,
            ],
            [
                ix0 + 3,
                iy1,
            ],
        ],
        dtype=np.int32,
    )

    cv2.fillPoly(
        image,
        [
            polygon
        ],
        LANDMARK_DARK_LEVEL,
    )

    # ========================================================
    # Very subtle inner gradient
    # ========================================================

    roi_x0 = max(
        0,
        ix0,
    )

    roi_y0 = max(
        0,
        iy0,
    )

    roi_x1 = min(
        image.shape[1],
        ix1,
    )

    roi_y1 = min(
        image.shape[0],
        iy1,
    )

    if (
        roi_x1 > roi_x0
        and roi_y1 > roi_y0
    ):

        roi = image[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ]

        h, w = roi.shape

        yy = np.linspace(
            0,
            1,
            h,
            dtype=np.float32,
        )[:, None]

        xx = np.linspace(
            0,
            1,
            w,
            dtype=np.float32,
        )[None, :]

        gradient = (
            1.5 * xx
            + 1.0 * yy
        )

        roi_f = (
            roi.astype(
                np.float32
            )
            - gradient
        )

        image[
            roi_y0:roi_y1,
            roi_x0:roi_x1,
        ] = np.clip(
            roi_f,
            0,
            255,
        ).astype(
            np.uint8
        )

    # ========================================================
    # Subtle edge response
    # ========================================================

    edge_width = (
        LANDMARK_EDGE_WIDTH_NM
    )

    draw_line(
        image,
        x0,
        y0,
        x1,
        y0,
        58,
        edge_width,
    )

    draw_line(
        image,
        x0,
        y1,
        x1,
        y1,
        54,
        edge_width,
    )

    draw_line(
        image,
        x0,
        y0,
        x0,
        y1,
        56,
        edge_width,
    )

    draw_line(
        image,
        x1,
        y0,
        x1,
        y1,
        53,
        edge_width,
    )


# ============================================================
# COMPLETE PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    """
    Generate the complete 10,000 x 10,000 nm scene.

    Exactly ONE dark landmark is inserted.
    """

    rng = np.random.default_rng(
        SEED
    )

    # --------------------------------------------------------
    # Base substrate
    # --------------------------------------------------------

    canvas = np.full(
        (
            PHYSICAL_SIZE_NM,
            PHYSICAL_SIZE_NM,
        ),
        32,
        dtype=np.uint8,
    )

    # ========================================================
    # 4 x 4 DRAM BLOCK ARRAY
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
            34,
            STREET_WIDTH_NM,
        )

    # ========================================================
    # SUBTLE MEMORY BLOCK BORDERS
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
                56,
                thickness=7,
                lineType=cv2.LINE_AA,
            )

    # ========================================================
    # UNIQUE SMALL DARK LANDMARK
    # ========================================================

    landmark_center_x = (
        TARGET_X_NM
        + REFERENCE_SIZE_NM / 2.0
    )

    landmark_center_y = (
        TARGET_Y_NM
        + REFERENCE_SIZE_NM / 2.0
    )

    draw_dark_landmark(
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
# POISSON / SHOT NOISE
# ============================================================

def add_poisson_noise(
    image,
    rng,
    photon_level,
):
    """
    Approximate electron-count / shot noise.
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


# ============================================================
# DETECTOR NOISE
# ============================================================

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


# ============================================================
# LOW FREQUENCY VARIATION
# ============================================================

def add_low_frequency_variation(
    image,
    rng,
    amplitude,
):
    """
    Smooth illumination/background variation.
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
        + amplitude * field
    )


# ============================================================
# SCAN-LINE VARIATION
# ============================================================

def add_scanline_noise(
    image,
    rng,
    amplitude,
):
    """
    Mild horizontal scan-line variation.
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


# ============================================================
# CHARGING-LIKE VARIATION
# ============================================================

def add_charging_variation(
    image,
    rng,
    amplitude,
):
    """
    Broad, weak charging-like intensity variation.

    Kept deliberately small so the landmark does not become
    artificially highlighted.
    """

    h, w = image.shape

    yy, xx = np.mgrid[
        0:h,
        0:w,
    ]

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
            0.35,
            0.50,
        )
    )

    sy = (
        h
        * rng.uniform(
            0.35,
            0.50,
        )
    )

    field = np.exp(
        -(
            (xx - cx) ** 2
            / (2 * sx * sx)
            +
            (yy - cy) ** 2
            / (2 * sy * sy)
        )
    )

    field -= field.mean()

    return (
        image
        + amplitude * field
    )


# ============================================================
# SEM ACQUISITION MODEL
# ============================================================

def simulate_sem(
    image,
    rng,
    blur_sigma,
    photon_level,
    detector_sigma,
    background_amplitude,
    scan_amplitude,
    charging_amplitude,
):
    """
    Apply SEM-like image formation/degradation.
    """

    image_f = image.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Beam / imaging blur
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
        sigmaX=1.15,
        sigmaY=1.15,
    )

    image_f += (
        0.25
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
        background_amplitude,
    )

    # --------------------------------------------------------
    # Scan-line variation
    # --------------------------------------------------------

    image_f = add_scanline_noise(
        image_f,
        rng,
        scan_amplitude,
    )

    # --------------------------------------------------------
    # Charging variation
    # --------------------------------------------------------

    image_f = add_charging_variation(
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
    print("=" * 76)
    print("DRIFT-SENSE — DRAM_07 GENERATOR")
    print("=" * 76)

    print()
    print("Background:")
    print(
        "Dense circular-contact DRAM array"
    )

    print()
    print("Unique structure:")
    print(
        "SMALL dark recessed rectangle"
    )

    print()
    print("Landmark physical size:")
    print(
        f"{LANDMARK_WIDTH_NM} x "
        f"{LANDMARK_HEIGHT_NM} nm"
    )

    print()
    print("Landmark at search scale:")
    print(
        f"{LANDMARK_WIDTH_NM / SEARCH_PIXEL_SIZE_NM:.1f} x "
        f"{LANDMARK_HEIGHT_NM / SEARCH_PIXEL_SIZE_NM:.1f} px"
    )

    print()
    print("Reference:")
    print(
        "1000 x 1000 px @ 1 nm/px"
    )

    print()
    print("Search:")
    print(
        "1000 x 1000 px @ 10 nm/px"
    )

    # ========================================================
    # STEP 1
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
    # STEP 2
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
    # STEP 3
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
        blur_sigma=0.60,
        photon_level=900,
        detector_sigma=1.2,
        background_amplitude=2.0,
        scan_amplitude=0.50,
        charging_amplitude=1.6,
    )

    # --------------------------------------------------------
    # Search
    # --------------------------------------------------------

    search = simulate_sem(
        search,
        search_rng,
        blur_sigma=0.52,
        photon_level=450,
        detector_sigma=2.2,
        background_amplitude=4.0,
        scan_amplitude=1.1,
        charging_amplitude=3.2,
    )

    # ========================================================
    # STEP 5
    # ========================================================

    print(
        "[5/6] Creating ground truth..."
    )

    # Physical coordinate -> search pixel coordinate.

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
        + target_width / 2.0
    )

    center_y = (
        search_y
        + target_height / 2.0
    )

    # Landmark size in search pixels.

    landmark_search_width = (
        LANDMARK_WIDTH_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    landmark_search_height = (
        LANDMARK_HEIGHT_NM
        / SEARCH_PIXEL_SIZE_NM
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
            "dram_07",

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
                "small_dark_recessed_rectangle",

            "landmark_size_nm":
                [
                    LANDMARK_WIDTH_NM,
                    LANDMARK_HEIGHT_NM,
                ],

            "landmark_size_search_px":
                [
                    landmark_search_width,
                    landmark_search_height,
                ],

            "landmark_count":
                1,
        },

        "pattern": {

            "background":
                "dense_circular_contact_array",

            "contact_pitch_nm":
                CONTACT_PITCH_NM,

            "contact_radius_nm":
                CONTACT_RADIUS_NM,

            "support_line_width_nm":
                SUPPORT_LINE_WIDTH_NM,

            "block_size_nm":
                BLOCK_SIZE_NM,

            "street_width_nm":
                STREET_WIDTH_NM,

            "landmark":
                "small_dark_recessed_rectangle",
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
                450,

            "reference_detector_sigma":
                1.2,

            "search_detector_sigma":
                2.2,
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
    print("=" * 76)
    print("DRAM_07 GENERATED SUCCESSFULLY")
    print("=" * 76)

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
    print("Target search box:")
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
    print("Actual dark landmark:")
    print(
        f"  {LANDMARK_WIDTH_NM} x "
        f"{LANDMARK_HEIGHT_NM} nm"
    )

    print(
        f"  {landmark_search_width:.1f} x "
        f"{landmark_search_height:.1f} search px"
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
    print("=" * 76)


if __name__ == "__main__":
    main()