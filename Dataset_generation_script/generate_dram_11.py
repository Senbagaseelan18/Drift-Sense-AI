#!/usr/bin/env python3

"""
DRIFT-SENSE
DRAM-11 Synthetic Dataset Generator
====================================

DRAM-11 introduces DIE-TO-DIE STRUCTURAL VARIATION.

Unlike previous pairs where every macro block had almost
the same density, this scene contains multiple DRAM die
styles:

    DENSE
    LOOSE
    WIDE
    MIXED

Each die has its own:

    * cell pitch
    * contact size
    * horizontal/vertical line spacing
    * local density
    * row offset
    * edge geometry

The reference is cropped from one particular die and that
same physical region is present in the 10x search image.

TARGET FEATURE
--------------

One small asymmetric cell displacement is inserted into
the target die.

It is deliberately subtle.

NOISE
-----

DRAM-11 uses LESS noise than DRAM-09 / DRAM-10.

Reference:
    very clean

Search:
    mild shot noise
    mild detector noise
    mild grain
    tiny scan variation
    very small blur

No strong banding.
No strong charging.
No large artificial noise structures.
"""

from pathlib import Path
import json

import cv2
import numpy as np


# ============================================================
# GLOBAL
# ============================================================

IMAGE_SIZE = 1000

PHYSICAL_SIZE_NM = 10000

REFERENCE_SIZE_NM = 1000

SEARCH_PIXEL_SIZE_NM = 10

REFERENCE_PIXEL_SIZE_NM = 1

SEED = 20260820


# ============================================================
# TARGET REFERENCE CROP
# ============================================================

TARGET_X_NM = 4010

TARGET_Y_NM = 4010


# ============================================================
# DIE GEOMETRY
# ============================================================

DIE_SIZE_NM = 2200

DIE_GAP_NM = 300

DIE_MARGIN_NM = 200


# ============================================================
# TARGET FEATURE
# ============================================================

LANDMARK_WIDTH_NM = 145

LANDMARK_HEIGHT_NM = 150


# ============================================================
# NOISE
# ============================================================

REFERENCE_DOSE = 2500

SEARCH_DOSE = 650

REFERENCE_READOUT_SIGMA = 0.25

SEARCH_READOUT_SIGMA = 0.65

REFERENCE_GRAIN = 0.004

SEARCH_GRAIN = 0.018

SEARCH_SCAN_VARIATION = 0.30

SEARCH_BLUR_SIGMA = 0.62


# ============================================================
# OUTPUT
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "results"
    / "generated_dataset_images"
    / "dram_11"
)


# ============================================================
# DIE PRESETS
# ============================================================

DIE_PRESETS = {

    "dense": {

        "pitch_x": 105,

        "pitch_y": 108,

        "contact_w": 42,

        "contact_h": 48,

        "rail_w": 7,

        "cross_w": 7,

        "row_offset": 18,

        "background": 34,

        "contact_intensity": 185,

        "rail_intensity": 69,
    },

    "loose": {

        "pitch_x": 155,

        "pitch_y": 160,

        "contact_w": 58,

        "contact_h": 65,

        "rail_w": 9,

        "cross_w": 8,

        "row_offset": 32,

        "background": 37,

        "contact_intensity": 181,

        "rail_intensity": 70,
    },

    "wide": {

        "pitch_x": 185,

        "pitch_y": 135,

        "contact_w": 68,

        "contact_h": 45,

        "rail_w": 10,

        "cross_w": 8,

        "row_offset": 42,

        "background": 35,

        "contact_intensity": 178,

        "rail_intensity": 67,
    },

    "mixed": {

        "pitch_x": 125,

        "pitch_y": 150,

        "contact_w": 48,

        "contact_h": 62,

        "rail_w": 8,

        "cross_w": 7,

        "row_offset": 26,

        "background": 36,

        "contact_intensity": 183,

        "rail_intensity": 68,
    },
}


# ============================================================
# DIE MAP
# ============================================================

DIE_MAP = [

    [
        "dense",
        "loose",
        "wide",
        "dense",
    ],

    [
        "loose",
        "mixed",
        "dense",
        "wide",
    ],

    [
        "wide",
        "dense",
        "loose",
        "mixed",
    ],

    [
        "dense",
        "wide",
        "mixed",
        "loose",
    ],
]


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
# DRAW ONE CELL
# ============================================================

def draw_cell(
    image,
    cx,
    cy,
    preset,
    row,
    col,
    rng,
    special=False,
):

    pitch_x = preset["pitch_x"]

    pitch_y = preset["pitch_y"]

    contact_w = preset["contact_w"]

    contact_h = preset["contact_h"]

    rail_w = preset["rail_w"]

    cross_w = preset["cross_w"]

    contact_intensity = (
        preset["contact_intensity"]
    )

    rail_intensity = (
        preset["rail_intensity"]
    )

    # --------------------------------------------------------
    # Small manufacturing variation
    # --------------------------------------------------------

    cx += rng.normal(
        0,
        1.0,
    )

    cy += rng.normal(
        0,
        1.0,
    )

    local_w = (
        contact_w
        + rng.normal(
            0,
            1.5,
        )
    )

    local_h = (
        contact_h
        + rng.normal(
            0,
            1.5,
        )
    )

    # --------------------------------------------------------
    # Vertical interconnect
    # --------------------------------------------------------

    draw_line(
        image,
        cx,
        cy - pitch_y * 0.46,
        cx,
        cy + pitch_y * 0.46,
        rail_intensity,
        rail_w,
    )

    # --------------------------------------------------------
    # Horizontal interconnect
    # --------------------------------------------------------

    direction = (
        -1
        if col % 2
        else 1
    )

    draw_line(
        image,
        cx - direction * pitch_x * 0.40,
        cy,
        cx + direction * pitch_x * 0.40,
        cy,
        rail_intensity + 2,
        cross_w,
    )

    # --------------------------------------------------------
    # Contact
    # --------------------------------------------------------

    x0 = int(
        round(
            cx - local_w / 2
        )
    )

    y0 = int(
        round(
            cy - local_h / 2
        )
    )

    x1 = int(
        round(
            cx + local_w / 2
        )
    )

    y1 = int(
        round(
            cy + local_h / 2
        )
    )

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
                contact_intensity
                + rng.normal(
                    0,
                    2.0,
                ),
                0,
                255,
            )
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Slight contact center
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            x0 + 5,
            y0 + 5,
        ),
        (
            x1 - 5,
            y1 - 5,
        ),
        int(
            contact_intensity - 10
        ),
        thickness=-1,
        lineType=cv2.LINE_AA,
    )


# ============================================================
# DRAW ONE DIE
# ============================================================

def draw_die(
    image,
    x0,
    y0,
    die_type,
    rng,
    target_die=False,
):

    preset = DIE_PRESETS[
        die_type
    ]

    pitch_x = preset["pitch_x"]

    pitch_y = preset["pitch_y"]

    # --------------------------------------------------------
    # Die background
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            int(x0),
            int(y0),
        ),
        (
            int(
                x0
                + DIE_SIZE_NM
            ),
            int(
                y0
                + DIE_SIZE_NM
            ),
        ),
        preset["background"],
        thickness=-1,
    )

    # --------------------------------------------------------
    # Subtle die frame
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            int(x0 + 18),
            int(y0 + 18),
        ),
        (
            int(
                x0
                + DIE_SIZE_NM
                - 18
            ),
            int(
                y0
                + DIE_SIZE_NM
                - 18
            ),
        ),
        52,
        thickness=7,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Rows
    # --------------------------------------------------------

    row = 0

    y = (
        y0
        + 80
    )

    while y < (
        y0
        + DIE_SIZE_NM
        - 60
    ):

        # Different row offsets create different density
        # signatures.

        offset = (
            preset["row_offset"]
            if row % 2
            else 0
        )

        x = (
            x0
            + 75
            + offset
        )

        col = 0

        while x < (
            x0
            + DIE_SIZE_NM
            - 50
        ):

            special = False

            # ------------------------------------------------
            # Target landmark location
            # ------------------------------------------------

            if target_die:

                feature_cx = (
                    TARGET_X_NM
                    + LANDMARK_WIDTH_NM
                    / 2
                )

                feature_cy = (
                    TARGET_Y_NM
                    + LANDMARK_HEIGHT_NM
                    / 2
                )

                distance = np.sqrt(
                    (
                        x
                        - feature_cx
                    ) ** 2
                    +
                    (
                        y
                        - feature_cy
                    ) ** 2
                )

                if distance < 90:

                    special = True

            # ------------------------------------------------
            # Normal cell
            # ------------------------------------------------

            if not special:

                draw_cell(
                    image,
                    x,
                    y,
                    preset,
                    row,
                    col,
                    rng,
                )

            x += pitch_x

            col += 1

        y += pitch_y

        row += 1

    # --------------------------------------------------------
    # Additional die-specific vertical rails
    # --------------------------------------------------------

    if die_type in (
        "dense",
        "wide",
    ):

        rail_spacing = (
            4 * pitch_x
        )

        x = (
            x0
            + rail_spacing
        )

        while x < (
            x0
            + DIE_SIZE_NM
        ):

            draw_line(
                image,
                x,
                y0 + 25,
                x,
                y0
                + DIE_SIZE_NM
                - 25,
                47,
                5,
            )

            x += rail_spacing

    # --------------------------------------------------------
    # Additional horizontal structure for loose dies
    # --------------------------------------------------------

    if die_type == "loose":

        y = (
            y0
            + 3 * pitch_y
        )

        while y < (
            y0
            + DIE_SIZE_NM
        ):

            draw_line(
                image,
                x0 + 25,
                y,
                x0
                + DIE_SIZE_NM
                - 25,
                y,
                48,
                5,
            )

            y += (
                4 * pitch_y
            )


# ============================================================
# UNIQUE TARGET FEATURE
# ============================================================

def draw_landmark(
    image,
):
    """
    Small asymmetric defect.

    Instead of a giant box, one normal region has:

        * one shifted contact
        * one missing contact
        * one short connecting bridge

    This creates a local structural signature.
    """

    cx = (
        TARGET_X_NM
        + LANDMARK_WIDTH_NM / 2
    )

    cy = (
        TARGET_Y_NM
        + LANDMARK_HEIGHT_NM / 2
    )

    # --------------------------------------------------------
    # Clear only a very small region
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            int(
                cx
                - LANDMARK_WIDTH_NM / 2
            ),
            int(
                cy
                - LANDMARK_HEIGHT_NM / 2
            ),
        ),
        (
            int(
                cx
                + LANDMARK_WIDTH_NM / 2
            ),
            int(
                cy
                + LANDMARK_HEIGHT_NM / 2
            ),
        ),
        37,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Shifted left contact
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            int(cx - 45),
            int(cy - 38),
        ),
        (
            int(cx - 5),
            int(cy + 4),
        ),
        184,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Shifted upper contact
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            int(cx + 10),
            int(cy - 48),
        ),
        (
            int(cx + 47),
            int(cy - 10),
        ),
        180,
        thickness=-1,
        lineType=cv2.LINE_AA,
    )

    # --------------------------------------------------------
    # Short bridge
    # --------------------------------------------------------

    draw_line(
        image,
        cx - 30,
        cy - 16,
        cx + 30,
        cy - 16,
        82,
        8,
    )

    # --------------------------------------------------------
    # Small missing-contact region
    # --------------------------------------------------------

    cv2.rectangle(
        image,
        (
            int(cx + 2),
            int(cy + 8),
        ),
        (
            int(cx + 35),
            int(cy + 43),
        ),
        30,
        thickness=-1,
    )

    # --------------------------------------------------------
    # Tiny bright edge
    # --------------------------------------------------------

    draw_line(
        image,
        cx + 12,
        cy - 47,
        cx + 47,
        cy - 47,
        108,
        4,
    )


# ============================================================
# PHYSICAL SCENE
# ============================================================

def generate_scene():

    rng = np.random.default_rng(
        SEED
    )

    canvas = np.full(
        (
            PHYSICAL_SIZE_NM,
            PHYSICAL_SIZE_NM,
        ),
        25,
        dtype=np.uint8,
    )

    # ========================================================
    # DIE POSITIONS
    # ========================================================

    positions = []

    for row in range(4):

        for col in range(4):

            x = (
                DIE_MARGIN_NM
                + col
                * (
                    DIE_SIZE_NM
                    + DIE_GAP_NM
                )
            )

            y = (
                DIE_MARGIN_NM
                + row
                * (
                    DIE_SIZE_NM
                    + DIE_GAP_NM
                )
            )

            positions.append(
                (
                    row,
                    col,
                    x,
                    y,
                )
            )

    # ========================================================
    # DRAW ALL DIES
    # ========================================================

    for (
        row,
        col,
        x,
        y,
    ) in positions:

        die_type = DIE_MAP[
            row
        ][
            col
        ]

        # Target crop belongs to die at approximately
        # row=1,col=1.

        target_die = (
            row == 1
            and col == 1
        )

        draw_die(
            canvas,
            x,
            y,
            die_type,
            rng,
            target_die=target_die,
        )

    # ========================================================
    # UNIQUE TARGET
    # ========================================================

    draw_landmark(
        canvas
    )

    # ========================================================
    # PHYSICAL SEM PSF
    # ========================================================

    canvas = cv2.GaussianBlur(
        canvas,
        (
            0,
            0,
        ),
        sigmaX=0.38,
        sigmaY=0.38,
    )

    return canvas


# ============================================================
# POISSON NOISE
# ============================================================

def poisson_noise(
    image,
    rng,
    dose,
):

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

    result = rng.poisson(
        counts
    ).astype(
        np.float32
    )

    result /= dose

    result *= 255.0

    return result


# ============================================================
# LIGHT CORRELATED GRAIN
# ============================================================

def add_light_grain(
    image,
    rng,
    strength,
):

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
        sigmaX=0.55,
        sigmaY=0.55,
    )

    std = grain.std()

    if std > 1e-6:

        grain /= std

    return (
        image
        * (
            1
            + strength
            * grain
        )
    )


# ============================================================
# LIGHT SCAN VARIATION
# ============================================================

def add_light_scan_variation(
    image,
    rng,
    strength,
):

    h, w = image.shape

    row_noise = rng.normal(
        0,
        strength,
        h,
    ).astype(
        np.float32
    )

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
        sigmaY=1.0,
    ).reshape(
        -1
    )

    return (
        image
        + row_noise[:, None]
    )


# ============================================================
# MILD FIELD VARIATION
# ============================================================

def add_field_variation(
    image,
    rng,
    strength,
):

    h, w = image.shape

    low = rng.normal(
        0,
        1,
        (
            10,
            10,
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
# SEARCH ACQUISITION
# ============================================================

def acquire_reference(
    image,
    rng,
):

    result = cv2.GaussianBlur(
        image.astype(
            np.float32
        ),
        (
            0,
            0,
        ),
        sigmaX=0.40,
        sigmaY=0.40,
    )

    result = poisson_noise(
        result,
        rng,
        REFERENCE_DOSE,
    )

    result += rng.normal(
        0,
        REFERENCE_READOUT_SIGMA,
        result.shape,
    )

    result = add_light_grain(
        result,
        rng,
        REFERENCE_GRAIN,
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(
        np.uint8
    )


def acquire_search(
    image,
    rng,
):

    result = cv2.GaussianBlur(
        image.astype(
            np.float32
        ),
        (
            0,
            0,
        ),
        sigmaX=SEARCH_BLUR_SIGMA,
        sigmaY=SEARCH_BLUR_SIGMA * 1.08,
    )

    result = poisson_noise(
        result,
        rng,
        SEARCH_DOSE,
    )

    result += rng.normal(
        0,
        SEARCH_READOUT_SIGMA,
        result.shape,
    )

    result = add_light_grain(
        result,
        rng,
        SEARCH_GRAIN,
    )

    result = add_light_scan_variation(
        result,
        rng,
        SEARCH_SCAN_VARIATION,
    )

    result = add_field_variation(
        result,
        rng,
        0.9,
    )

    return np.clip(
        result,
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
    print("DRIFT-SENSE — DRAM_11 GENERATOR")
    print("=" * 78)

    print()
    print(
        "DRAM-11 structure:"
    )

    print(
        "  DENSE + LOOSE + WIDE + MIXED dies"
    )

    print()
    print(
        "Noise:"
    )

    print(
        "  LOW / controlled SEM noise"
    )

    print()

    # ========================================================
    # PHYSICAL SCENE
    # ========================================================

    print(
        "[1/6] Generating multi-density DRAM scene..."
    )

    scene = generate_scene()

    # ========================================================
    # REFERENCE
    # ========================================================

    print(
        "[2/6] Extracting reference crop..."
    )

    reference = scene[
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
            "Reference must be 1000x1000."
        )

    # ========================================================
    # SEARCH
    # ========================================================

    print(
        "[3/6] Creating 10x search..."
    )

    search = cv2.resize(
        scene,
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
            "Search must be 1000x1000."
        )

    # ========================================================
    # REFERENCE ACQUISITION
    # ========================================================

    print(
        "[4/6] Applying clean reference acquisition..."
    )

    reference_rng = np.random.default_rng(
        SEED + 100
    )

    reference = acquire_reference(
        reference,
        reference_rng,
    )

    # ========================================================
    # SEARCH ACQUISITION
    # ========================================================

    print(
        "[5/6] Applying mild search noise..."
    )

    search_rng = np.random.default_rng(
        SEED + 200
    )

    search = acquire_search(
        search,
        search_rng,
    )

    # ========================================================
    # TARGET COORDINATES
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

    target_w = (
        REFERENCE_SIZE_NM
        / SEARCH_PIXEL_SIZE_NM
    )

    target_h = target_w

    center_x = (
        search_x
        + target_w / 2
    )

    center_y = (
        search_y
        + target_h / 2
    )

    # ========================================================
    # OUTPUT
    # ========================================================

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
            "dram_11",

        "architecture":
            "DRAM",

        "seed":
            SEED,

        "structure_family":
            "multi_density_dram_die_array",

        "die_layout":
            DIE_MAP,

        "die_types": {

            "dense":
                DIE_PRESETS["dense"],

            "loose":
                DIE_PRESETS["loose"],

            "wide":
                DIE_PRESETS["wide"],

            "mixed":
                DIE_PRESETS["mixed"],
        },

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
                    target_w,
                    target_h,
                ],

            "search_center_xy":
                [
                    center_x,
                    center_y,
                ],

            "landmark_type":
                "small_asymmetric_shifted_missing_contact",

            "landmark_size_nm":
                [
                    LANDMARK_WIDTH_NM,
                    LANDMARK_HEIGHT_NM,
                ],

            "landmark_count":
                1,
        },

        "noise": {

            "reference_dose":
                REFERENCE_DOSE,

            "search_dose":
                SEARCH_DOSE,

            "reference_readout_sigma":
                REFERENCE_READOUT_SIGMA,

            "search_readout_sigma":
                SEARCH_READOUT_SIGMA,

            "reference_grain":
                REFERENCE_GRAIN,

            "search_grain":
                SEARCH_GRAIN,

            "search_scan_variation":
                SEARCH_SCAN_VARIATION,

            "search_blur_sigma":
                SEARCH_BLUR_SIGMA,
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
    # REPORT
    # ========================================================

    print()
    print("=" * 78)
    print("DRAM_11 GENERATED SUCCESSFULLY")
    print("=" * 78)

    print()
    print(
        "Die density layout:"
    )

    for row in DIE_MAP:

        print(
            "   "
            + " | ".join(
                row
            )
        )

    print()

    print(
        "Reference:"
    )

    print(
        f"   {reference_path}"
    )

    print()

    print(
        "Search:"
    )

    print(
        f"   {search_path}"
    )

    print()

    print(
        "Target origin:"
    )

    print(
        f"   ({TARGET_X_NM}, "
        f"{TARGET_Y_NM}) nm"
    )

    print()

    print(
        "Search target:"
    )

    print(
        f"   x = {search_x:.1f}px"
    )

    print(
        f"   y = {search_y:.1f}px"
    )

    print(
        f"   w = {target_w:.1f}px"
    )

    print(
        f"   h = {target_h:.1f}px"
    )

    print()

    print(
        "Noise level: LOW"
    )

    print()

    print(
        f"Output:"
    )

    print(
        f"   {OUTPUT_DIR}"
    )

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()