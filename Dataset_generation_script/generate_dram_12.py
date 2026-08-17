#!/usr/bin/env python3
"""
DRIFT-SENSE - DRAM-12 synthetic pair generator

Creates:
    results/dram_12/reference_100x.png
    results/dram_12/search_10x.png
    results/dram_12/ground_truth.json

The reference is a 1000 nm x 1000 nm physical crop at 1 nm/px.
The search is the complete 10000 nm x 10000 nm scene at 10 nm/px.
The same physical scene therefore guarantees a true reference/search pair.
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

SEED = 20260821

# Top-left origin of the reference crop in physical coordinates.
TARGET_X_NM = 4140
TARGET_Y_NM = 4140

DIE_GAP_NM = 300
DIE_MARGIN_NM = 200

# Automatically fit exactly four dies across the 10,000 nm scene:
#
#   2*margin + 4*die + 3*gap <= physical_size
#
# With the current values:
#
#   400 + 4*2175 + 900 = 10000 nm
#
# Keeping this derived prevents the previous 10100 nm geometry error.
DIE_SIZE_NM = (
    PHYSICAL_SIZE_NM
    - 2 * DIE_MARGIN_NM
    - 3 * DIE_GAP_NM
) // 4

LANDMARK_WIDTH_NM = 140
LANDMARK_HEIGHT_NM = 150

REFERENCE_DOSE = 2600
SEARCH_DOSE = 720

REFERENCE_BEAM_SIGMA = 0.36
SEARCH_BEAM_SIGMA_X = 0.58
SEARCH_BEAM_SIGMA_Y = 0.68

REFERENCE_READOUT_SIGMA = 0.22
SEARCH_READOUT_SIGMA = 0.58

REFERENCE_GRAIN = 0.003
SEARCH_GRAIN = 0.012

SEARCH_COLUMN_VARIATION = 0.75
SEARCH_SCAN_VARIATION = 0.22
SEARCH_FIELD_VARIATION = 0.75
SEARCH_GAIN_VARIATION = 0.006

SEARCH_HOT_PIXEL_PROBABILITY = 0.00015
SEARCH_DEAD_PIXEL_PROBABILITY = 0.00010

BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "results" / "generated_dataset_images" / "dram_12"


# ============================================================
# DRAM DIE PRESETS
# ============================================================

DIE_PRESETS = {
    "diamond": {
        "pitch_x": 128,
        "pitch_y": 138,
        "contact_w": 42,
        "contact_h": 38,
        "rail_width": 7,
        "wordline_width": 8,
        "offset": 42,
        "contact_intensity": 188,
        "rail_intensity": 67,
        "background": 31,
    },
    "compact": {
        "pitch_x": 105,
        "pitch_y": 112,
        "contact_w": 37,
        "contact_h": 34,
        "rail_width": 7,
        "wordline_width": 7,
        "offset": 28,
        "contact_intensity": 184,
        "rail_intensity": 66,
        "background": 32,
    },
    "open": {
        "pitch_x": 158,
        "pitch_y": 160,
        "contact_w": 52,
        "contact_h": 46,
        "rail_width": 9,
        "wordline_width": 8,
        "offset": 48,
        "contact_intensity": 180,
        "rail_intensity": 65,
        "background": 34,
    },
}

DIE_MAP = [
    ["diamond", "compact", "open", "diamond"],
    ["open", "diamond", "compact", "open"],
    ["compact", "open", "diamond", "compact"],
    ["diamond", "compact", "open", "diamond"],
]


# ============================================================
# SAFE DRAWING HELPERS
# ============================================================

def draw_line(
    image,
    x1,
    y1,
    x2,
    y2,
    intensity,
    width=1,
):
    """
    IMPORTANT:
    width has a safe default so a missing width can never cause
    the previous TypeError again.
    """

    cv2.line(
        image,
        (int(round(x1)), int(round(y1))),
        (int(round(x2)), int(round(y2))),
        int(np.clip(intensity, 0, 255)),
        max(1, int(round(width))),
        lineType=cv2.LINE_AA,
    )


def draw_diamond_contact(image, cx, cy, width, height, intensity):
    half_w = width / 2.0
    half_h = height / 2.0

    points = np.array(
        [
            [cx, cy - half_h],
            [cx + half_w, cy],
            [cx, cy + half_h],
            [cx - half_w, cy],
        ],
        dtype=np.float32,
    )

    points = np.round(points).astype(np.int32)

    cv2.fillPoly(
        image,
        [points],
        int(np.clip(intensity, 0, 255)),
    )

    # Centered inner diamond.
    inner_scale = 0.72
    inner = (
        np.array([cx, cy], dtype=np.float32)
        + (points.astype(np.float32) - np.array([cx, cy]))
        * inner_scale
    )

    inner = np.round(inner).astype(np.int32)

    cv2.fillPoly(
        image,
        [inner],
        int(np.clip(intensity - 12, 0, 255)),
    )


# ============================================================
# NORMAL DRAM CELL
# ============================================================

def draw_normal_cell(image, cx, cy, preset, row, col, rng):
    pitch_x = preset["pitch_x"]
    pitch_y = preset["pitch_y"]

    # Small manufacturing variation.
    cx += rng.normal(0.0, 0.8)
    cy += rng.normal(0.0, 0.8)

    # Vertical bitline.
    draw_line(
        image,
        cx,
        cy - pitch_y * 0.47,
        cx,
        cy + pitch_y * 0.47,
        preset["rail_intensity"],
        preset["rail_width"],
    )

    # Alternating horizontal wordline.
    direction = 1.0 if row % 2 == 0 else -1.0

    draw_line(
        image,
        cx - direction * pitch_x * 0.42,
        cy,
        cx + direction * pitch_x * 0.42,
        cy,
        preset["rail_intensity"] + 3,
        preset["wordline_width"],
    )

    # Staggered storage contact.
    offset_sign = 1.0 if row % 2 == 0 else -1.0

    contact_x = (
        cx
        + offset_sign * preset["offset"] * 0.20
    )

    contact_y = cy + (5 if col % 2 == 0 else -5)

    draw_diamond_contact(
        image,
        contact_x,
        contact_y,
        preset["contact_w"] + rng.normal(0, 1.0),
        preset["contact_h"] + rng.normal(0, 1.0),
        preset["contact_intensity"] + rng.normal(0, 1.7),
    )


# ============================================================
# DIE
# ============================================================

def draw_die(image, x0, y0, die_type, rng, is_target=False):
    preset = DIE_PRESETS[die_type]

    pitch_x = preset["pitch_x"]
    pitch_y = preset["pitch_y"]

    # Die body.
    cv2.rectangle(
        image,
        (int(x0), int(y0)),
        (
            int(x0 + DIE_SIZE_NM),
            int(y0 + DIE_SIZE_NM),
        ),
        int(preset["background"]),
        thickness=-1,
    )

    # Die border.
    cv2.rectangle(
        image,
        (int(x0 + 18), int(y0 + 18)),
        (
            int(x0 + DIE_SIZE_NM - 18),
            int(y0 + DIE_SIZE_NM - 18),
        ),
        48,
        thickness=6,
        lineType=cv2.LINE_AA,
    )

    # Horizontal wordline bands.
    row_index = 0
    y = y0 + 75

    while y < y0 + DIE_SIZE_NM - 55:
        line_intensity = 50 if row_index % 2 == 0 else 45

        # Explicit keyword width: no positional-argument mistake.
        draw_line(
            image=image,
            x1=x0 + 25,
            y1=y,
            x2=x0 + DIE_SIZE_NM - 25,
            y2=y,
            intensity=line_intensity,
            width=5,
        )

        y += pitch_y
        row_index += 1

    # Cell array.
    row = 0
    y = y0 + 78

    while y < y0 + DIE_SIZE_NM - 55:
        stagger = 0 if row % 2 == 0 else preset["offset"]

        x = x0 + 70 + stagger
        col = 0

        while x < x0 + DIE_SIZE_NM - 45:
            landmark_region = False

            if is_target:
                landmark_cx = (
                    TARGET_X_NM + LANDMARK_WIDTH_NM / 2.0
                )
                landmark_cy = (
                    TARGET_Y_NM + LANDMARK_HEIGHT_NM / 2.0
                )

                distance = np.hypot(
                    x - landmark_cx,
                    y - landmark_cy,
                )

                if distance < 100:
                    landmark_region = True

            if not landmark_region:
                draw_normal_cell(
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

    # Larger-scale vertical bitline bundles.
    bundle_spacing = pitch_x * 5
    x = x0 + bundle_spacing

    while x < x0 + DIE_SIZE_NM - 10:
        draw_line(
            image=image,
            x1=x,
            y1=y0 + 25,
            x2=x,
            y2=y0 + DIE_SIZE_NM - 25,
            intensity=43,
            width=4,
        )
        x += bundle_spacing


# ============================================================
# UNIQUE SMALL LANDMARK
# ============================================================

def draw_landmark(image):
    """
    Small asymmetric landmark.

        diamond       diamond
             \-------/
                |
        diamond

    Lower-right contact is intentionally absent.
    """

    cx = TARGET_X_NM + LANDMARK_WIDTH_NM / 2.0
    cy = TARGET_Y_NM + LANDMARK_HEIGHT_NM / 2.0

    # Clear only the small local region.
    cv2.rectangle(
        image,
        (
            int(cx - LANDMARK_WIDTH_NM / 2.0),
            int(cy - LANDMARK_HEIGHT_NM / 2.0),
        ),
        (
            int(cx + LANDMARK_WIDTH_NM / 2.0),
            int(cy + LANDMARK_HEIGHT_NM / 2.0),
        ),
        32,
        thickness=-1,
    )

    # Three contacts.
    draw_diamond_contact(image, cx - 38, cy - 28, 38, 34, 188)
    draw_diamond_contact(image, cx + 35, cy - 25, 38, 34, 185)
    draw_diamond_contact(image, cx - 36, cy + 35, 38, 34, 181)

    # Missing fourth contact is deliberate.

    # Short bridge.
    draw_line(
        image,
        cx - 28,
        cy - 4,
        cx + 27,
        cy - 4,
        77,
        8,
    )

    # Asymmetric diagonal.
    draw_line(
        image,
        cx - 27,
        cy - 4,
        cx - 8,
        cy + 26,
        82,
        7,
    )

    # Small edge feature.
    draw_line(
        image,
        cx + 18,
        cy - 42,
        cx + 45,
        cy - 17,
        105,
        4,
    )

    # Small dark notch.
    cv2.rectangle(
        image,
        (int(cx + 5), int(cy + 5)),
        (int(cx + 28), int(cy + 25)),
        29,
        thickness=-1,
    )


# ============================================================
# GEOMETRY VALIDATION
# ============================================================

def validate_geometry():
    if TARGET_X_NM < 0 or TARGET_Y_NM < 0:
        raise ValueError("Target origin cannot be negative.")

    if TARGET_X_NM + REFERENCE_SIZE_NM > PHYSICAL_SIZE_NM:
        raise ValueError("Reference exceeds scene width.")

    if TARGET_Y_NM + REFERENCE_SIZE_NM > PHYSICAL_SIZE_NM:
        raise ValueError("Reference exceeds scene height.")

    # Check that the 4x4 die grid fits inside the scene.
    grid_extent = (
        2 * DIE_MARGIN_NM
        + 4 * DIE_SIZE_NM
        + 3 * DIE_GAP_NM
    )

    if grid_extent > PHYSICAL_SIZE_NM:
        raise ValueError(
            f"Internal geometry error: die grid is "
            f"{grid_extent} nm but scene is "
            f"{PHYSICAL_SIZE_NM} nm."
        )

    # Ensure the computed die size is valid and that the complete
    # 4x4 array fits exactly within the physical scene.
    if DIE_SIZE_NM <= 0:
        raise ValueError(
            "Computed DIE_SIZE_NM is not positive."
        )

    if grid_extent != PHYSICAL_SIZE_NM:
        raise ValueError(
            f"Internal geometry mismatch: grid is "
            f"{grid_extent} nm; expected "
            f"{PHYSICAL_SIZE_NM} nm."
        )

    # Landmark must be inside the reference crop.
    landmark_x = TARGET_X_NM + LANDMARK_WIDTH_NM / 2
    landmark_y = TARGET_Y_NM + LANDMARK_HEIGHT_NM / 2

    if not (
        TARGET_X_NM <= landmark_x <= TARGET_X_NM + REFERENCE_SIZE_NM
    ):
        raise ValueError("Landmark is outside reference X range.")

    if not (
        TARGET_Y_NM <= landmark_y <= TARGET_Y_NM + REFERENCE_SIZE_NM
    ):
        raise ValueError("Landmark is outside reference Y range.")


# ============================================================
# PHYSICAL SCENE
# ============================================================

def generate_physical_scene():
    validate_geometry()

    rng = np.random.default_rng(SEED)

    scene = np.full(
        (PHYSICAL_SIZE_NM, PHYSICAL_SIZE_NM),
        24,
        dtype=np.uint8,
    )

    for row in range(4):
        for col in range(4):
            x0 = (
                DIE_MARGIN_NM
                + col * (DIE_SIZE_NM + DIE_GAP_NM)
            )
            y0 = (
                DIE_MARGIN_NM
                + row * (DIE_SIZE_NM + DIE_GAP_NM)
            )

            die_type = DIE_MAP[row][col]

            # Target crop is inside die row=1, col=1.
            is_target = row == 1 and col == 1

            draw_die(
                scene,
                x0,
                y0,
                die_type,
                rng,
                is_target=is_target,
            )

    # Insert the unique local landmark.
    draw_landmark(scene)

    # Tiny physical-space beam response.
    scene = cv2.GaussianBlur(
        scene,
        (0, 0),
        sigmaX=REFERENCE_BEAM_SIGMA,
        sigmaY=REFERENCE_BEAM_SIGMA,
    )

    return scene


# ============================================================
# NOISE / ACQUISITION
# ============================================================

def add_poisson_noise(image, rng, dose):
    image_f = np.clip(image.astype(np.float32), 0, 255)

    counts = np.maximum(
        image_f / 255.0 * dose,
        0.01,
    )

    result = rng.poisson(counts).astype(np.float32)
    result = result / dose * 255.0

    return result


def add_column_response(image, rng, strength):
    h, w = image.shape

    columns = rng.normal(0, 1, w).astype(np.float32)

    columns = cv2.GaussianBlur(
        columns.reshape(1, -1),
        (0, 0),
        sigmaX=7.0,
        sigmaY=1.0,
    ).reshape(-1)

    std = columns.std()
    if std > 1e-6:
        columns /= std

    gain = 1.0 + (strength / 100.0) * columns

    return image * gain[None, :]


def add_low_frequency_field(image, rng, strength):
    h, w = image.shape

    low = rng.normal(0, 1, (12, 12)).astype(np.float32)

    field = cv2.resize(
        low,
        (w, h),
        interpolation=cv2.INTER_CUBIC,
    )

    field -= field.mean()

    std = field.std()
    if std > 1e-6:
        field /= std

    return image + field * strength


def add_horizontal_scan(image, rng, strength):
    h, w = image.shape

    rows = rng.normal(0, 1, h).astype(np.float32)

    rows = cv2.GaussianBlur(
        rows.reshape(-1, 1),
        (1, 0),
        sigmaX=0,
        sigmaY=3.0,
    ).reshape(-1)

    std = rows.std()
    if std > 1e-6:
        rows /= std

    return image + rows[:, None] * strength


def add_detector_gain(image, rng, strength):
    gain = rng.normal(
        1.0,
        strength,
        image.shape,
    ).astype(np.float32)

    gain = cv2.GaussianBlur(
        gain,
        (0, 0),
        sigmaX=0.6,
        sigmaY=0.6,
    )

    return image * gain


def add_light_grain(image, rng, strength):
    grain = rng.normal(
        0,
        1,
        image.shape,
    ).astype(np.float32)

    grain = cv2.GaussianBlur(
        grain,
        (0, 0),
        sigmaX=0.5,
        sigmaY=0.5,
    )

    std = grain.std()
    if std > 1e-6:
        grain /= std

    return image * (1.0 + strength * grain)


def add_sparse_events(image, rng):
    output = image.copy()

    random = rng.random(image.shape)

    hot = random < SEARCH_HOT_PIXEL_PROBABILITY

    dead = (
        (random >= SEARCH_HOT_PIXEL_PROBABILITY)
        &
        (
            random
            <
            SEARCH_HOT_PIXEL_PROBABILITY
            + SEARCH_DEAD_PIXEL_PROBABILITY
        )
    )

    output[hot] = np.clip(
        output[hot].astype(np.float32) + 30,
        0,
        255,
    )

    output[dead] *= 0.45

    return output


# ============================================================
# REFERENCE IMAGE
# ============================================================

def simulate_reference(image, rng):
    result = cv2.GaussianBlur(
        image.astype(np.float32),
        (0, 0),
        sigmaX=0.38,
        sigmaY=0.38,
    )

    result = add_poisson_noise(
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

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# SEARCH IMAGE
# ============================================================

def simulate_search(image, rng):
    # Slight anisotropic SEM blur.
    result = cv2.GaussianBlur(
        image.astype(np.float32),
        (0, 0),
        sigmaX=SEARCH_BEAM_SIGMA_X,
        sigmaY=SEARCH_BEAM_SIGMA_Y,
    )

    # Lower-dose shot noise.
    result = add_poisson_noise(
        result,
        rng,
        SEARCH_DOSE,
    )

    # Read noise.
    result += rng.normal(
        0,
        SEARCH_READOUT_SIGMA,
        result.shape,
    )

    # DRAM-12-specific mild acquisition effects.
    result = add_column_response(
        result,
        rng,
        SEARCH_COLUMN_VARIATION,
    )

    result = add_horizontal_scan(
        result,
        rng,
        SEARCH_SCAN_VARIATION,
    )

    result = add_low_frequency_field(
        result,
        rng,
        SEARCH_FIELD_VARIATION,
    )

    result = add_detector_gain(
        result,
        rng,
        SEARCH_GAIN_VARIATION,
    )

    result = add_light_grain(
        result,
        rng,
        SEARCH_GRAIN,
    )

    result = add_sparse_events(
        result,
        rng,
    )

    return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# MANIFEST
# ============================================================

def create_manifest():
    search_x = TARGET_X_NM / SEARCH_PIXEL_SIZE_NM
    search_y = TARGET_Y_NM / SEARCH_PIXEL_SIZE_NM

    target_w = REFERENCE_SIZE_NM / SEARCH_PIXEL_SIZE_NM
    target_h = target_w

    center_x = search_x + target_w / 2.0
    center_y = search_y + target_h / 2.0

    return {
        "pair_id": "dram_12",
        "architecture": "DRAM",
        "seed": SEED,
        "structure_family": "staggered_diamond_contact_dram",

        "reference": {
            "filename": "reference_100x.png",
            "width_px": 1000,
            "height_px": 1000,
            "pixel_size_nm": 1,
            "magnification": "100x",
        },

        "search": {
            "filename": "search_10x.png",
            "width_px": 1000,
            "height_px": 1000,
            "pixel_size_nm": 10,
            "magnification": "10x",
        },

        "target": {
            "physical_origin_nm": [
                TARGET_X_NM,
                TARGET_Y_NM,
            ],
            "search_box_xywh": [
                float(search_x),
                float(search_y),
                float(target_w),
                float(target_h),
            ],
            "search_center_xy": [
                float(center_x),
                float(center_y),
            ],
            "landmark_type": "double_contact_missing_bridge",
            "landmark_size_nm": [
                LANDMARK_WIDTH_NM,
                LANDMARK_HEIGHT_NM,
            ],
            "landmark_size_search_px": [
                LANDMARK_WIDTH_NM / SEARCH_PIXEL_SIZE_NM,
                LANDMARK_HEIGHT_NM / SEARCH_PIXEL_SIZE_NM,
            ],
            "landmark_count": 1,
        },

        "die_layout": DIE_MAP,
        "die_presets": DIE_PRESETS,

        "imaging": {
            "reference_dose": REFERENCE_DOSE,
            "search_dose": SEARCH_DOSE,
            "reference_beam_sigma": REFERENCE_BEAM_SIGMA,
            "search_beam_sigma_x": SEARCH_BEAM_SIGMA_X,
            "search_beam_sigma_y": SEARCH_BEAM_SIGMA_Y,
        },

        "noise": {
            "search_column_response": SEARCH_COLUMN_VARIATION,
            "search_horizontal_scan": SEARCH_SCAN_VARIATION,
            "search_field_variation": SEARCH_FIELD_VARIATION,
            "search_detector_gain": SEARCH_GAIN_VARIATION,
            "search_readout_sigma": SEARCH_READOUT_SIGMA,
            "search_grain": SEARCH_GRAIN,
            "search_hot_pixel_probability": SEARCH_HOT_PIXEL_PROBABILITY,
            "search_dead_pixel_probability": SEARCH_DEAD_PIXEL_PROBABILITY,
        },

        "generation": {
            "physical_scene_size_nm": [
                PHYSICAL_SIZE_NM,
                PHYSICAL_SIZE_NM,
            ],
            "same_physical_scene": True,
            "reference_is_exact_crop": True,
            "search_is_area_downsampled": True,
            "scale_ratio": 10,
        },

        "coordinate_convention": {
            "origin": "top_left",
            "x_direction": "right",
            "y_direction": "down",
        },
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print()
    print("=" * 78)
    print("DRIFT-SENSE - DRAM_12 GENERATOR")
    print("=" * 78)

    print()
    print("Structure:")
    print("  staggered diamond-contact DRAM")
    print("  alternating wordlines + bitlines")
    print("  dense / compact / open die variation")

    print()
    print("Landmark:")
    print("  small asymmetric double-contact + missing bridge")

    print()
    print("Noise:")
    print("  mild column response")
    print("  weak horizontal scan variation")
    print("  low-frequency field")
    print("  light detector gain variation")

    print()
    print("[0/6] Validating geometry...")
    validate_geometry()
    print("      Geometry OK")
    print(
        f"      Scene: {PHYSICAL_SIZE_NM} x {PHYSICAL_SIZE_NM} nm"
    )
    print(
        f"      Die: {DIE_SIZE_NM} x {DIE_SIZE_NM} nm"
    )
    print(
        f"      Gap: {DIE_GAP_NM} nm"
    )
    print(
        f"      Margin: {DIE_MARGIN_NM} nm"
    )
    print(
        f"      4x4 die extent: "
        f"{2 * DIE_MARGIN_NM + 4 * DIE_SIZE_NM + 3 * DIE_GAP_NM} nm"
    )

    print()
    print("[1/6] Generating DRAM-12 physical scene...")
    scene = generate_physical_scene()
    print(
        f"      Physical scene: "
        f"{scene.shape[1]} x {scene.shape[0]} nm"
    )

    print()
    print("[2/6] Extracting 100x reference...")

    reference = scene[
        TARGET_Y_NM:TARGET_Y_NM + REFERENCE_SIZE_NM,
        TARGET_X_NM:TARGET_X_NM + REFERENCE_SIZE_NM,
    ].copy()

    if reference.shape != (1000, 1000):
        raise RuntimeError(
            f"Reference shape is {reference.shape}, expected (1000, 1000)."
        )

    print("      Reference: 1000 x 1000 px @ 1 nm/px")

    print()
    print("[3/6] Creating 10x wide search...")

    search = cv2.resize(
        scene,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    if search.shape != (1000, 1000):
        raise RuntimeError(
            f"Search shape is {search.shape}, expected (1000, 1000)."
        )

    print("      Search: 1000 x 1000 px @ 10 nm/px")

    print()
    print("[4/6] Simulating clean reference...")

    reference_rng = np.random.default_rng(SEED + 100)

    reference = simulate_reference(
        reference,
        reference_rng,
    )

    print()
    print("[5/6] Simulating mild search acquisition...")

    search_rng = np.random.default_rng(SEED + 200)

    search = simulate_search(
        search,
        search_rng,
    )

    print()
    print("[6/6] Saving outputs...")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    reference_path = OUTPUT_DIR / "reference_100x.png"
    search_path = OUTPUT_DIR / "search_10x.png"
    manifest_path = OUTPUT_DIR / "ground_truth.json"

    if not cv2.imwrite(
        str(reference_path),
        reference,
    ):
        raise RuntimeError(
            f"Could not save {reference_path}"
        )

    if not cv2.imwrite(
        str(search_path),
        search,
    ):
        raise RuntimeError(
            f"Could not save {search_path}"
        )

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            create_manifest(),
            f,
            indent=4,
        )

    search_x = TARGET_X_NM / SEARCH_PIXEL_SIZE_NM
    search_y = TARGET_Y_NM / SEARCH_PIXEL_SIZE_NM
    target_w = REFERENCE_SIZE_NM / SEARCH_PIXEL_SIZE_NM
    target_h = target_w

    print()
    print("=" * 78)
    print("DRAM_12 GENERATED SUCCESSFULLY")
    print("=" * 78)

    print()
    print("Reference:")
    print(f"  {reference_path}")

    print()
    print("Search:")
    print(f"  {search_path}")

    print()
    print("Manifest:")
    print(f"  {manifest_path}")

    print()
    print("Ground truth in search image:")
    print(f"  x      = {search_x:.1f} px")
    print(f"  y      = {search_y:.1f} px")
    print(f"  width  = {target_w:.1f} px")
    print(f"  height = {target_h:.1f} px")
    print(
        f"  center = "
        f"({search_x + target_w / 2:.1f}, "
        f"{search_y + target_h / 2:.1f}) px"
    )

    print()
    print("Landmark:")
    print(
        f"  {LANDMARK_WIDTH_NM} x "
        f"{LANDMARK_HEIGHT_NM} nm"
    )
    print(
        f"  {LANDMARK_WIDTH_NM / SEARCH_PIXEL_SIZE_NM:.1f} x "
        f"{LANDMARK_HEIGHT_NM / SEARCH_PIXEL_SIZE_NM:.1f} px "
        f"in search"
    )

    print()
    print("Output directory:")
    print(f"  {OUTPUT_DIR}")

    print()
    print("=" * 78)


if __name__ == "__main__":
    main()