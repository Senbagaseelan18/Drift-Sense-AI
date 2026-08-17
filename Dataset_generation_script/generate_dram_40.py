import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import os

# =========================================================================
# PHYSICS & ARTIFACT ENGINES
# =========================================================================

def apply_lens_distortion(img, k1):
    if abs(k1) < 0.001: return img
    h, w = img.shape
    K = np.array([[w/1.5, 0, w/2], 
                  [0, h/1.5, h/2], 
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, 0, 0, 0], dtype=np.float32)
    return cv2.undistort(img, K, D)

def apply_vignette_and_gamma(img, vignette_strength, gamma):
    if vignette_strength <= 0 and abs(gamma - 1.0) < 0.01: return img
    h, w = img.shape
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    
    mask = 1.0 - np.clip(radius * vignette_strength, 0, 1)
    img_v = img * mask
    
    if abs(gamma - 1.0) > 0.01:
        img_v = 255.0 * ((img_v / 255.0) ** (1.0 / gamma))
        
    return np.clip(img_v, 0, 255).astype(np.float32)

def apply_charging_streaks(img, streaks_per_100, intensity):
    if streaks_per_100 <= 0 or intensity <= 0: return img
    h, w = img.shape
    out = img.copy()
    
    prob_per_row = streaks_per_100 / 100.0
    for row in range(h):
        if np.random.rand() < prob_per_row:
            streak_len = np.random.randint(w//4, w)
            start_x = np.random.randint(0, w - streak_len)
            gradient = np.linspace(intensity, 0, streak_len)
            out[row, start_x:start_x+streak_len] += gradient
            
    return np.clip(out, 0, 255)

def apply_raster_artifacts(img, shear_amp, jitter_std):
    if shear_amp <= 0 and jitter_std <= 0: return img
    h, w = img.shape
    out = np.zeros_like(img, dtype=np.float32)
    for row in range(h):
        shift = shear_amp * (row / (h - 1.0)) + np.random.normal(0, jitter_std)
        out[row, :] = np.interp(
            np.arange(w) - shift, 
            np.arange(w), 
            img[row, :], 
            left=img[row, 0], 
            right=img[row, -1]
        )
    return out

def apply_acquisition_noise(img, dose, speckle_sigma, salt_pepper_prob):
    counts = (np.clip(img, 0, 255) / 255.0) * dose
    noisy_counts = np.random.poisson(counts).astype(np.float32)
    img_noise = (noisy_counts / dose) * 255.0

    if speckle_sigma > 0:
        speckle = np.random.normal(1.0, speckle_sigma, img_noise.shape)
        img_noise *= speckle

    if salt_pepper_prob > 0:
        rand_matrix = np.random.rand(*img_noise.shape)
        img_noise[rand_matrix < (salt_pepper_prob / 2.0)] = 0.0
        img_noise[(rand_matrix >= (salt_pepper_prob / 2.0)) & (rand_matrix < salt_pepper_prob)] = 255.0

    return np.clip(img_noise, 0, 255).astype(np.uint8)

# =========================================================================
# LAYOUT & GEOMETRY ENGINE WITH UNIQUE SECTION FILTERS & "+" SYMBOL
# =========================================================================

def draw_circular_via_unique_sections_layout(width_px, height_px, pixel_scale_nm, start_x_nm, start_y_nm, meta):
    BACKGROUND = 35.0
    img = np.full((height_px, width_px), BACKGROUND, dtype=np.float32)

    x_coords = np.arange(width_px, dtype=np.float32) * pixel_scale_nm + start_x_nm
    y_coords = np.arange(height_px, dtype=np.float32) * pixel_scale_nm + start_y_nm
    X, Y = np.meshgrid(x_coords, y_coords)

    mat_size = meta["mat_size_nm"]
    strip = meta["strip_width_nm"]
    scale = meta["feature_size_scale"]
    
    block_x = (X // (mat_size + strip)).astype(np.int32)
    block_y = (Y // (mat_size + strip)).astype(np.int32)
    local_x = X % (mat_size + strip)
    local_y = Y % (mat_size + strip)

    in_mat = (local_x < mat_size) & (local_y < mat_size)
    peripheral_mask = ~in_mat
    img[peripheral_mask] = 20.0 # Dark scribe lines

    # =========================================================================
    # ORTHOGONAL CROSSBAR / CIRCULAR VIA ARRAY
    # =========================================================================
    pitch = 100.0 * scale
    trace_cd = 20.0 * scale
    
    vert_trace_mask = (X % pitch) < trace_cd
    horz_trace_mask = (Y % pitch) < trace_cd
    img[in_mat & (vert_trace_mask | horz_trace_mask)] = 75.0

    via_cd = 45.0 * scale
    via_radius = via_cd / 2.0
    dx = np.abs((X % pitch) - (pitch / 2.0))
    dy = np.abs((Y % pitch) - (pitch / 2.0))
    via_mask = (dx**2 + dy**2) < (via_radius**2)
    img[in_mat & via_mask] = 230.0

    # =========================================================================
    # SECTION-SPECIFIC UNIQUE FILTERS (Drastically Different Per Mat/Block)
    # =========================================================================
    section_id = (block_x * 3 + block_y * 7) % 4
    
    # Style 0: Standard High Contrast
    mask_style_0 = in_mat & (section_id == 0)
    # Style 1: Dim & Washed Out
    mask_style_1 = in_mat & (section_id == 1)
    img[mask_style_1] = img[mask_style_1] * 0.4 + 15.0
    
    # Style 2: Glowing / Overexposed Boost
    mask_style_2 = in_mat & (section_id == 2)
    img[mask_style_2] = np.clip(img[mask_style_2] * 1.5, 0, 255)
    
    # Style 3: Inverted Tone / Heavy Contrast Shift
    mask_style_3 = in_mat & (section_id == 3)
    img[mask_style_3] = 255.0 - (img[mask_style_3] * 0.8)

    # =========================================================================
    # SINGLE GUARANTEED FIDUCIAL (THE "+" SYMBOL)
    # =========================================================================
    gt_phys_x = meta["gt_x"] * 10.0
    gt_phys_y = meta["gt_y"] * 10.0
    
    # Clear a dark isolation pad beneath the cross so it pops clearly against any section style
    pad_mask = (np.abs(X - gt_phys_x) < 120.0) & (np.abs(Y - gt_phys_y) < 120.0)
    img[pad_mask] = 30.0
    
    # Draw the single thick "+" symbol
    cross_dx = np.abs(X - gt_phys_x)
    cross_dy = np.abs(Y - gt_phys_y)
    cross_mask = pad_mask & (((cross_dx <= 12.0) & (cross_dy <= 80.0)) | ((cross_dy <= 12.0) & (cross_dx <= 80.0)))
    img[cross_mask] = 255.0

    return img

def generate_pair(index):
    # FIXED SEED hardcoded to 12345
    master_seed = 12345
    np.random.seed(master_seed)
    
    meta = {
        "architecture": "circular_via_unique_sections_cross",
        "pair_id": index,
        "seed": master_seed,
        
        "mat_size_nm": 2600.0,
        "strip_width_nm": 320.0,
        "p_straddle_boundary": 0.35,
        
        "feature_size_scale": 1.00,
        "linewidth_bias_nm": 0.00,
        "pattern_collapse_threshold_nm": 10.00,
        "corner_rounding_px": 0.00,
        
        "beam_spot_size_nm": 5.00,
        "beam_astigmatism_ratio": 1.00,
        "barrel_distortion": 0.00,
        "vignette_strength": 0.00,
        "gamma": 1.00,
        "charging_streaks_per_100_rows": 0.00,
        "charging_streak_intensity": 0.00,
        
        "dose_reference": 2000.00,
        "dose_search": 200.00,
        "shear_amplitude_px": 1.50,       
        "drift_jitter_px": 0.50,          
        "speckle_noise_sigma": 0.00,
        "salt_and_pepper_prob": 0.00
    }

    if np.random.rand() < meta["p_straddle_boundary"]:
        period = meta["mat_size_nm"] + meta["strip_width_nm"]
        block = np.random.randint(1, 4)
        if np.random.rand() > 0.5:
            gt_x_nm = (block * period) - (meta["strip_width_nm"] / 2.0)
            gt_y_nm = np.random.uniform(500.0, 9500.0)
        else:
            gt_x_nm = np.random.uniform(500.0, 9500.0)
            gt_y_nm = (block * period) - (meta["strip_width_nm"] / 2.0)
            
        random_gt_x = gt_x_nm / 10.0
        random_gt_y = gt_y_nm / 10.0
    else:
        random_gt_x = np.random.uniform(50.0, 950.0)
        random_gt_y = np.random.uniform(50.0, 950.0)

    meta["gt_x"] = round(random_gt_x, 2)
    meta["gt_y"] = round(random_gt_y, 2)
    meta["gt_box"] = [round(random_gt_x - 50, 2), round(random_gt_y - 50, 2), 100, 100]

    print(f"Generating Deterministic Dataset with Unique Sections & '+' Symbol (Seed: {master_seed})...")

    # =========================================================================
    # SEARCH IMAGE GENERATION
    # =========================================================================
    search_clean_1nm = draw_circular_via_unique_sections_layout(
        width_px=1000, height_px=1000, pixel_scale_nm=10.0,
        start_x_nm=0.0, start_y_nm=0.0, meta=meta
    )
    
    if meta["corner_rounding_px"] > 0:
        cr = int(meta["corner_rounding_px"] * 2 + 1)
        search_clean_1nm = cv2.GaussianBlur(search_clean_1nm, (cr, cr), 0)

    search_downsampled = cv2.resize(search_clean_1nm, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    
    sigma_x = (meta["beam_spot_size_nm"] / 10.0) / 2.355 
    sigma_y = sigma_x * meta["beam_astigmatism_ratio"]
    k_x, k_y = int(6 * sigma_x + 1) | 1, int(6 * sigma_y + 1) | 1
    search_blurred = cv2.GaussianBlur(search_downsampled, (k_x, k_y), sigmaX=sigma_x, sigmaY=sigma_y)

    search_distorted = apply_lens_distortion(search_blurred, meta["barrel_distortion"])
    search_vignette = apply_vignette_and_gamma(search_distorted, meta["vignette_strength"], meta["gamma"])

    np.random.seed(master_seed + 1)
    search_charged = apply_charging_streaks(search_vignette, meta["charging_streaks_per_100_rows"], meta["charging_streak_intensity"])
    search_drifted = apply_raster_artifacts(search_charged, meta["shear_amplitude_px"], meta["drift_jitter_px"])
    search_final = apply_acquisition_noise(search_drifted, meta["dose_search"], meta["speckle_noise_sigma"], meta["salt_and_pepper_prob"])

    # =========================================================================
    # REFERENCE IMAGE GENERATION
    # =========================================================================
    phys_cx = meta["gt_x"] * 10.0
    phys_cy = meta["gt_y"] * 10.0
    
    ref_clean = draw_circular_via_unique_sections_layout(
        width_px=1000, height_px=1000, pixel_scale_nm=1.0,
        start_x_nm=phys_cx - 500.0, start_y_nm=phys_cy - 500.0, meta=meta
    )
    
    if meta["corner_rounding_px"] > 0:
        cr = int(meta["corner_rounding_px"] * 2 + 1)
        ref_clean = cv2.GaussianBlur(ref_clean, (cr, cr), 0)
        
    sigma_ref_x = (meta["beam_spot_size_nm"] / 1.0) / 2.355
    sigma_ref_y = sigma_ref_x * meta["beam_astigmatism_ratio"]
    kr_x, kr_y = int(6 * sigma_ref_x + 1) | 1, int(6 * sigma_ref_y + 1) | 1
    ref_blurred = cv2.GaussianBlur(ref_clean, (kr_x, kr_y), sigmaX=sigma_ref_x, sigmaY=sigma_ref_y)

    np.random.seed(master_seed + 2) 
    ref_final = apply_acquisition_noise(ref_blurred, meta["dose_reference"], speckle_sigma=0.0, salt_pepper_prob=0.0)

    # SAVE TO DISK
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "generated_dataset_images" / "dram_40"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(search_final).save(str(OUTPUT_DIR / "search_10x.png"))
    Image.fromarray(ref_final).save(str(OUTPUT_DIR / "reference_100x.png"))

    ground_truth_path = OUTPUT_DIR / "ground_truth.json"
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    print("Initializing Custom Pattern Generator...")
    generate_pair(1)
    print(f"Dataset generation complete! Files saved in {Path(__file__).resolve().parent.parent / 'results' / 'dram_40'}")

if __name__ == "__main__":
    main()