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
# LAYOUT & GEOMETRY ENGINE (DIAGONAL FINS + DISTINGUISHING FILTER + "/" IN SQUARE)
# =========================================================================

def draw_diagonal_fin_layout(width_px, height_px, pixel_scale_nm, start_x_nm, start_y_nm, meta):
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
    # DIAGONAL FIN / PILL ARRAY (Matching image_9cd4de)
    # =========================================================================
    grid_pitch = 260.0 * scale
    grid_lines = ((X % grid_pitch) < 4.0) | ((Y % grid_pitch) < 4.0)
    img[in_mat & grid_lines] = 60.0

    pitch_d = 50.0 * scale
    diag_coord = (X - Y) % pitch_d
    trans_coord = (X + Y) % (120.0 * scale)

    fin_mask = (diag_coord < 25.0 * scale) & (trans_coord < (90.0 * scale))
    img[in_mat & fin_mask & ~grid_lines] = 180.0

    # Bright contact points on each structure
    contact_coord_x = X % (120.0 * scale)
    contact_coord_y = Y % pitch_d
    contact_mask = (np.abs(contact_coord_x - 60.0 * scale) < 8.0) & (np.abs(contact_coord_y - 12.0) < 6.0)
    img[in_mat & contact_mask] = 240.0

    # =========================================================================
    # DISTINGUISHING FILTER (Unique contrast/brightness per section block)
    # =========================================================================
    section_id = (block_x * 2 + block_y * 3) % 3
    
    mask_style_1 = in_mat & (section_id == 1)
    img[mask_style_1] = np.clip(img[mask_style_1] * 1.3 + 15.0, 0, 255)
    
    mask_style_2 = in_mat & (section_id == 2)
    img[mask_style_2] = np.clip(255.0 - (img[mask_style_2] * 0.7), 0, 255)

    # =========================================================================
    # SINGLE GUARANTEED FIDUCIAL ("/" INSIDE A SQUARE)
    # =========================================================================
    gt_phys_x = meta["gt_x"] * 10.0
    gt_phys_y = meta["gt_y"] * 10.0
    
    fid_dx = np.abs(X - gt_phys_x)
    fid_dy = np.abs(Y - gt_phys_y)
    
    # Clear isolation pad
    pad_mask = (fid_dx < 140.0) & (fid_dy < 140.0)
    img[pad_mask] = 30.0
    
    # Outer Square Boundary
    square_out = (fid_dx <= 100.0) & (fid_dy <= 100.0)
    square_in = (fid_dx < 80.0) & (fid_dy < 80.0)
    img[pad_mask & square_out & ~square_in] = 220.0
    
    # Inner background for the slash
    img[pad_mask & square_in] = 40.0
    
    # Diagonal slash "/" inside the square box
    dist_to_slash = np.abs((X - gt_phys_x) + (Y - gt_phys_y)) / np.sqrt(2)
    slash_mask = pad_mask & square_in & (dist_to_slash < 10.0) & (fid_dx < 60.0) & (fid_dy < 60.0)
    img[slash_mask] = 255.0

    return img

def generate_pair(index):
    # FIXED SEED hardcoded to 33445
    master_seed = 33445
    np.random.seed(master_seed)
    
    meta = {
        "architecture": "diagonal_fin_square_slash",
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
        "charging_streaks_per_100_rows": 5.00,
        "charging_streak_intensity": 25.00,
        
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

    print(f"Generating Deterministic Dataset with Diagonal Fins & Square Slash (Seed: {master_seed})...")

    # =========================================================================
    # SEARCH IMAGE GENERATION
    # =========================================================================
    search_clean_1nm = draw_diagonal_fin_layout(
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
    
    ref_clean = draw_diagonal_fin_layout(
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
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "generated_dataset_images" / "dram_32"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(search_final).save(str(OUTPUT_DIR / "search_10x.png"))
    Image.fromarray(ref_final).save(str(OUTPUT_DIR / "reference_100x.png"))

    ground_truth_path = OUTPUT_DIR / "ground_truth.json"
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    print("Initializing Custom Pattern Generator...")
    generate_pair(1)
    print(f"Dataset generation complete! Files saved in {Path(__file__).resolve().parent.parent / 'results' / 'dram_32'}")

if __name__ == "__main__":
    main()