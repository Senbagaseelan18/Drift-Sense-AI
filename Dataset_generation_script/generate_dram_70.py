import json
import numpy as np
import cv2
from PIL import Image
import os

from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_70"
# =========================================================================
# PHYSICS & ARTIFACT ENGINES (Balanced SEM Filters)
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
# LAYOUT & GEOMETRY ENGINE (SQUARE/CROSS ARRAY + EDGE-STRADDLING CIRCLES)
# =========================================================================

def draw_square_cross_circle_outline_layout(width_px, height_px, pixel_scale_nm, start_x_nm, start_y_nm, meta):
    BACKGROUND = 35.0
    img = np.full((height_px, width_px), BACKGROUND, dtype=np.float32)

    x_coords = np.arange(width_px, dtype=np.float32) * pixel_scale_nm + start_x_nm
    y_coords = np.arange(height_px, dtype=np.float32) * pixel_scale_nm + start_y_nm
    X, Y = np.meshgrid(x_coords, y_coords)

    mat_size = meta["mat_size_nm"]
    strip = meta["strip_width_nm"]
    scale = meta["feature_size_scale"]
    
    local_x = X % (mat_size + strip)
    local_y = Y % (mat_size + strip)

    in_mat = (local_x < mat_size) & (local_y < mat_size)
    peripheral_mask = ~in_mat
    img[peripheral_mask] = 20.0 # Dark scribe lines (The edges)

    # =========================================================================
    # SQUARE & CROSS ARRAY (Matching image_30d280.png)
    # =========================================================================
    pitch = 160.0 * scale
    cx = (X % pitch) - (pitch / 2.0)
    cy = (Y % pitch) - (pitch / 2.0)
    
    # 1. Protruding Dark Gray Crosses
    cross_vert = (np.abs(cx) < 12.0 * scale) & (np.abs(cy) < 70.0 * scale)
    cross_horz = (np.abs(cy) < 12.0 * scale) & (np.abs(cx) < 70.0 * scale)
    img[in_mat & (cross_vert | cross_horz)] = 95.0
    
    # 2. Bright Outer Squares
    outer_sq = (np.abs(cx) < 40.0 * scale) & (np.abs(cy) < 40.0 * scale)
    img[in_mat & outer_sq] = 210.0
    
    # 3. Dim Inner Cores
    inner_sq = (np.abs(cx) < 22.0 * scale) & (np.abs(cy) < 22.0 * scale)
    img[in_mat & inner_sq] = 160.0

    # =========================================================================
    # MULTIPLE GUARANTEED FIDUCIALS (CIRCLE OUTLINES ON THE EDGES)
    # =========================================================================
    for gx, gy in [(meta["gt_x"] * 10.0, meta["gt_y"] * 10.0), 
                   (meta["gt2_x"] * 10.0, meta["gt2_y"] * 10.0)]:
        
        fid_dx = X - gx
        fid_dy = Y - gy
        dist_sq = fid_dx**2 + fid_dy**2
        
        # Dark Isolation Pad centered right on the edge/scribe line
        pad_mask = (np.abs(fid_dx) < 120.0 * scale) & (np.abs(fid_dy) < 120.0 * scale)
        img[pad_mask] = 28.0
        
        # Hollow Circle Outline
        outer_r_sq = (60.0 * scale)**2
        inner_r_sq = (45.0 * scale)**2
        circle_mask = pad_mask & (dist_sq <= outer_r_sq) & (dist_sq >= inner_r_sq)
        img[circle_mask] = 255.0

    return img

def generate_pair(index):
    # FIXED SEED hardcoded to 88776
    master_seed = 88776
    np.random.seed(master_seed)
    
    meta = {
        "architecture": "square_cross_edge_circles",
        "pair_id": index,
        "seed": master_seed,
        
        "mat_size_nm": 2600.0,
        "strip_width_nm": 320.0,
        
        "feature_size_scale": 1.00,
        "linewidth_bias_nm": 0.00,
        "pattern_collapse_threshold_nm": 10.00,
        "corner_rounding_px": 0.00,
        
        # Balanced SEM Filters
        "beam_spot_size_nm": 4.50,              
        "beam_astigmatism_ratio": 1.05,         
        "barrel_distortion": 0.00,              
        "vignette_strength": 0.12,              
        "gamma": 1.00,                          
        "charging_streaks_per_100_rows": 6.00, 
        "charging_streak_intensity": 18.00,
        
        "dose_reference": 2500.00,              
        "dose_search": 200.00,                  
        "shear_amplitude_px": 1.50,             
        "drift_jitter_px": 0.50,                
        "speckle_noise_sigma": 0.05,            
        "salt_and_pepper_prob": 0.005           
    }

    # =========================================================================
    # EXTREME EDGE STRADDLING LOGIC (FORCING ONTO THE BOUNDARIES)
    # =========================================================================
    period = meta["mat_size_nm"] + meta["strip_width_nm"]
    
    # Target 1 (Primary Ground Truth) -> Forced onto a VERTICAL EDGE
    # Placed on the scribe line between Block (0,y) and Block (1,y)
    gt_x_nm = period - (meta["strip_width_nm"] / 2.0)
    gt_y_nm = period / 2.0  # Center of the block vertically
    
    meta["gt_x"] = round(gt_x_nm / 10.0, 2)
    meta["gt_y"] = round(gt_y_nm / 10.0, 2)
    meta["gt_box"] = [round(meta["gt_x"] - 50, 2), round(meta["gt_y"] - 50, 2), 100, 100]

    # Target 2 (Secondary Target) -> Forced onto a HORIZONTAL EDGE
    # Placed on the scribe line between Block (x,1) and Block (x,2)
    gt2_x_nm = period + (period / 2.0) # Center of the block horizontally
    gt2_y_nm = period * 2.0 - (meta["strip_width_nm"] / 2.0)
    
    meta["gt2_x"] = round(gt2_x_nm / 10.0, 2)
    meta["gt2_y"] = round(gt2_y_nm / 10.0, 2)

    print(f"Generating Dataset: Square/Cross Array & Edge-Straddling Circles (Seed: {master_seed})...")
    print(f"  Target 1 located on VERTICAL EDGE at X={meta['gt_x']}px")
    print(f"  Target 2 located on HORIZONTAL EDGE at Y={meta['gt2_y']}px")

    # =========================================================================
    # SEARCH IMAGE GENERATION
    # =========================================================================
    # Generate directly at 1000x1000 with 10 nm/px to avoid expensive 10000x10000 work
    search_clean_1nm = draw_square_cross_circle_outline_layout(
        width_px=1000, height_px=1000, pixel_scale_nm=10.0,
        start_x_nm=0.0, start_y_nm=0.0, meta=meta
    )
    
    if meta["corner_rounding_px"] > 0:
        cr = int(meta["corner_rounding_px"] * 2 + 1)
        search_clean_1nm = cv2.GaussianBlur(search_clean_1nm, (cr, cr), 0)

    # already produced at target size
    search_downsampled = search_clean_1nm
    
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
    # REFERENCE IMAGE GENERATION (Centered on Target 1 - Vertical Edge)
    # =========================================================================
    phys_cx = meta["gt_x"] * 10.0
    phys_cy = meta["gt_y"] * 10.0
    
    ref_clean = draw_square_cross_circle_outline_layout(
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
    output_dir = OUT
    os.makedirs(output_dir, exist_ok=True)
    
    Image.fromarray(search_final).save(str(OUT / "search_10x.png"))
    Image.fromarray(ref_final).save(str(OUT / "reference_100x.png"))

    with open(OUT / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    print("Initializing Custom Pattern Generator...")
    generate_pair(1)
    print("Dataset generation complete! Files saved in 'dataset_square_cross_edge_circles'.")

if __name__ == "__main__":
    main()