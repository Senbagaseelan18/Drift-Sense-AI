import json
import numpy as np
import cv2
from PIL import Image
import os

from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results" / "generated_dataset_images" / "dram_74"
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

def apply_vignette(img, vignette_strength):
    if vignette_strength <= 0: return img
    h, w = img.shape
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    mask = 1.0 - np.clip(radius * vignette_strength, 0, 1)
    return (img * mask).astype(np.float32)

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

# NEW: Applies wildly different noise, dose, and contrast filters to each section
def apply_regional_artifacts(img, meta, is_search=True):
    h, w = img.shape
    out = img.copy()
    
    # 1 pixel = 10nm (after downsampling)
    period_px = int((meta["mat_size_nm"] + meta["strip_width_nm"]) / 10.0)
    
    for bx in range(3):
        for by in range(3):
            x_start = bx * period_px
            x_end = (bx + 1) * period_px
            y_start = by * period_px
            y_end = (by + 1) * period_px
            
            x_start, x_end = min(x_start, w), min(x_end, w)
            y_start, y_end = min(y_start, h), min(y_end, h)
            
            if x_start >= w or y_start >= h: continue
            
            region = out[y_start:y_end, x_start:x_end]
            
            # Seed based on block coordinates to ensure distinct but repeatable filters
            np.random.seed(meta["seed"] + bx * 7 + by * 13)
            
            if is_search:
                # SEARCH IMAGE: Extreme variations per section
                contrast = np.random.uniform(0.6, 1.4)
                brightness = np.random.uniform(-40, 40)
                local_dose = np.random.uniform(30.0, 300.0) # Varies from extremely noisy to clean
                speckle = np.random.uniform(0.0, 0.15)
            else:
                # REFERENCE IMAGE: Uniform, cleaner baseline for the ground truth
                contrast = 1.0
                brightness = 0.0
                local_dose = meta["dose_reference"]
                speckle = 0.0
            
            # Apply Brightness/Contrast
            region = np.clip(region * contrast + brightness, 0, 255)
            
            # Apply Poisson Dose Noise
            counts = (region / 255.0) * local_dose
            noisy_counts = np.random.poisson(counts).astype(np.float32)
            region = (noisy_counts / local_dose) * 255.0
            
            # Apply Speckle Noise
            if speckle > 0:
                speckle_noise = np.random.normal(1.0, speckle, region.shape)
                region *= speckle_noise
                
            out[y_start:y_end, x_start:x_end] = region
            
    return np.clip(out, 0, 255).astype(np.uint8)

# =========================================================================
# LAYOUT & GEOMETRY ENGINE (STAGGERED PINNED OVALS)
# =========================================================================

def draw_staggered_pinned_ovals_layout(width_px, height_px, pixel_scale_nm, start_x_nm, start_y_nm, meta):
    BACKGROUND = 22.0
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
    img[peripheral_mask] = 12.0 # Dark scribe lines separating sections

    # =========================================================================
    # STAGGERED PINNED OVALS ARRAY (Matching image_305d87.png)
    # =========================================================================
    pitch_x = 100.0 * scale
    pitch_y = 150.0 * scale
    
    # Grid 1
    cx1 = (X % pitch_x) - (pitch_x / 2.0)
    cy1 = (Y % (2 * pitch_y)) - pitch_y
    
    # Grid 2 (Staggered)
    cx2 = ((X - pitch_x/2.0) % pitch_x) - (pitch_x / 2.0)
    cy2 = ((Y - pitch_y) % (2 * pitch_y)) - pitch_y
    
    # 1. Vertical Pins (Dim tails extending from ovals)
    pin_w = 4.0 * scale
    pin_h = 45.0 * scale
    pin1 = (np.abs(cx1) < pin_w) & (np.abs(cy1) < pin_h)
    pin2 = (np.abs(cx2) < pin_w) & (np.abs(cy2) < pin_h)
    
    img[in_mat & (pin1 | pin2)] = 110.0
    
    # 2. Bright Ovals
    rx = 18.0 * scale
    ry = 32.0 * scale
    oval1 = ((cx1**2)/(rx**2) + (cy1**2)/(ry**2)) <= 1.0
    oval2 = ((cx2**2)/(rx**2) + (cy2**2)/(ry**2)) <= 1.0
    
    img[in_mat & (oval1 | oval2)] = 240.0

    return img

def generate_pair(index):
    # FIXED SEED hardcoded to 19283
    master_seed = 19283
    np.random.seed(master_seed)
    
    meta = {
        "architecture": "pinned_ovals_regional_filters",
        "pair_id": index,
        "seed": master_seed,
        
        "mat_size_nm": 2600.0,
        "strip_width_nm": 320.0,
        
        "feature_size_scale": 1.00,
        "linewidth_bias_nm": 0.00,
        "pattern_collapse_threshold_nm": 10.00,
        "corner_rounding_px": 0.00,
        
        # Base Global SEM Filters (Regional applied later)
        "beam_spot_size_nm": 4.50,              
        "beam_astigmatism_ratio": 1.02,         
        "barrel_distortion": 0.00,              
        "vignette_strength": 0.15,              
        "charging_streaks_per_100_rows": 10.00, 
        "charging_streak_intensity": 25.00,
        
        "dose_reference": 2500.00,              
        "shear_amplitude_px": 2.00,             
        "drift_jitter_px": 0.60,                
    }

    # =========================================================================
    # BOUNDARY STRADDLING LOGIC (MEETING AREA CROP)
    # =========================================================================
    # Place Ground Truth EXACTLY on the vertical scribe line between Section 0 and Section 1
    
    gt_x_nm = meta["mat_size_nm"] + (meta["strip_width_nm"] / 2.0)
    # Place Y somewhere safely inside the vertical height of the block
    gt_y_nm = meta["mat_size_nm"] / 2.0 
    
    meta["gt_x"] = round(gt_x_nm / 10.0, 2)
    meta["gt_y"] = round(gt_y_nm / 10.0, 2)
    meta["gt_box"] = [round(meta["gt_x"] - 50, 2), round(meta["gt_y"] - 50, 2), 100, 100]

    print(f"Generating Dataset: Regional Filters & Boundary Crop (Seed: {master_seed})...")
    print(f"  Reference Crop centered exactly at SECTION MEETING AREA: X={meta['gt_x']}px")

    # =========================================================================
    # SEARCH IMAGE GENERATION
    # =========================================================================
    # Generate directly at 1000x1000 with 10 nm/px to avoid expensive 10000x10000 work
    search_clean_1nm = draw_staggered_pinned_ovals_layout(
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
    search_vignette = apply_vignette(search_distorted, meta["vignette_strength"])
    
    # Global Raster & Charging Artifacts
    np.random.seed(master_seed + 1)
    search_charged = apply_charging_streaks(search_vignette, meta["charging_streaks_per_100_rows"], meta["charging_streak_intensity"])
    search_drifted = apply_raster_artifacts(search_charged, meta["shear_amplitude_px"], meta["drift_jitter_px"])
    
    # NEW: Apply Regional Dose and Contrast variations (Patchwork sections)
    search_final = apply_regional_artifacts(search_drifted, meta, is_search=True)

    # =========================================================================
    # REFERENCE IMAGE GENERATION (Cropped exactly over the boundary junction)
    # =========================================================================
    phys_cx = meta["gt_x"] * 10.0
    phys_cy = meta["gt_y"] * 10.0
    
    ref_clean = draw_staggered_pinned_ovals_layout(
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

    # Reference Image uses a clean baseline dose
    ref_final = apply_regional_artifacts(ref_blurred, meta, is_search=False)

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
    print("Dataset generation complete! Files saved in 'dataset_regional_filters_boundary'.")

if __name__ == "__main__":
    main()