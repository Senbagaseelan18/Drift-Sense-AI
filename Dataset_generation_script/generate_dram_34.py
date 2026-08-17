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
    """Applies Barrel (+k1) or Pincushion (-k1) distortion."""
    if abs(k1) < 0.001: return img
    h, w = img.shape
    # Approximate camera matrix for synthetic FOV
    K = np.array([[w/1.5, 0, w/2], 
                  [0, h/1.5, h/2], 
                  [0, 0, 1]], dtype=np.float32)
    D = np.array([k1, 0, 0, 0], dtype=np.float32)
    return cv2.undistort(img, K, D)

def apply_vignette_and_gamma(img, vignette_strength, gamma):
    """Simulates detector corner falloff and contrast curve."""
    h, w = img.shape
    X, Y = np.meshgrid(np.linspace(-1, 1, w), np.linspace(-1, 1, h))
    radius = np.sqrt(X**2 + Y**2)
    
    # Vignette mask
    mask = 1.0 - np.clip(radius * vignette_strength, 0, 1)
    img_v = img * mask
    
    # Gamma correction
    if abs(gamma - 1.0) > 0.01:
        img_v = 255.0 * ((img_v / 255.0) ** (1.0 / gamma))
        
    return np.clip(img_v, 0, 255).astype(np.float32)

def apply_charging_streaks(img, streaks_per_100, intensity):
    """Simulates dielectric surface charging (horizontal streaks)."""
    if streaks_per_100 <= 0 or intensity <= 0: return img
    h, w = img.shape
    out = img.copy()
    
    prob_per_row = streaks_per_100 / 100.0
    for row in range(h):
        if np.random.rand() < prob_per_row:
            # Generate a fading horizontal streak
            streak_len = np.random.randint(w//4, w)
            start_x = np.random.randint(0, w - streak_len)
            gradient = np.linspace(intensity, 0, streak_len)
            out[row, start_x:start_x+streak_len] += gradient
            
    return np.clip(out, 0, 255)

def apply_raster_artifacts(img, shear_amp, jitter_std):
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
    # 1. Base Poisson Noise (Shot Noise)
    counts = (np.clip(img, 0, 255) / 255.0) * dose
    noisy_counts = np.random.poisson(counts).astype(np.float32)
    img_noise = (noisy_counts / dose) * 255.0

    # 2. Multiplicative Speckle
    if speckle_sigma > 0:
        speckle = np.random.normal(1.0, speckle_sigma, img_noise.shape)
        img_noise *= speckle

    # 3. Salt and Pepper
    if salt_pepper_prob > 0:
        rand_matrix = np.random.rand(*img_noise.shape)
        img_noise[rand_matrix < (salt_pepper_prob / 2.0)] = 0.0
        img_noise[(rand_matrix >= (salt_pepper_prob / 2.0)) & (rand_matrix < salt_pepper_prob)] = 255.0

    return np.clip(img_noise, 0, 255).astype(np.uint8)

# =========================================================================
# LAYOUT & GEOMETRY ENGINE
# =========================================================================

def draw_crossbar_layout(width_px, height_px, pixel_scale_nm, start_x_nm, start_y_nm, meta):
    BACKGROUND = 40.0
    img = np.full((height_px, width_px), BACKGROUND, dtype=np.float32)

    x_coords = np.arange(width_px, dtype=np.float32) * pixel_scale_nm + start_x_nm
    y_coords = np.arange(height_px, dtype=np.float32) * pixel_scale_nm + start_y_nm
    X, Y = np.meshgrid(x_coords, y_coords)

    mat_size = meta["mat_size_nm"]
    strip = meta["strip_width_nm"]
    scale = meta["feature_size_scale"]
    bias = meta["linewidth_bias_nm"]
    collapse_thresh = meta["pattern_collapse_threshold_nm"]
    
    # Downcast to int32 for RAM safety
    block_x = (X // (mat_size + strip)).astype(np.int32)
    block_y = (Y // (mat_size + strip)).astype(np.int32)
    local_x = X % (mat_size + strip)
    local_y = Y % (mat_size + strip)

    in_mat = (local_x < mat_size) & (local_y < mat_size)
    peripheral_mask = ~in_mat
    img[peripheral_mask] = 30.0 # Darker scribe lines

    # =========================================================================
    # ORTHOGONAL CROSSBAR / VIA MATRIX
    # =========================================================================
    pitch = 100.0 * scale
    
    # Faint underlying M1/M2 traces
    trace_cd = (20.0 * scale) + bias
    if trace_cd > collapse_thresh:
        vert_trace_mask = (X % pitch) < trace_cd
        horz_trace_mask = (Y % pitch) < trace_cd
        img[in_mat & (vert_trace_mask | horz_trace_mask)] = 75.0

    # Bright Circular Top-Vias
    via_cd = (45.0 * scale) + bias
    if via_cd > collapse_thresh:
        via_radius = via_cd / 2.0
        dx = np.abs((X % pitch) - (pitch / 2.0))
        dy = np.abs((Y % pitch) - (pitch / 2.0))
        dist_sq = dx**2 + dy**2
        via_mask = dist_sq < (via_radius**2)
        
        # 100% density - perfectly regular grid
        img[in_mat & via_mask] = 230.0

    # =========================================================================
    # COMPLEX ALIGNMENT FIDUCIAL (AT GROUND TRUTH CENTER)
    # =========================================================================
    gt_phys_x = meta["gt_x"] * 10.0
    gt_phys_y = meta["gt_y"] * 10.0
    
    fid_dx = np.abs(X - gt_phys_x)
    fid_dy = np.abs(Y - gt_phys_y)
    
    # 1. Outer Bounding Box (Thick outline)
    box_outer = (fid_dx <= 180) & (fid_dy <= 180)
    box_inner = (fid_dx <= 140) & (fid_dy <= 140)
    img[box_outer & ~box_inner] = 160.0
    
    # 2. Dark inner background to make the cross pop
    img[box_inner] = 50.0
    
    # 3. Inner Crosshair
    cross_mask = box_inner & ((fid_dx <= 20) | (fid_dy <= 20))
    img[cross_mask] = 160.0
    
    # 4. Quincunx Dot Pattern (Center + 4 Corners)
    # Equation for 5 circles
    c_dist_sq = fid_dx**2 + fid_dy**2
    tl_dist_sq = (X - (gt_phys_x - 90))**2 + (Y - (gt_phys_y - 90))**2
    tr_dist_sq = (X - (gt_phys_x + 90))**2 + (Y - (gt_phys_y - 90))**2
    bl_dist_sq = (X - (gt_phys_x - 90))**2 + (Y - (gt_phys_y + 90))**2
    br_dist_sq = (X - (gt_phys_x + 90))**2 + (Y - (gt_phys_y + 90))**2
    
    dot_radius_sq = 18**2
    dots_mask = (c_dist_sq < dot_radius_sq) | \
                (tl_dist_sq < dot_radius_sq) | \
                (tr_dist_sq < dot_radius_sq) | \
                (bl_dist_sq < dot_radius_sq) | \
                (br_dist_sq < dot_radius_sq)
                
    # Center dot inner core (ultra bright)
    core_mask = (c_dist_sq < 8**2)
    
    img[dots_mask & box_inner] = 200.0
    img[core_mask] = 255.0

    return img

def generate_pair(index):
    # HARDCODED DETERMINISTIC SEED
    master_seed = 4242
    np.random.seed(master_seed)
    
    # Build metadata from UI parameters
    meta = {
        "architecture": "crossbar_array_v1",
        "pair_id": index,
        "seed": master_seed,
        
        # Die layout
        "mat_size_nm": 2600.0,
        "strip_width_nm": 320.0,
        "p_straddle_boundary": 0.35,
        
        # Geometry & Optics
        "feature_size_scale": 1.0,
        "linewidth_bias_nm": 0.0,
        "pattern_collapse_threshold_nm": 10.0,
        "corner_rounding_px": 2.0,
        
        # SEM Physics
        "beam_spot_size_nm": 8.0,
        "beam_astigmatism_ratio": 1.1,
        "barrel_distortion": 0.05,
        "vignette_strength": 0.3,
        "gamma": 0.85,
        "charging_streaks_per_100_rows": 5.0,
        "charging_streak_intensity": 35.0,
        
        # Acquisition
        "dose_reference": 2000.0,
        "dose_search": 150.0,
        "shear_amplitude_px": 1.5,       
        "drift_jitter_px": 0.5,          
        "speckle_noise_sigma": 0.08,
        "salt_and_pepper_prob": 0.002
    }

    # Custom Ground Truth Placement (Boundary Straddling Logic)
    if np.random.rand() < meta["p_straddle_boundary"]:
        # Force coordinate onto a vertical or horizontal trench boundary
        period = meta["mat_size_nm"] + meta["strip_width_nm"]
        block = np.random.randint(1, 4)
        if np.random.rand() > 0.5:
            # Vertical trench
            gt_x_nm = (block * period) - (meta["strip_width_nm"] / 2.0)
            gt_y_nm = np.random.uniform(500.0, 9500.0)
        else:
            # Horizontal trench
            gt_x_nm = np.random.uniform(500.0, 9500.0)
            gt_y_nm = (block * period) - (meta["strip_width_nm"] / 2.0)
            
        random_gt_x = gt_x_nm / 10.0
        random_gt_y = gt_y_nm / 10.0
    else:
        # Standard random placement
        random_gt_x = np.random.uniform(50.0, 950.0)
        random_gt_y = np.random.uniform(50.0, 950.0)

    meta["gt_x"] = round(random_gt_x, 2)
    meta["gt_y"] = round(random_gt_y, 2)
    meta["gt_box"] = [round(random_gt_x - 50, 2), round(random_gt_y - 50, 2), 100, 100]

    print(f"Generating Deterministic Metrology Dataset (Seed: {master_seed})...")

    # =========================================================================
    # SEARCH IMAGE GENERATION (Low-Res, High Distortion, Wide Field)
    # =========================================================================
    search_clean_1nm = draw_crossbar_layout(
        width_px=1000, height_px=1000, pixel_scale_nm=10.0,
        start_x_nm=0.0, start_y_nm=0.0, meta=meta
    )
    
    # Corner Rounding via high-res Gaussian before downsampling
    if meta["corner_rounding_px"] > 0:
        cr = int(meta["corner_rounding_px"] * 2 + 1)
        search_clean_1nm = cv2.GaussianBlur(search_clean_1nm, (cr, cr), 0)

    search_downsampled = cv2.resize(search_clean_1nm, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    
    # Optics: Astigmatic Beam Blur
    sigma_x = (meta["beam_spot_size_nm"] / 10.0) / 2.355 
    sigma_y = sigma_x * meta["beam_astigmatism_ratio"]
    k_x, k_y = int(6 * sigma_x + 1) | 1, int(6 * sigma_y + 1) | 1
    search_blurred = cv2.GaussianBlur(search_downsampled, (k_x, k_y), sigmaX=sigma_x, sigmaY=sigma_y)

    # Optics: Camera Distortion, Vignette, Gamma
    search_distorted = apply_lens_distortion(search_blurred, meta["barrel_distortion"])
    search_vignette = apply_vignette_and_gamma(search_distorted, meta["vignette_strength"], meta["gamma"])

    # Acquisition: Charging, Drift, and Noise
    np.random.seed(master_seed + 1)
    search_charged = apply_charging_streaks(search_vignette, meta["charging_streaks_per_100_rows"], meta["charging_streak_intensity"])
    search_drifted = apply_raster_artifacts(search_charged, meta["shear_amplitude_px"], meta["drift_jitter_px"])
    search_final = apply_acquisition_noise(search_drifted, meta["dose_search"], meta["speckle_noise_sigma"], meta["salt_and_pepper_prob"])

    # =========================================================================
    # REFERENCE IMAGE GENERATION (High-Res, Pristine, Focused)
    # =========================================================================
    phys_cx = meta["gt_x"] * 10.0
    phys_cy = meta["gt_y"] * 10.0
    
    ref_clean = draw_crossbar_layout(
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
    # Reference images typically don't have severe drift/charging due to higher integration times
    ref_final = apply_acquisition_noise(ref_blurred, meta["dose_reference"], speckle_sigma=0.01, salt_pepper_prob=0.0)

    # SAVE TO DISK
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "generated_dataset_images" / "dram_34"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(search_final).save(str(OUTPUT_DIR / "search_10x.png"))
    Image.fromarray(ref_final).save(str(OUTPUT_DIR / "reference_100x.png"))

    ground_truth_path = OUTPUT_DIR / "ground_truth.json"
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    print("Initializing Metrology Array Generator...")
    # Generate exactly one pair, perfectly deterministic
    generate_pair(1)
    print(f"Dataset generation complete! Files saved in {Path(__file__).resolve().parent.parent / 'results' / 'dram_34'}")

if __name__ == "__main__":
    main()