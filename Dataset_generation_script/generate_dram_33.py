import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import os

# =========================================================================
# CLEAN ACQUISITION ENGINE (Fixed to keep proper 0-255 range)
# =========================================================================

def apply_clean_acquisition(img, dose):
    # Keep original pixel intensities cleanly without blowing out the range
    return np.clip(img, 0, 255).astype(np.uint8)

# =========================================================================
# LAYOUT & GEOMETRY ENGINE (16-MAT DIAGONAL PILS + "/" IN SQUARE)
# =========================================================================

def draw_clean_diagonal_fin_layout(width_px, height_px, pixel_scale_nm, start_x_nm, start_y_nm, meta):
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
    # 16-MAT DIAGONAL FIN / PILL ARRAY (Clean & Uniform)
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
    # FIXED SEED hardcoded to 55667
    master_seed = 55667
    np.random.seed(master_seed)
    
    meta = {
        "architecture": "clean_16mat_diagonal_square_slash",
        "pair_id": index,
        "seed": master_seed,
        
        "mat_size_nm": 2600.0,
        "strip_width_nm": 320.0,
        "p_straddle_boundary": 0.35,
        
        "feature_size_scale": 1.00,
        "linewidth_bias_nm": 0.00,
        "pattern_collapse_threshold_nm": 10.00,
        "corner_rounding_px": 0.00,
        
        "beam_spot_size_nm": 3.00,
        "dose_reference": 5000.00,
        "dose_search": 5000.00
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

    print(f"Generating Clean Unfiltered Dataset (Seed: {master_seed})...")

    # =========================================================================
    # SEARCH IMAGE GENERATION
    # =========================================================================
    search_clean_1nm = draw_clean_diagonal_fin_layout(
        width_px=1000, height_px=1000, pixel_scale_nm=10.0,
        start_x_nm=0.0, start_y_nm=0.0, meta=meta
    )
    search_downsampled = cv2.resize(search_clean_1nm, (1000, 1000), interpolation=cv2.INTER_AREA)
    
    sigma_search = (meta["beam_spot_size_nm"] / 10.0) / 2.355 
    k_search = int(6 * sigma_search + 1) | 1
    search_blurred = cv2.GaussianBlur(search_downsampled, (k_search, k_search), sigmaX=sigma_search, sigmaY=sigma_search)
    search_final = apply_clean_acquisition(search_blurred, meta["dose_search"])

    # =========================================================================
    # REFERENCE IMAGE GENERATION
    # =========================================================================
    phys_cx = meta["gt_x"] * 10.0
    phys_cy = meta["gt_y"] * 10.0
    
    ref_clean = draw_clean_diagonal_fin_layout(
        width_px=1000, height_px=1000, pixel_scale_nm=1.0,
        start_x_nm=phys_cx - 500.0, start_y_nm=phys_cy - 500.0, meta=meta
    )
    
    sigma_ref = (meta["beam_spot_size_nm"] / 1.0) / 2.355
    k_ref = int(6 * sigma_ref + 1) | 1
    ref_blurred = cv2.GaussianBlur(ref_clean, (k_ref, k_ref), sigmaX=sigma_ref, sigmaY=sigma_ref)
    ref_final = apply_clean_acquisition(ref_blurred, meta["dose_reference"])

    # SAVE TO DISK
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "generated_dataset_images" / "dram_33"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(search_final).save(str(OUTPUT_DIR / "search_10x.png"))
    Image.fromarray(ref_final).save(str(OUTPUT_DIR / "reference_100x.png"))

    ground_truth_path = OUTPUT_DIR / "ground_truth.json"
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

def main():
    print("Initializing Clean High-Quality Generator...")
    generate_pair(1)
    print(f"Dataset generation complete! Files saved in {Path(__file__).resolve().parent.parent / 'results' / 'dram_33'}")

if __name__ == "__main__":
    main()