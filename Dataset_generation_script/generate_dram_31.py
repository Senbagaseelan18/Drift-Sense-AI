import json
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import os

def apply_sem_edge_brightening(img, strength=0.6):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
    brightened = cv2.addWeighted(img, 1.0, gradient, strength, 0.0)
    return np.clip(brightened, 0, 255)

def draw_staggered_dram_layout(width_px, height_px, pixel_scale_nm, start_x_nm, start_y_nm, meta):
    BACKGROUND = 50.0
    img = np.full((height_px, width_px), BACKGROUND, dtype=np.float32)

    x_coords = np.arange(width_px, dtype=np.float32) * pixel_scale_nm + start_x_nm
    y_coords = np.arange(height_px, dtype=np.float32) * pixel_scale_nm + start_y_nm
    X, Y = np.meshgrid(x_coords, y_coords)

    mat_size = meta["mat_size_nm"]
    strip = meta["strip_width_nm"]
    seed = meta["seed"]
    
    # Explicitly using np.int32 prevents massive RAM spikes
    block_x = (X // (mat_size + strip)).astype(np.int32)
    block_y = (Y // (mat_size + strip)).astype(np.int32)
    local_x = X % (mat_size + strip)
    local_y = Y % (mat_size + strip)

    in_mat = (local_x < mat_size) & (local_y < mat_size)
    peripheral_mask = ~in_mat

    img[peripheral_mask] = 40.0

    # =========================================================================
    # STAGGERED DRAM ARRAY
    # =========================================================================
    mat_hash = (block_x * 73856 + block_y * 19349 + seed) % 100
    
    pitch_mult = np.ones_like(X)
    pitch_mult[mat_hash < 20] = 0.8
    pitch_mult[(mat_hash >= 20) & (mat_hash < 50)] = 1.0
    pitch_mult[(mat_hash >= 50) & (mat_hash < 80)] = 1.25
    pitch_mult[mat_hash >= 80] = 1.5

    base_pitch = 60.0
    base_cd = 28.0 + meta.get("linewidth_bias_nm", 0.0)
    
    eff_pitch = base_pitch * pitch_mult
    eff_cd = base_cd * pitch_mult

    row_idx = (Y // eff_pitch).astype(np.int32)
    stagger_shift = (row_idx % 2) * (eff_pitch / 2.0)
    X_staggered = X + stagger_shift

    wl_mask = (Y % eff_pitch) < eff_cd
    img[in_mat & wl_mask] = 110.0

    gap_cx = (eff_pitch + eff_cd) / 2.0
    gap_cy = (eff_pitch + eff_cd) / 2.0
    
    dx = np.abs((X_staggered % eff_pitch) - gap_cx)
    dy = np.abs((Y % eff_pitch) - gap_cy)
    
    contact_mask = (dx <= eff_cd / 2.0) & (dy <= eff_cd / 1.5) 
    
    cell_x = (X_staggered // eff_pitch).astype(np.int32)
    cell_y = (Y // eff_pitch).astype(np.int32)
    defect_hash = (cell_x * 91231 + cell_y * 514229 + seed) % 10000
    yield_mask = defect_hash > (meta.get("missing_contact_prob", 0.05) * 10000)
    
    img[in_mat & contact_mask & yield_mask] = 200.0

    # =========================================================================
    # SINGLE GUARANTEED FIDUCIAL (AT GROUND TRUTH CENTER)
    # =========================================================================
    
    # Calculate the exact physical coordinate of the Ground Truth box
    gt_phys_x = meta["gt_x"] * 10.0
    gt_phys_y = meta["gt_y"] * 10.0
    
    # Choose either 0 (Cross) or 1 (Slash) based on the seed
    anchor_type = seed % 2 
    
    if anchor_type == 0:
        # Draw a Single Cross (+) Marker
        gt_cross_dx = np.abs(X - gt_phys_x)
        gt_cross_dy = np.abs(Y - gt_phys_y)
        
        img[(gt_cross_dx <= 150.0) & (gt_cross_dy <= 30.0)] = 240.0
        img[(gt_cross_dx <= 50.0) & (gt_cross_dy <= 120.0)] = 15.0
        
    else:
        # Draw a Single Diagonal Slash (/) Marker
        gt_slash_pad = (np.abs(X - gt_phys_x) < 200.0) & (np.abs(Y - gt_phys_y) < 150.0)
        img[gt_slash_pad] = 160.0
        
        gt_dist_to_slash = np.abs((X - gt_phys_x) + (Y - gt_phys_y)) / np.sqrt(2)
        gt_slash_line = gt_slash_pad & (gt_dist_to_slash < 15.0) & (np.abs(X - gt_phys_x) < 140.0)
        img[gt_slash_line] = 10.0

    return img

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

def apply_acquisition_noise(img, dose, detector_sigma):
    counts = (np.clip(img, 0, 255) / 255.0) * dose
    noisy_counts = np.random.poisson(counts).astype(np.float32)
    img = (noisy_counts / dose) * 255.0

    if detector_sigma > 0:
        img += np.random.normal(0, detector_sigma, img.shape)

    return np.clip(img, 0, 255).astype(np.uint8)

def generate_pair(index):
    """Generates a single synthetic image pair and metadata."""
    
    # HARDCODED SEED ensures the script generates the exact same data every run
    master_seed = 4242
    np.random.seed(master_seed)
    
    random_gt_x = np.random.uniform(50.0, 950.0)
    random_gt_y = np.random.uniform(50.0, 950.0)

    metadata = {
        "architecture": "dram_staggered",
        "pair_id": index,
        "gt_x": round(random_gt_x, 2),
        "gt_y": round(random_gt_y, 2),
        "gt_box": [round(random_gt_x - 50, 2), round(random_gt_y - 50, 2), 100, 100],
        "seed": master_seed,
        "beam_spot_size_nm": 5.0,
        "missing_contact_prob": 0.05,
        "dose_reference": 2000,
        "dose_search": 200,
        "shear_amplitude_px": 0.5,       
        "drift_jitter_px": 0.2,          
        "detector_noise_sigma_ref": 2.0,
        "detector_noise_sigma_search": 3.0, 
        "mat_size_nm": 2600.0,
        "strip_width_nm": 320.0,
        "linewidth_bias_nm": 0.0
    }

    print(f"Generating Deterministic Pair ID {index} (Seed: {master_seed})...")

    # SEARCH IMAGE
    search_clean_1nm = draw_staggered_dram_layout(
        width_px=1000, height_px=1000, pixel_scale_nm=10.0,
        start_x_nm=0.0, start_y_nm=0.0, meta=metadata
    )
    search_downsampled = cv2.resize(search_clean_1nm, (1000, 1000), interpolation=cv2.INTER_NEAREST)
    search_bright = apply_sem_edge_brightening(search_downsampled, strength=0.6)
    
    sigma_search = (metadata["beam_spot_size_nm"] / 10.0) / 2.355 
    k_search = int(6 * sigma_search + 1) | 1
    search_blurred = cv2.GaussianBlur(search_bright, (k_search, k_search), sigmaX=sigma_search, sigmaY=sigma_search)

    np.random.seed(master_seed + 1)
    search_drifted = apply_raster_artifacts(search_blurred, metadata["shear_amplitude_px"], metadata["drift_jitter_px"])
    search_final = apply_acquisition_noise(search_drifted, metadata["dose_search"], metadata["detector_noise_sigma_search"])

    # REFERENCE IMAGE
    phys_cx = metadata["gt_x"] * 10.0
    phys_cy = metadata["gt_y"] * 10.0
    
    ref_clean = draw_staggered_dram_layout(
        width_px=1000, height_px=1000, pixel_scale_nm=1.0,
        start_x_nm=phys_cx - 500.0, start_y_nm=phys_cy - 500.0, meta=metadata
    )
    ref_bright = apply_sem_edge_brightening(ref_clean, strength=0.6)
    
    sigma_ref = (metadata["beam_spot_size_nm"] / 1.0) / 2.355
    k_ref = int(6 * sigma_ref + 1) | 1
    ref_blurred = cv2.GaussianBlur(ref_bright, (k_ref, k_ref), sigmaX=sigma_ref, sigmaY=sigma_ref)

    np.random.seed(master_seed + 2) 
    ref_final = apply_acquisition_noise(ref_blurred, metadata["dose_reference"], metadata["detector_noise_sigma_ref"])

    # SAVE TO DISK
    OUTPUT_DIR = Path(__file__).resolve().parent.parent / "results" / "generated_dataset_images" / "dram_31"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    Image.fromarray(search_final).save(str(OUTPUT_DIR / "search_10x.png"))
    Image.fromarray(ref_final).save(str(OUTPUT_DIR / "reference_100x.png"))

    ground_truth_path = OUTPUT_DIR / "ground_truth.json"
    with open(ground_truth_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

def main():
    print("Initializing Single-Pair Deterministic Generator...")
    # Generate only a single dataset pair
    generate_pair(1)
    print(f"Dataset generation complete! Files saved in {Path(__file__).resolve().parent.parent / 'results' / 'dram_31'}")

if __name__ == "__main__":
    main()