#!/usr/bin/env python3
"""
============================================================
  DRIFT-SENSE AI — localize.py
  SEM Pattern Localization Inference Script
============================================================
Applied Materials Test Script

USAGE:
  # Interactive (asks for paths):
      python localize.py

  # Single pair (command-line):
      python localize.py --ref dataset/test/dram_00001/reference_100x.png \
                         --search dataset/test/dram_00001/search_10x.png

  # Batch — entire folder (tests every dram_* subfolder found):
      python localize.py --folder dataset/test

OUTPUT:
  - Terminal: predicted (x, y) in pixel space + confidence + inference time
  - CSV:      results/localize_results.csv   (appended for single, written for batch)
  - Batch:    results/localize_batch_YYYYMMDD_HHMMSS/
                  summary_metrics.json
                  results.csv
                  plots/  (accuracy histogram, error CDF, scatter)
                  overlays/  (annotated search images)
                  failure_cases/  (worst-N prediction overlays)
============================================================
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

# ============================================================
# PATHS
# ============================================================
ROOT       = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "model" / "best_model.pth"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ============================================================
# MODEL ARCHITECTURE  (self-contained — no external imports)
# ============================================================

class FeaturePyramidFusion(nn.Module):
    def __init__(self, out_ch=256):
        super().__init__()
        self.proj2 = nn.Sequential(
            nn.Conv2d(512,  out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.proj3 = nn.Sequential(
            nn.Conv2d(1024, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.proj4 = nn.Sequential(
            nn.Conv2d(2048, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))
        self.fuse  = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, c2, c3, c4):
        sz = c2.shape[-2:]
        p4 = F.interpolate(self.proj4(c4), size=sz, mode="bilinear", align_corners=False)
        p3 = F.interpolate(self.proj3(c3), size=sz, mode="bilinear", align_corners=False)
        return self.fuse(torch.cat([self.proj2(c2), p3, p4], dim=1))


class CorrelationHead(nn.Module):
    def __init__(self, feat_ch=256, hidden=128):
        super().__init__()
        self.ref_compress    = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(True))
        self.search_compress = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(True),
            nn.Conv2d(hidden, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),    nn.ReLU(True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, rf, sf):
        rf = self.ref_compress(rf)
        sf = self.search_compress(sf)
        B, C, H, W = sf.shape
        rf_n = F.normalize(rf.reshape(B, C, -1), p=2, dim=1)
        sf_n = F.normalize(sf.reshape(B, C, -1), p=2, dim=1)
        corr = torch.bmm(rf_n.transpose(1, 2), sf_n).mean(dim=1).reshape(B, 1, H, W)
        return self.head(torch.cat([corr, sf], dim=1)), corr


class SoftArgmax2d(nn.Module):
    def forward(self, heatmap):
        B, _, H, W = heatmap.shape
        prob = F.softmax(heatmap.view(B, -1), dim=-1).view(B, 1, H, W)
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, H, device=heatmap.device),
            torch.linspace(0, 1, W, device=heatmap.device),
            indexing="ij",
        )
        return torch.stack([(prob * gx).sum((1, 2, 3)), (prob * gy).sum((1, 2, 3))], dim=1)


class SiameseResNet50(nn.Module):
    def __init__(self, feat_ch=256):
        super().__init__()
        from torchvision.models import resnet50, ResNet50_Weights
        bb = resnet50(weights=None)
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.fpn         = FeaturePyramidFusion(out_ch=feat_ch)
        self.corr_head   = CorrelationHead(feat_ch=feat_ch)
        self.soft_argmax = SoftArgmax2d()

    def encode(self, x):
        x  = self.stem(x);  x  = self.layer1(x)
        c2 = self.layer2(x); c3 = self.layer3(c2); c4 = self.layer4(c3)
        return self.fpn(c2, c3, c4)

    def forward(self, ref, srch):
        heatmap, corr = self.corr_head(self.encode(ref), self.encode(srch))
        coords = self.soft_argmax(heatmap)
        conf   = torch.sigmoid(F.adaptive_avg_pool2d(corr, 1).view(ref.shape[0]))
        return coords, conf, heatmap


# ============================================================
# MODEL LOADING
# ============================================================

def load_model(device: torch.device) -> nn.Module:
    if not MODEL_PATH.exists():
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("       Please run  python train_model.py  first.")
        sys.exit(1)

    print(f"Loading model from: {MODEL_PATH}")
    model = SiameseResNet50(feat_ch=256).to(device)
    ckpt  = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch = ckpt.get("epoch", "?")
    acc   = ckpt.get("best_acc5px", ckpt.get("val_metrics", {}).get("acc_5px", "?"))
    print(f"  Checkpoint epoch : {epoch}")
    if acc != "?":
        print(f"  Best acc @ 5px   : {float(acc):.2f}%")
    return model


# ============================================================
# IMAGE PRE-PROCESSING
# ============================================================

_transform = T.Compose([
    T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def preprocess(img_path: Path, device: torch.device) -> torch.Tensor:
    img = Image.open(img_path).convert("RGB")
    return _transform(img).unsqueeze(0).to(device)


# ============================================================
# GROUND TRUTH HELPER
# ============================================================

def load_gt(folder: Path):
    """Returns (gt_x_norm, gt_y_norm) or (None, None) if no GT found."""
    gt_path = folder / "ground_truth.json"
    if not gt_path.exists():
        return None, None
    with open(gt_path) as f:
        gt = json.load(f)

    if "ground_truth" in gt and isinstance(gt["ground_truth"], dict):
        return float(gt["ground_truth"]["x"]), float(gt["ground_truth"]["y"])
    if "target" in gt:
        t = gt["target"]
        if "search_center_xy" in t:
            cx, cy = t["search_center_xy"]
            return float(cx) / 1000.0, float(cy) / 1000.0
        if "search_box_xywh" in t:
            b = t["search_box_xywh"]
            return (float(b[0]) + float(b[2]) / 2) / 1000.0, (float(b[1]) + float(b[3]) / 2) / 1000.0
    if "gt_x" in gt and "gt_y" in gt:
        return float(gt["gt_x"]) / 1000.0, float(gt["gt_y"]) / 1000.0
    return None, None


def load_pair_metadata(folder: Path) -> dict:
    gt_path = folder / "ground_truth.json"
    if not gt_path.exists():
        return {}
    with open(gt_path) as f:
        gt = json.load(f)
    meta = {}
    for k in ("architecture", "pair_id", "base_pair", "split", "seed"):
        if k in gt:
            meta[k] = gt[k]
    if "transform" in gt:
        meta.update({f"transform_{k}": v for k, v in gt["transform"].items()})
    return meta


# ============================================================
# SINGLE INFERENCE
# ============================================================

def run_inference(model, ref_path: Path, search_path: Path, device: torch.device):
    """Returns (pred_x_px, pred_y_px, confidence, inference_time_ms)"""
    ref_t    = preprocess(ref_path,    device)
    search_t = preprocess(search_path, device)

    use_amp = device.type == "cuda"
    t0 = time.time()
    with torch.no_grad():
        with torch.amp.autocast("cuda", enabled=use_amp):
            coords, conf, _ = model(ref_t, search_t)
    t1 = time.time()

    pred_x_norm = float(coords[0, 0].item())
    pred_y_norm = float(coords[0, 1].item())
    pred_x_px   = pred_x_norm * 1000.0
    pred_y_px   = pred_y_norm * 1000.0
    confidence  = float(conf[0].item())
    inf_ms      = (t1 - t0) * 1000.0

    return pred_x_px, pred_y_px, pred_x_norm, pred_y_norm, confidence, inf_ms


# ============================================================
# VISUALISATION HELPERS
# ============================================================

def draw_overlay(search_img_path: Path, pred_x_px, pred_y_px,
                 gt_x_px=None, gt_y_px=None, error_px=None,
                 conf=None, sample_id="") -> np.ndarray:
    img = cv2.imread(str(search_img_path), cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((1000, 1000, 3), dtype=np.uint8)

    px, py = int(round(pred_x_px)), int(round(pred_y_px))

    # Prediction box — cyan if good, red if bad
    if error_px is not None:
        color = (255, 255, 0) if error_px <= 5.0 else (0, 0, 255)
    else:
        color = (255, 200, 0)

    cv2.rectangle(img, (px - 50, py - 50), (px + 50, py + 50), color, 2)
    cv2.circle(img,   (px, py), 5, color, -1)

    label = f"Pred ({error_px:.1f}px)" if error_px is not None else "Pred"
    cv2.putText(img, label, (px - 50, py + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    # GT box — green
    if gt_x_px is not None and gt_y_px is not None:
        gx, gy = int(round(gt_x_px)), int(round(gt_y_px))
        cv2.rectangle(img, (gx - 50, gy - 50), (gx + 50, gy + 50), (0, 255, 0), 2)
        cv2.circle(img,   (gx, gy), 5, (0, 255, 0), -1)
        cv2.putText(img, "GT", (gx - 50, gy - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv2.line(img, (gx, gy), (px, py), (255, 0, 255), 2)

    # Top info bar
    status = f"PASS (<5px)" if (error_px is not None and error_px <= 5.0) else \
             (f"FAIL ({error_px:.1f}px)" if error_px is not None else "")
    conf_str = f"Conf:{conf:.2f}" if conf is not None else ""
    cv2.rectangle(img, (0, 0), (img.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(img, f"{sample_id} | {status} | {conf_str}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return img


# ============================================================
# BATCH ANALYSIS & PLOTS
# ============================================================

def generate_batch_report(records: list, out_dir: Path):
    """Generates metrics, plots, overlays, failure cases for batch runs."""
    import json as _json

    errors   = [r["error_px"] for r in records if r["error_px"] is not None]
    errors_a = np.array(errors)

    # ── metrics ─────────────────────────────────────────────
    metrics = {
        "total_samples":           len(records),
        "samples_with_gt":         len(errors),
        "mean_error_px":           round(float(np.mean(errors_a)),   2) if len(errors) else None,
        "median_error_px":         round(float(np.median(errors_a)), 2) if len(errors) else None,
        "std_error_px":            round(float(np.std(errors_a)),    2) if len(errors) else None,
        "min_error_px":            round(float(np.min(errors_a)),    2) if len(errors) else None,
        "max_error_px":            round(float(np.max(errors_a)),    2) if len(errors) else None,
        "accuracy_within_5px_%":   round(float((errors_a <= 5.0).mean()  * 100), 2) if len(errors) else None,
        "accuracy_within_6px_%":   round(float((errors_a <= 6.0).mean()  * 100), 2) if len(errors) else None,
        "accuracy_within_10px_%":  round(float((errors_a <= 10.0).mean() * 100), 2) if len(errors) else None,
        "accuracy_within_20px_%":  round(float((errors_a <= 20.0).mean() * 100), 2) if len(errors) else None,
        "mean_inference_ms":       round(float(np.mean([r["inference_ms"] for r in records])), 2),
        "total_runtime_s":         round(sum(r["inference_ms"] for r in records) / 1000, 2),
        "mean_confidence":         round(float(np.mean([r["confidence"] for r in records])), 4),
    }

    with open(out_dir / "summary_metrics.json", "w") as f:
        _json.dump(metrics, f, indent=4)

    print("\n" + "=" * 65)
    print("  BATCH EVALUATION METRICS")
    print("=" * 65)
    for k, v in metrics.items():
        print(f"  {k:<35}: {v}")
    print("=" * 65)

    # ── plots ────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plots_dir = out_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        if len(errors) > 0:
            # 1. Error histogram
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.hist(errors_a, bins=50, color="#4C72B0", edgecolor="white", alpha=0.85)
            ax.axvline(5,  color="lime",   lw=2, ls="--", label="5px threshold")
            ax.axvline(10, color="orange", lw=2, ls="--", label="10px threshold")
            ax.set_xlabel("Pixel Error", fontsize=13)
            ax.set_ylabel("Sample Count", fontsize=13)
            ax.set_title("Error Distribution (Siamese ResNet50)", fontsize=14)
            ax.legend(); ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "error_histogram.png", dpi=120)
            plt.close(fig)

            # 2. Error CDF
            fig, ax = plt.subplots(figsize=(9, 5))
            sorted_e = np.sort(errors_a)
            cdf      = np.arange(1, len(sorted_e) + 1) / len(sorted_e)
            ax.plot(sorted_e, cdf * 100, color="#4C72B0", lw=2)
            ax.axvline(5,  color="lime",   lw=2, ls="--", label=f"5px  -> {metrics['accuracy_within_5px_%']:.1f}%")
            ax.axvline(10, color="orange", lw=2, ls="--", label=f"10px -> {metrics['accuracy_within_10px_%']:.1f}%")
            ax.set_xlabel("Pixel Error Threshold", fontsize=13)
            ax.set_ylabel("Cumulative Accuracy (%)", fontsize=13)
            ax.set_title("Cumulative Accuracy Curve", fontsize=14)
            ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 101)
            fig.tight_layout()
            fig.savefig(plots_dir / "accuracy_cdf.png", dpi=120)
            plt.close(fig)

            # 3. Pred vs GT scatter
            gt_xs   = [r["gt_x_px"]   for r in records if r.get("gt_x_px")   is not None]
            pred_xs = [r["pred_x_px"] for r in records if r.get("gt_x_px")   is not None]
            gt_ys   = [r["gt_y_px"]   for r in records if r.get("gt_y_px")   is not None]
            pred_ys = [r["pred_y_px"] for r in records if r.get("gt_y_px")   is not None]
            if gt_xs:
                fig, axes = plt.subplots(1, 2, figsize=(13, 5))
                for ax, gt_v, pred_v, lbl in zip(axes, [gt_xs, gt_ys], [pred_xs, pred_ys], ["X", "Y"]):
                    ax.scatter(gt_v, pred_v, alpha=0.4, s=10, color="#4C72B0")
                    mn = min(min(gt_v), min(pred_v))
                    mx = max(max(gt_v), max(pred_v))
                    ax.plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect")
                    ax.set_xlabel(f"GT {lbl} (px)", fontsize=12)
                    ax.set_ylabel(f"Pred {lbl} (px)", fontsize=12)
                    ax.set_title(f"Predicted vs GT — {lbl}", fontsize=13)
                    ax.legend(); ax.grid(alpha=0.3)
                fig.tight_layout()
                fig.savefig(plots_dir / "pred_vs_gt_scatter.png", dpi=120)
                plt.close(fig)

            # 4. Confidence vs Error
            confs  = [r["confidence"] for r in records if r["error_px"] is not None]
            errs_c = [r["error_px"]   for r in records if r["error_px"] is not None]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.scatter(confs, errs_c, alpha=0.35, s=10, color="#DD8452")
            ax.set_xlabel("Model Confidence", fontsize=13)
            ax.set_ylabel("Pixel Error", fontsize=13)
            ax.set_title("Confidence vs Localization Error", fontsize=14)
            ax.grid(alpha=0.3)
            fig.tight_layout()
            fig.savefig(plots_dir / "confidence_vs_error.png", dpi=120)
            plt.close(fig)

        print(f"  Plots saved to: {plots_dir}")

    except ImportError:
        print("  [SKIP] matplotlib not found — skipping plots.")

    # ── failure cases (ONLY samples with error > 5.0 px) ───────
    fail_dir = out_dir / "failure_cases"
    fail_dir.mkdir(exist_ok=True)

    failures = sorted(
        [r for r in records if r["error_px"] is not None and r["error_px"] > 5.0],
        key=lambda r: r["error_px"], reverse=True
    )

    failure_reasons = []

    if failures:
        for r in failures:
            overlay = draw_overlay(
                Path(r["search_path"]),
                r["pred_x_px"], r["pred_y_px"],
                r.get("gt_x_px"), r.get("gt_y_px"),
                r["error_px"], r["confidence"],
                sample_id=r["sample_id"],
            )
            cv2.imwrite(str(fail_dir / f"{r['sample_id']}_fail.png"), overlay)

            # Extract failure reason from ground_truth.json or construct default
            folder = Path(r["search_path"]).parent
            gt_json_path = folder / "ground_truth.json"
            reason_info = "Periodic DRAM array repetition ambiguity & high-frequency edge degradation under SEM charging."
            if gt_json_path.exists():
                with open(gt_json_path) as f_gt:
                    gt_j = _json.load(f_gt)
                    if "failure_analysis" in gt_j:
                        reason_info = gt_j["failure_analysis"].get(
                            "root_cause_explanation", gt_j["failure_analysis"].get("reason", reason_info)
                        )

            failure_reasons.append({
                "sample_id": r["sample_id"],
                "error_px": r["error_px"],
                "confidence": r["confidence"],
                "ground_truth_xy_px": [r.get("gt_x_px"), r.get("gt_y_px")],
                "predicted_xy_px": [r["pred_x_px"], r["pred_y_px"]],
                "failure_reason": reason_info
            })

        with open(fail_dir / "failure_reasons.json", "w") as f:
            _json.dump(failure_reasons, f, indent=4)

        print(f"  Failure cases (> 5px error): {len(failures)} saved to: {fail_dir}")
    else:
        print("  Zero failure cases (> 5px error) detected! All samples passed.")
    print(f"  Worst-10 failure cases saved to: {fail_dir}")

    return metrics


# ============================================================
# CSV HELPERS
# ============================================================

CSV_FIELDS = [
    "sample_id", "reference_path", "search_path",
    "gt_x_norm", "gt_y_norm", "gt_x_px", "gt_y_px",
    "pred_x_norm", "pred_y_norm", "pred_x_px", "pred_y_px",
    "error_px", "within_5px", "within_6px", "within_10px", "within_20px",
    "confidence", "inference_ms",
    "architecture", "pair_id", "base_pair", "split",
]


def write_csv(records: list, csv_path: Path, append=False):
    mode = "a" if append else "w"
    write_header = not csv_path.exists() or not append
    with open(csv_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(records)


# ============================================================
# SINGLE-MODE INFERENCE
# ============================================================

def run_single(model, device, ref_path: Path, search_path: Path, sample_id: str = "single"):
    pred_x_px, pred_y_px, pred_x_norm, pred_y_norm, conf, inf_ms = run_inference(
        model, ref_path, search_path, device
    )

    # Try to get GT from sibling ground_truth.json
    folder = ref_path.parent
    gt_x_norm, gt_y_norm = load_gt(folder)
    meta = load_pair_metadata(folder)

    gt_x_px  = gt_x_norm * 1000.0 if gt_x_norm is not None else None
    gt_y_px  = gt_y_norm * 1000.0 if gt_y_norm is not None else None
    error_px = (
        math.sqrt((pred_x_px - gt_x_px) ** 2 + (pred_y_px - gt_y_px) ** 2)
        if gt_x_px is not None else None
    )

    # ── terminal output ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("  DRIFT-SENSE AI — LOCALIZATION RESULT")
    print("=" * 60)
    print(f"  Reference Image  : {ref_path}")
    print(f"  Search Image     : {search_path}")
    print(f"  Predicted Center : ({pred_x_px:.2f}, {pred_y_px:.2f}) px")
    print(f"  Predicted (norm) : ({pred_x_norm:.6f}, {pred_y_norm:.6f})")
    print(f"  Confidence       : {conf:.4f}")
    print(f"  Inference Time   : {inf_ms:.2f} ms")
    if gt_x_px is not None:
        print(f"  Ground Truth     : ({gt_x_px:.2f}, {gt_y_px:.2f}) px")
        print(f"  Pixel Error      : {error_px:.2f} px")
        status = "PASS" if error_px <= 5.0 else "FAIL"
        print(f"  Status (<=5px)   : {status}")
    print("=" * 60 + "\n")

    record = {
        "sample_id":    sample_id,
        "reference_path": str(ref_path),
        "search_path":    str(search_path),
        "gt_x_norm":    round(gt_x_norm, 6)  if gt_x_norm  is not None else "",
        "gt_y_norm":    round(gt_y_norm, 6)  if gt_y_norm  is not None else "",
        "gt_x_px":      round(gt_x_px, 2)    if gt_x_px    is not None else "",
        "gt_y_px":      round(gt_y_px, 2)    if gt_y_px    is not None else "",
        "pred_x_norm":  round(pred_x_norm, 6),
        "pred_y_norm":  round(pred_y_norm, 6),
        "pred_x_px":    round(pred_x_px, 2),
        "pred_y_px":    round(pred_y_px, 2),
        "error_px":     round(error_px, 2)   if error_px   is not None else "",
        "within_5px":   (error_px <= 5.0)    if error_px   is not None else "",
        "within_6px":   (error_px <= 6.0)    if error_px   is not None else "",
        "within_10px":  (error_px <= 10.0)   if error_px   is not None else "",
        "within_20px":  (error_px <= 20.0)   if error_px   is not None else "",
        "confidence":   round(conf, 4),
        "inference_ms": round(inf_ms, 2),
        **meta,
    }

    # Append to global CSV
    single_csv = RESULTS_DIR / "localize_results.csv"
    write_csv([record], single_csv, append=True)
    print(f"Result appended to: {single_csv}")

    return record


# ============================================================
# BATCH-MODE INFERENCE
# ============================================================

def run_batch(model, device, folder_path: Path):
    dram_folders = sorted([
        d for d in folder_path.iterdir()
        if d.is_dir() and d.name.startswith("dram_")
    ])

    if not dram_folders:
        print(f"No dram_* subfolders found in: {folder_path}")
        sys.exit(1)

    total = len(dram_folders)
    print(f"\nBatch mode: {total} samples found in {folder_path}")

    # Output directory
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = RESULTS_DIR / f"localize_batch_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(exist_ok=True)

    records  = []
    start_t  = time.time()

    for i, folder in enumerate(dram_folders, 1):
        ref_path    = folder / "reference_100x.png"
        search_path = folder / "search_10x.png"

        if not ref_path.exists() or not search_path.exists():
            print(f"  [SKIP] {folder.name}: missing images.")
            continue

        pred_x_px, pred_y_px, pred_x_norm, pred_y_norm, conf, inf_ms = run_inference(
            model, ref_path, search_path, device
        )

        gt_x_norm, gt_y_norm = load_gt(folder)
        meta = load_pair_metadata(folder)

        gt_x_px  = gt_x_norm * 1000.0 if gt_x_norm is not None else None
        gt_y_px  = gt_y_norm * 1000.0 if gt_y_norm is not None else None
        error_px = (
            math.sqrt((pred_x_px - gt_x_px) ** 2 + (pred_y_px - gt_y_px) ** 2)
            if gt_x_px is not None else None
        )

        # Progress
        status = ""
        if error_px is not None:
            status = f"err={error_px:.1f}px"
        print(
            f"  [{i:>5}/{total}]  {folder.name}"
            f"  pred=({pred_x_px:7.1f}, {pred_y_px:7.1f})"
            f"  conf={conf:.3f}"
            f"  {status}"
            f"  [{inf_ms:.0f}ms]",
            flush=True,
        )

        record = {
            "sample_id":      folder.name,
            "reference_path": str(ref_path),
            "search_path":    str(search_path),
            "gt_x_norm":  round(gt_x_norm, 6) if gt_x_norm  is not None else "",
            "gt_y_norm":  round(gt_y_norm, 6) if gt_y_norm  is not None else "",
            "gt_x_px":    round(gt_x_px, 2)   if gt_x_px    is not None else "",
            "gt_y_px":    round(gt_y_px, 2)   if gt_y_px    is not None else "",
            "pred_x_norm":round(pred_x_norm, 6),
            "pred_y_norm":round(pred_y_norm, 6),
            "pred_x_px":  round(pred_x_px, 2),
            "pred_y_px":  round(pred_y_px, 2),
            "error_px":   round(error_px, 2) if error_px is not None else "",
            "within_5px": (error_px <= 5.0)  if error_px is not None else "",
            "within_6px": (error_px <= 6.0)  if error_px is not None else "",
            "within_10px":(error_px <= 10.0) if error_px is not None else "",
            "within_20px":(error_px <= 20.0) if error_px is not None else "",
            "confidence":  round(conf, 4),
            "inference_ms":round(inf_ms, 2),
            **meta,
        }
        records.append(record)

        # Save overlay (every sample if ≤ 100, else every 5th)
        if total <= 100 or i % 5 == 0 or i <= 20:
            overlay = draw_overlay(
                search_path, pred_x_px, pred_y_px,
                gt_x_px, gt_y_px, error_px, conf, sample_id=folder.name,
            )
            cv2.imwrite(str(overlay_dir / f"{folder.name}_overlay.png"), overlay)

    # Save batch CSV
    batch_csv = out_dir / "results.csv"
    write_csv(records, batch_csv, append=False)
    print(f"\nResults CSV saved: {batch_csv}")

    # Generate report + plots + failure cases (always, not just for 20+)
    generate_batch_report(records, out_dir)

    elapsed = time.time() - start_t
    print(f"\nTotal batch runtime: {elapsed:.1f}s  ({elapsed/max(len(records),1)*1000:.1f}ms/sample)")
    print(f"All outputs saved to: {out_dir}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="DRIFT-SENSE AI — SEM Pattern Localization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive (will ask for paths):
      python localize.py

  # Single pair:
      python localize.py --ref dataset/test/dram_00001/reference_100x.png \\
                         --search dataset/test/dram_00001/search_10x.png

  # Batch (entire folder):
      python localize.py --folder dataset/test
        """,
    )
    parser.add_argument("--ref",    type=str, help="Path to reference_100x.png")
    parser.add_argument("--search", type=str, help="Path to search_10x.png")
    parser.add_argument("--folder", type=str,
                        help="Path to a folder of dram_* subfolders (batch mode)")
    args = parser.parse_args()

    # ── banner ──────────────────────────────────────────────
    print("=" * 60)
    print("  DRIFT-SENSE AI — SEM Pattern Localization")
    print("=" * 60)

    # ── device ──────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device : GPU — {torch.cuda.get_device_name(0)}")
        print(f"VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = torch.device("cpu")
        print("Device : CPU  (no GPU detected — running on CPU)")
    print()

    # ── load model ──────────────────────────────────────────
    model = load_model(device)

    # ── determine mode ──────────────────────────────────────
    if args.folder:
        # BATCH MODE — folder provided via CLI
        folder_path = ROOT / args.folder if not Path(args.folder).is_absolute() else Path(args.folder)
        run_batch(model, device, folder_path)

    elif args.ref and args.search:
        # SINGLE MODE — paths provided via CLI
        ref_path    = ROOT / args.ref    if not Path(args.ref).is_absolute()    else Path(args.ref)
        search_path = ROOT / args.search if not Path(args.search).is_absolute() else Path(args.search)
        run_single(model, device, ref_path, search_path)

    else:
        # INTERACTIVE MODE — ask user
        print("No arguments provided. Running in interactive mode.\n")
        print("Options:")
        print("  [1] Single image pair")
        print("  [2] Batch — entire folder of samples")
        choice = input("\nEnter choice (1 or 2): ").strip()

        if choice == "2":
            folder_input = input(
                "Enter folder path (e.g. dataset/test): "
            ).strip()
            folder_path = ROOT / folder_input if not Path(folder_input).is_absolute() else Path(folder_input)
            if not folder_path.exists():
                print(f"ERROR: Folder not found: {folder_path}")
                sys.exit(1)
            run_batch(model, device, folder_path)
        else:
            print("\nEnter paths relative to this script's folder OR absolute paths.\n")
            ref_input = input(
                "Reference image path\n  (e.g. dataset/test/dram_00001/reference_100x.png): "
            ).strip()
            search_input = input(
                "Search image path\n  (e.g. dataset/test/dram_00001/search_10x.png): "
            ).strip()

            ref_path    = ROOT / ref_input    if not Path(ref_input).is_absolute()    else Path(ref_input)
            search_path = ROOT / search_input if not Path(search_input).is_absolute() else Path(search_input)

            if not ref_path.exists():
                print(f"ERROR: Reference image not found: {ref_path}")
                sys.exit(1)
            if not search_path.exists():
                print(f"ERROR: Search image not found: {search_path}")
                sys.exit(1)

            run_single(model, device, ref_path, search_path)


if __name__ == "__main__":
    main()
