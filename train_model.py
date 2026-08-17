#!/usr/bin/env python3
"""
============================================================
  DRIFT-SENSE AI — Siamese ResNet50 Training Pipeline
============================================================
Single self-contained training file.

Architecture : Siamese ResNet50 + FPN + Correlation Head + Soft-Argmax
Dataset      : dataset/train  &  dataset/val  (auto-detected next to this file)
Output       : model/best_model.pth  (saved next to this file)

Training Config:
  - Epochs       : 50
  - Batch size   : 16
  - Optimizer    : AdamW  (lr=5e-4, weight_decay=1e-4)
  - Scheduler    : OneCycleLR (cosine, 10% warmup)
  - Precision    : AMP float16
  - Loss         : Wing Loss (50%) + Gaussian Heatmap KL (45%) + Confidence MSE (5%)

Run:
    python train_model.py
============================================================
"""

import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image, ImageFile
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ============================================================
# PATHS  (always relative to this script's location)
# ============================================================
ROOT        = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "dataset"
MODEL_DIR   = ROOT / "model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# TRAINING HYPER-PARAMETERS
# ============================================================
EPOCHS       = 50
BATCH_SIZE   = 16
IMAGE_SIZE   = 224
FEAT_CH      = 256          # FPN output channels
LR           = 5e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS  = 2
SAVE_EVERY   = 5            # save periodic checkpoint every N epochs

# Loss weights
ALPHA = 0.50   # Wing Loss
BETA  = 0.45   # Gaussian Heatmap KL
GAMMA = 0.05   # Confidence MSE


# ============================================================
# 1.  DATASET
# ============================================================

class SEMPairDataset(Dataset):
    """Loads SEM reference/search pairs with normalised (x,y) ground truth."""

    MEAN = [0.485, 0.456, 0.406]
    STD  = [0.229, 0.224, 0.225]

    def __init__(self, dataset_root: Path, split: str, augment: bool = False):
        split_dir = dataset_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Dataset split not found: {split_dir}")

        self.folders = sorted([
            d for d in split_dir.iterdir()
            if d.is_dir() and d.name.startswith("dram_")
        ])
        print(f"  [{split:>5}]  {len(self.folders)} samples  ({split_dir})")

        self.augment = augment

        base = [T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor()]
        aug  = [
            T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
            T.RandomApply([T.ColorJitter(brightness=0.1, contrast=0.1)],      p=0.3),
            T.RandomRotation(degrees=3),
            T.ToTensor(),
        ]
        norm = T.Normalize(mean=self.MEAN, std=self.STD)

        self.tf_base = T.Compose(base + [norm])
        self.tf_aug  = T.Compose(aug  + [norm])

    def __len__(self):
        return len(self.folders)

    def _load_gt(self, gt_path: Path):
        with open(gt_path) as f:
            gt = json.load(f)

        # Support all GT formats produced by generate_dataset.py
        if "ground_truth" in gt and isinstance(gt["ground_truth"], dict):
            x = float(gt["ground_truth"]["x"])
            y = float(gt["ground_truth"]["y"])
        elif "target" in gt:
            t = gt["target"]
            if "search_center_xy" in t:
                x = float(t["search_center_xy"][0]) / 1000.0
                y = float(t["search_center_xy"][1]) / 1000.0
            elif "search_box_xywh" in t:
                b = t["search_box_xywh"]
                x = (float(b[0]) + float(b[2]) / 2) / 1000.0
                y = (float(b[1]) + float(b[3]) / 2) / 1000.0
            else:
                x, y = 0.5, 0.5
        else:
            x = float(gt.get("gt_x", 0.5))
            y = float(gt.get("gt_y", 0.5))

        return float(np.clip(x, 0.0, 1.0)), float(np.clip(y, 0.0, 1.0))

    def __getitem__(self, idx):
        folder = self.folders[idx]
        ref    = Image.open(folder / "reference_100x.png").convert("RGB")
        search = Image.open(folder / "search_10x.png").convert("RGB")
        gt_x, gt_y = self._load_gt(folder / "ground_truth.json")

        tf = self.tf_aug if self.augment else self.tf_base
        return {
            "reference": tf(ref),
            "search":    tf(search),
            "target_x":  torch.tensor(gt_x, dtype=torch.float32),
            "target_y":  torch.tensor(gt_y, dtype=torch.float32),
        }


# ============================================================
# 2.  MODEL  (Siamese ResNet50 + FPN + Correlation Head)
# ============================================================

class FeaturePyramidFusion(nn.Module):
    """Fuses ResNet50 layer2/3/4 into a single 256-ch 28x28 feature map."""

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
        sz  = c2.shape[-2:]
        p4  = F.interpolate(self.proj4(c4), size=sz, mode="bilinear", align_corners=False)
        p3  = F.interpolate(self.proj3(c3), size=sz, mode="bilinear", align_corners=False)
        p2  = self.proj2(c2)
        return self.fuse(torch.cat([p2, p3, p4], dim=1))


class CorrelationHead(nn.Module):
    """Cosine cross-correlation + conv heatmap head."""

    def __init__(self, feat_ch=256, hidden=128):
        super().__init__()
        self.ref_proj    = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(True))
        self.search_proj = nn.Sequential(
            nn.Conv2d(feat_ch, hidden, 1, bias=False), nn.BatchNorm2d(hidden), nn.ReLU(True))
        self.head = nn.Sequential(
            nn.Conv2d(hidden + 1, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden), nn.ReLU(True),
            nn.Conv2d(hidden, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 1, 1),
        )

    def forward(self, ref_feat, srch_feat):
        rf = self.ref_proj(ref_feat)
        sf = self.search_proj(srch_feat)
        B, C, H, W = sf.shape
        rf_n  = F.normalize(rf.reshape(B, C, -1), p=2, dim=1)
        sf_n  = F.normalize(sf.reshape(B, C, -1), p=2, dim=1)
        corr  = torch.bmm(rf_n.transpose(1, 2), sf_n).mean(dim=1).reshape(B, 1, H, W)
        heatmap = self.head(torch.cat([corr, sf], dim=1))
        return heatmap, corr


class SoftArgmax2d(nn.Module):
    """Converts a heatmap into a continuous (x, y) coordinate in [0, 1]."""

    def forward(self, heatmap):
        B, _, H, W = heatmap.shape
        prob = F.softmax(heatmap.view(B, -1), dim=-1).view(B, 1, H, W)
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, H, device=heatmap.device),
            torch.linspace(0, 1, W, device=heatmap.device),
            indexing="ij",
        )
        px = (prob * gx).sum(dim=(1, 2, 3))
        py = (prob * gy).sum(dim=(1, 2, 3))
        return torch.stack([px, py], dim=1)   # [B, 2]


class SiameseResNet50(nn.Module):
    """
    Full model:
      Shared ResNet50 encoder (layer1 frozen) ->
      Multi-scale FPN ->
      Correlation Head ->
      Soft-Argmax
    """

    def __init__(self, feat_ch=256, pretrained=True):
        super().__init__()
        bb = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)

        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.relu, bb.maxpool)
        self.layer1 = bb.layer1   # frozen below
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4

        # Freeze layer1 — keeps low-level edges stable, saves memory
        for p in self.layer1.parameters():
            p.requires_grad = False

        self.fpn        = FeaturePyramidFusion(out_ch=feat_ch)
        self.corr_head  = CorrelationHead(feat_ch=feat_ch)
        self.soft_argmax = SoftArgmax2d()

    def encode(self, x):
        x  = self.stem(x)
        x  = self.layer1(x)
        c2 = self.layer2(x)
        c3 = self.layer3(c2)
        c4 = self.layer4(c3)
        return self.fpn(c2, c3, c4)

    def forward(self, reference, search):
        ref_feat  = self.encode(reference)
        srch_feat = self.encode(search)

        heatmap, corr = self.corr_head(ref_feat, srch_feat)
        coords        = self.soft_argmax(heatmap)
        confidence    = torch.sigmoid(
            F.adaptive_avg_pool2d(corr, 1).view(heatmap.shape[0])
        )
        return coords, confidence, heatmap


# ============================================================
# 3.  LOSS FUNCTIONS
# ============================================================

class WingLoss(nn.Module):
    """Aggressively penalises small errors — ideal for sub-5px accuracy."""
    def __init__(self, w=10.0, eps=2.0):
        super().__init__()
        self.w = w
        self.eps = eps
        self.C = w - w * float(np.log(1.0 + w / eps))

    def forward(self, pred, target):
        diff = (pred - target).abs()
        loss = torch.where(
            diff < self.w,
            self.w * torch.log(1.0 + diff / self.eps),
            diff - self.C,
        )
        return loss.mean()


class GaussianHeatmapLoss(nn.Module):
    """KL divergence between predicted heatmap and a 2-D Gaussian centred at GT."""
    def __init__(self, sigma=0.6):
        super().__init__()
        self.sigma = sigma

    def _make_gt(self, txy, H, W, device):
        B  = txy.shape[0]
        gy, gx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device),
            torch.linspace(0, 1, W, device=device),
            indexing="ij",
        )
        gx = gx.unsqueeze(0)  # [1, H, W]
        gy = gy.unsqueeze(0)
        gtx = txy[:, 0].view(B, 1, 1)
        gty = txy[:, 1].view(B, 1, 1)
        gauss = torch.exp(
            -((gx - gtx) ** 2 + (gy - gty) ** 2) / (2 * self.sigma ** 2)
        )
        gauss = gauss / (gauss.sum(dim=(1, 2), keepdim=True) + 1e-8)
        return gauss.unsqueeze(1)   # [B, 1, H, W]

    def forward(self, heatmap, target_xy):
        B, _, H, W = heatmap.shape
        gt       = self._make_gt(target_xy, H, W, heatmap.device)
        log_prob = F.log_softmax(heatmap.view(B, -1), dim=-1).view(B, 1, H, W)
        return -(gt * log_prob).sum(dim=(1, 2, 3)).mean()


class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.wing     = WingLoss(w=10.0, eps=2.0)
        self.heatmap  = GaussianHeatmapLoss(sigma=0.6)
        self.conf_mse = nn.MSELoss()

    def forward(self, coords, confidence, heatmap, target):
        wing_l   = self.wing(coords, target)
        heat_l   = self.heatmap(heatmap, target)
        conf_l   = self.conf_mse(confidence.float(), torch.ones_like(confidence).float())
        total    = ALPHA * wing_l + BETA * heat_l + GAMMA * conf_l
        return total, {
            "total":   total.item(),
            "wing":    wing_l.item(),
            "heatmap": heat_l.item(),
            "conf":    conf_l.item(),
        }


# ============================================================
# 4.  TRAINING & VALIDATION
# ============================================================

def validate(model, loader, device):
    model.eval()
    errors = []
    with torch.no_grad():
        for batch in loader:
            ref    = batch["reference"].to(device, non_blocking=True)
            search = batch["search"].to(device, non_blocking=True)
            tx     = batch["target_x"].to(device, non_blocking=True)
            ty     = batch["target_y"].to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                coords, _, _ = model(ref, search)

            pred_x_px = coords[:, 0] * 1000.0
            pred_y_px = coords[:, 1] * 1000.0
            true_x_px = tx * 1000.0
            true_y_px = ty * 1000.0

            err = torch.sqrt((pred_x_px - true_x_px) ** 2 + (pred_y_px - true_y_px) ** 2)
            errors.extend(err.cpu().numpy().tolist())

    arr = np.array(errors)
    return {
        "mean_px":  float(np.mean(arr)),
        "median_px":float(np.median(arr)),
        "acc_5px":  float((arr <= 5.0).mean() * 100),
        "acc_10px": float((arr <= 10.0).mean() * 100),
    }


def train_one_epoch(model, loader, loss_fn, optimizer, scheduler, scaler, device, epoch):
    model.train()
    totals = {"total": 0.0, "wing": 0.0, "heatmap": 0.0, "conf": 0.0}
    steps  = 0
    log_every = max(1, len(loader) // 5)   # print ~5 times per epoch

    for step, batch in enumerate(loader, 1):
        ref    = batch["reference"].to(device, non_blocking=True)
        search = batch["search"].to(device, non_blocking=True)
        tx     = batch["target_x"].to(device, non_blocking=True)
        ty     = batch["target_y"].to(device, non_blocking=True)
        tgt    = torch.stack([tx, ty], dim=1)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            coords, conf, heatmap = model(ref, search)
            loss, ld = loss_fn(coords, conf, heatmap, tgt)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        for k in totals:
            totals[k] += ld[k]
        steps += 1

        if step % log_every == 0:
            lr = scheduler.get_last_lr()[0]
            print(
                f"  Ep[{epoch}] step {step}/{len(loader)}"
                f"  loss={ld['total']:.4f}"
                f"  wing={ld['wing']:.4f}"
                f"  heat={ld['heatmap']:.4f}"
                f"  lr={lr:.2e}",
                flush=True,
            )

    return {k: v / max(steps, 1) for k, v in totals.items()}


# ============================================================
# 5.  MAIN
# ============================================================

def main():
    # ── banner ──────────────────────────────────────────────
    print("=" * 70)
    print("  DRIFT-SENSE AI — SIAMESE RESNET50 TRAINING")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device      : {device}", flush=True)
    if device.type == "cuda":
        print(f"GPU         : {torch.cuda.get_device_name(0)}")
        print(f"VRAM        : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    print(f"Dataset     : {DATASET_DIR}")
    print(f"Model output: {MODEL_DIR}")
    print(f"Epochs      : {EPOCHS}")
    print(f"Batch size  : {BATCH_SIZE}")
    print(f"Image size  : {IMAGE_SIZE}")

    if not DATASET_DIR.exists():
        print(f"\nERROR: Dataset not found at {DATASET_DIR}")
        print("Please run  python generate_dataset.py  first.")
        sys.exit(1)

    # ── datasets ────────────────────────────────────────────
    print("\nLoading datasets...")
    train_ds = SEMPairDataset(DATASET_DIR, split="train", augment=True)
    val_ds   = SEMPairDataset(DATASET_DIR, split="val",   augment=False)

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── model ───────────────────────────────────────────────
    print("\nBuilding Siamese ResNet50 model...")
    model = SiameseResNet50(feat_ch=FEAT_CH, pretrained=True).to(device)

    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params    : {total:,}")
    print(f"Trainable params: {trainable:,}")

    # ── loss / optimiser / scheduler ────────────────────────
    loss_fn   = CombinedLoss().to(device)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=WEIGHT_DECAY,
    )
    scheduler = OneCycleLR(
        optimizer,
        max_lr=LR,
        total_steps=len(train_loader) * EPOCHS,
        pct_start=0.1,
        anneal_strategy="cos",
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # ── training loop ────────────────────────────────────────
    best_acc5px = 0.0
    history     = []

    print("\n" + "=" * 70)
    print("  TRAINING START")
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        print(f"\n--- Epoch [{epoch}/{EPOCHS}] ---", flush=True)
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, loss_fn, optimizer, scheduler, scaler, device, epoch
        )
        val_metrics = validate(model, val_loader, device)

        elapsed = time.time() - t0
        print(
            f"\nEpoch {epoch} | "
            f"train_loss={train_metrics['total']:.4f} | "
            f"mean_err={val_metrics['mean_px']:.2f}px | "
            f"median={val_metrics['median_px']:.2f}px | "
            f"acc@5px={val_metrics['acc_5px']:.2f}% | "
            f"acc@10px={val_metrics['acc_10px']:.2f}% | "
            f"time={elapsed:.0f}s",
            flush=True,
        )

        # Save periodic checkpoint
        if epoch % SAVE_EVERY == 0:
            ckpt_path = MODEL_DIR / f"checkpoint_epoch_{epoch}.pth"
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics":        val_metrics,
            }, ckpt_path)
            print(f"  Saved checkpoint -> {ckpt_path}", flush=True)

        # Save best model
        if val_metrics["acc_5px"] > best_acc5px:
            best_acc5px = val_metrics["acc_5px"]
            best_path   = MODEL_DIR / "best_model.pth"
            torch.save({
                "epoch":              epoch,
                "model_state_dict":   model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_acc5px":        best_acc5px,
                "val_metrics":        val_metrics,
            }, best_path)
            print(f"  *** New best acc@5px = {best_acc5px:.2f}% -> {best_path} ***", flush=True)

        history.append({"epoch": epoch, **train_metrics, **val_metrics})

    # ── save final model ─────────────────────────────────────
    final_path = MODEL_DIR / "final_model.pth"
    torch.save({
        "epoch":            EPOCHS,
        "model_state_dict": model.state_dict(),
        "best_acc5px":      best_acc5px,
    }, final_path)

    # ── save training history ────────────────────────────────
    import json as _json
    with open(MODEL_DIR / "training_history.json", "w") as f:
        _json.dump(history, f, indent=2)

    # ── final summary ────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Best accuracy (<=5px) : {best_acc5px:.2f}%")
    print(f"  Best model saved at   : {MODEL_DIR / 'best_model.pth'}")
    print(f"  Final model saved at  : {final_path}")
    print(f"  Training history      : {MODEL_DIR / 'training_history.json'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
