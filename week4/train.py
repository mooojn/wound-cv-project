"""
week4/train.py
==============
Trains the U-Net segmentation model on the wound dataset.

Loss    : Combined Dice Loss + Binary Cross-Entropy (BCEWithLogitsLoss)
Metrics : IoU (Jaccard) and Dice Coefficient tracked per epoch.
Saves   : best model weights + training history CSV + training curves PNG.

Usage (from project root):
    python week4/train.py
    python week4/train.py --epochs 30 --batch-size 8 --device cuda
"""

import argparse
import csv
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from model import get_segmentation_model

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(pred_logits: torch.Tensor, targets: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """
    Soft Dice Loss computed on sigmoid-activated predictions.

    Args:
        pred_logits : Raw model output  (B, 1, H, W)
        targets     : Binary masks      (B, 1, H, W) with values in {0, 1}
        smooth      : Laplace smoothing to avoid division by zero
    """
    preds = torch.sigmoid(pred_logits)
    preds_flat   = preds.view(preds.size(0), -1)
    targets_flat = targets.view(targets.size(0), -1)

    intersection = (preds_flat * targets_flat).sum(dim=1)
    denominator  = preds_flat.sum(dim=1) + targets_flat.sum(dim=1)

    dice = 1.0 - (2.0 * intersection + smooth) / (denominator + smooth)
    return dice.mean()


def combined_loss(
    pred_logits: torch.Tensor,
    targets: torch.Tensor,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """50% BCEWithLogitsLoss + 50% Dice Loss — effective for imbalanced masks."""
    bce  = nn.BCEWithLogitsLoss()(pred_logits, targets)
    dice = dice_loss(pred_logits, targets)
    return bce_weight * bce + (1.0 - bce_weight) * dice


# ---------------------------------------------------------------------------
# Metric helpers (computed on CPU numpy after each epoch)
# ---------------------------------------------------------------------------

def compute_iou(pred_logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds   = (torch.sigmoid(pred_logits) > threshold).float()
    p_flat  = preds.view(-1).cpu().numpy()
    t_flat  = targets.view(-1).cpu().numpy()
    intersection = np.logical_and(p_flat, t_flat).sum()
    union        = np.logical_or(p_flat, t_flat).sum()
    return float(intersection / (union + 1e-7))


def compute_dice(pred_logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5) -> float:
    preds   = (torch.sigmoid(pred_logits) > threshold).float()
    p_flat  = preds.view(-1).cpu().numpy().astype(bool)
    t_flat  = targets.view(-1).cpu().numpy().astype(bool)
    intersection = np.logical_and(p_flat, t_flat).sum()
    return float(2.0 * intersection / (p_flat.sum() + t_flat.sum() + 1e-7))


# ---------------------------------------------------------------------------
# Training function
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    log.info("━━━ Week 4 — Segmentation Training Pipeline ━━━")

    # ── Directories ─────────────────────────────────────────────────────────
    weights_dir = Path("week4/weights")
    metrics_dir = Path("week4/metrics")
    weights_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        if args.device == "cuda":
            log.warning("CUDA requested but not available — falling back to CPU.")
        device = torch.device("cpu")
    log.info("Training device: %s", device)

    # ── Load manifest ────────────────────────────────────────────────────────
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "Run prepare_segmentation_dataset.py first!"
        )

    df = pd.read_csv(manifest_path)
    train_records = df[df["split"] == "train"].to_dict(orient="records")
    val_records   = df[df["split"] == "val"].to_dict(orient="records")
    log.info("Dataset → train: %d | val: %d", len(train_records), len(val_records))

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_dataset = SegmentationDataset(train_records, augment=True,  image_size=args.image_size)
    val_dataset   = SegmentationDataset(val_records,   augment=False, image_size=args.image_size)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=0, pin_memory=False
    )
    val_loader = DataLoader(
        val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=0, pin_memory=False
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = get_segmentation_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights="imagenet",
        in_channels=3,
        num_classes=1,
    )
    model.to(device)

    # ── Optimiser + scheduler ─────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # ── Training loop ─────────────────────────────────────────────────────────
    history = {
        "train_loss": [], "train_iou": [], "train_dice": [],
        "val_loss":   [], "val_iou":   [], "val_dice":   [],
    }

    best_val_iou    = 0.0
    best_weights_path = weights_dir / "best_seg_model.pth"

    for epoch in range(1, args.epochs + 1):

        # ── Train ──────────────────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        all_train_preds, all_train_masks = [], []

        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss   = combined_loss(logits, masks, bce_weight=args.bce_weight)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            all_train_preds.append(logits.detach().cpu())
            all_train_masks.append(masks.detach().cpu())

        train_loss = running_loss / len(train_dataset)
        all_train_preds = torch.cat(all_train_preds)
        all_train_masks = torch.cat(all_train_masks)
        train_iou  = compute_iou(all_train_preds, all_train_masks)
        train_dice = compute_dice(all_train_preds, all_train_masks)

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        running_val_loss = 0.0
        all_val_preds, all_val_masks = [], []

        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits      = model(imgs)
                loss        = combined_loss(logits, masks, bce_weight=args.bce_weight)

                running_val_loss += loss.item() * imgs.size(0)
                all_val_preds.append(logits.cpu())
                all_val_masks.append(masks.cpu())

        val_loss = running_val_loss / len(val_dataset)
        all_val_preds = torch.cat(all_val_preds)
        all_val_masks = torch.cat(all_val_masks)
        val_iou  = compute_iou(all_val_preds, all_val_masks)
        val_dice = compute_dice(all_val_preds, all_val_masks)

        scheduler.step()

        # ── Record ─────────────────────────────────────────────────────────
        history["train_loss"].append(train_loss)
        history["train_iou"].append(train_iou)
        history["train_dice"].append(train_dice)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)
        history["val_dice"].append(val_dice)

        log.info(
            "Epoch %2d/%2d | "
            "Train Loss: %.4f  IoU: %.4f  Dice: %.4f | "
            "Val   Loss: %.4f  IoU: %.4f  Dice: %.4f",
            epoch, args.epochs,
            train_loss, train_iou, train_dice,
            val_loss, val_iou, val_dice,
        )

        # ── Checkpoint ─────────────────────────────────────────────────────
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), best_weights_path)
            log.info("  >>> New best model saved! (Val IoU: %.4f)", best_val_iou)

    log.info("━━━ Training complete. Best Val IoU: %.4f ━━━", best_val_iou)

    # ── Save history CSV ──────────────────────────────────────────────────────
    history_df = pd.DataFrame(history)
    history_csv = metrics_dir / "training_history.csv"
    history_df.to_csv(history_csv, index_label="epoch")
    log.info("Training history → %s", history_csv)

    # ── Plot training curves ───────────────────────────────────────────────────
    log.info("Plotting training curves...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    epochs_range = range(1, args.epochs + 1)
    palette = {"train": "#e74c3c", "val": "#3498db"}

    for ax, metric, title in zip(
        axes,
        [("train_loss", "val_loss"), ("train_iou", "val_iou"), ("train_dice", "val_dice")],
        ["Loss (Combined BCE + Dice)", "IoU (Jaccard)", "Dice Coefficient"],
    ):
        ax.plot(epochs_range, history[metric[0]], label="Train", color=palette["train"],
                linewidth=2.5, marker="o", markersize=4)
        ax.plot(epochs_range, history[metric[1]], label="Val",   color=palette["val"],
                linewidth=2.5, marker="x", markersize=5, linestyle="--")
        ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
        ax.set_xlabel("Epoch", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.4)

    plt.tight_layout()
    curves_path = metrics_dir / "training_curves.png"
    plt.savefig(curves_path, dpi=300)
    plt.close()
    log.info("Training curves → %s", curves_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 4 — Wound Segmentation Training")

    parser.add_argument("--manifest",     type=str,   default="data/cleaned_week4/segmentation_manifest.csv")
    parser.add_argument("--architecture", type=str,   default="unet",    help="unet | unetplusplus | fpn | deeplabv3plus")
    parser.add_argument("--encoder",      type=str,   default="resnet34", help="smp encoder name")
    parser.add_argument("--epochs",       type=int,   default=25)
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--image-size",   type=int,   default=256)
    parser.add_argument("--lr",           type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--bce-weight",   type=float, default=0.5,  help="Weight for BCE in combined loss (rest goes to Dice)")
    parser.add_argument("--device",       type=str,   default="cuda")

    args = parser.parse_args()

    # Auto-fall-back to CPU
    if not torch.cuda.is_available():
        args.device = "cpu"

    train(args)
