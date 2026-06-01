"""
week4/predict.py
================
Runs inference on the test split and saves side-by-side visualisations:
    Original Image | Ground Truth Mask | Predicted Mask | Overlay

Outputs:
    week4/predictions/sample_XXXX.png   — per-sample visualisation grids
    week4/predictions/prediction_grid.png — combined overview of N samples

Usage (from project root):
    python week4/predict.py
    python week4/predict.py --num-samples 20 --device cpu
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader

from dataset import SegmentationDataset
from model import get_segmentation_model

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


def denormalise(tensor: torch.Tensor) -> np.ndarray:
    """Convert a normalised (C, H, W) tensor back to a (H, W, 3) uint8 image."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_mask(image: np.ndarray, mask: np.ndarray, color=(255, 50, 50), alpha=0.45) -> np.ndarray:
    """Blend a binary mask over the original image with transparency."""
    overlay = image.copy()
    wound_pixels = mask.squeeze() > 0.5
    overlay[wound_pixels] = (
        (1 - alpha) * image[wound_pixels] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


# ---------------------------------------------------------------------------

def predict(args: argparse.Namespace) -> None:
    log.info("━━━ Week 4 — Segmentation Predictions & Visualisation ━━━")

    out_dir = Path("week4/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    log.info("Inference device: %s", device)

    # ── Load manifest → test split ─────────────────────────────────────────────
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    test_records = df[df["split"] == "test"].to_dict(orient="records")

    # Limit to requested number of samples
    samples = test_records[: args.num_samples]
    log.info("Generating visualisations for %d samples.", len(samples))

    # ── Dataset & loader (batch_size=1 for individual saves) ──────────────────
    test_dataset = SegmentationDataset(samples, augment=False, image_size=args.image_size)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0)

    # ── Load model ────────────────────────────────────────────────────────────
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}. Train first!")

    model = get_segmentation_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    log.info("Loaded weights: %s", weights_path)

    # ── Collect predictions ────────────────────────────────────────────────────
    saved_items = []   # (img_np, gt_mask_np, pred_mask_np, overlay_np)

    with torch.no_grad():
        for idx, (imgs, masks) in enumerate(test_loader):
            imgs   = imgs.to(device)
            logits = model(imgs)
            pred   = (torch.sigmoid(logits) > args.threshold).float()

            img_np      = denormalise(imgs[0])
            gt_np       = masks[0].squeeze().cpu().numpy()
            pred_np     = pred[0].squeeze().cpu().numpy()
            overlay_np  = overlay_mask(img_np, pred_np[..., np.newaxis])

            saved_items.append((img_np, gt_np, pred_np, overlay_np))

            # ── Save individual 4-panel figure ────────────────────────────────
            fig, axes = plt.subplots(1, 4, figsize=(18, 4))
            fig.patch.set_facecolor("#1a1a2e")

            panels = [
                (img_np,      "Original Image",      "viridis"),
                (gt_np,       "Ground Truth Mask",   "bone"),
                (pred_np,     "Predicted Mask",      "bone"),
                (overlay_np,  "Prediction Overlay",  "viridis"),
            ]

            for ax, (data, title, cmap) in zip(axes, panels):
                if data.ndim == 3:
                    ax.imshow(data)
                else:
                    ax.imshow(data, cmap=cmap, vmin=0, vmax=1)
                ax.set_title(title, color="white", fontsize=10, fontweight="bold", pad=6)
                ax.axis("off")

            rec    = samples[idx]
            sample_idx = rec.get("index", idx)
            plt.suptitle(
                f"Sample {sample_idx:04d}",
                color="white", fontsize=12, fontweight="bold", y=1.01
            )
            plt.tight_layout()

            save_path = out_dir / f"sample_{sample_idx:04d}.png"
            plt.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="#1a1a2e")
            plt.close()

    log.info("Saved %d individual prediction images → %s", len(saved_items), out_dir)

    # ── Combined overview grid ─────────────────────────────────────────────────
    n      = len(saved_items)
    cols   = 4          # columns per row: orig | gt | pred | overlay
    rows   = n          # one row per sample

    fig = plt.figure(figsize=(cols * 4.5, rows * 4), facecolor="#0f0f23")
    gs  = gridspec.GridSpec(rows, cols, figure=fig, hspace=0.05, wspace=0.03)

    col_titles = ["Original", "Ground Truth", "Prediction", "Overlay"]

    for r, (img, gt, pred, ov) in enumerate(saved_items):
        data_list = [img, gt, pred, ov]
        for c, data in enumerate(data_list):
            ax = fig.add_subplot(gs[r, c])
            if data.ndim == 3:
                ax.imshow(data)
            else:
                ax.imshow(data, cmap="bone", vmin=0, vmax=1)
            ax.axis("off")
            if r == 0:
                ax.set_title(col_titles[c], color="#00d4ff", fontsize=11,
                             fontweight="bold", pad=5)

    plt.suptitle(
        "Wound Segmentation — Test Predictions Overview",
        color="white", fontsize=15, fontweight="bold", y=1.005
    )

    grid_path = out_dir / "prediction_grid.png"
    plt.savefig(grid_path, dpi=180, bbox_inches="tight", facecolor="#0f0f23")
    plt.close()
    log.info("Prediction grid saved → %s", grid_path)
    log.info("━━━ Prediction & Visualisation Complete ━━━")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 4 — Segmentation Predictions")

    parser.add_argument("--manifest",     type=str,   default="data/cleaned_week4/segmentation_manifest.csv")
    parser.add_argument("--weights",      type=str,   default="week4/weights/best_seg_model.pth")
    parser.add_argument("--architecture", type=str,   default="unet")
    parser.add_argument("--encoder",      type=str,   default="resnet34")
    parser.add_argument("--image-size",   type=int,   default=256)
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--num-samples",  type=int,   default=15,
                        help="Number of test samples to visualise")
    parser.add_argument("--device",       type=str,   default="cuda")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = "cpu"

    predict(args)
