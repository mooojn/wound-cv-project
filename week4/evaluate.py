"""
week4/evaluate.py
=================
Evaluates the best segmentation model on the held-out TEST split.

Metrics computed:
  - IoU (Jaccard Index)
  - Dice Coefficient (F1 for segmentation)
  - Pixel Accuracy
  - Precision  (pixel-level)
  - Recall     (pixel-level)

Outputs:
  week4/metrics/segmentation_evaluation.json   — full metrics report
  week4/metrics/evaluation_curves.png          — IoU / Dice bar chart

Usage (from project root):
    python week4/evaluate.py
    python week4/evaluate.py --weights week4/weights/best_seg_model.pth --device cpu
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
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


# ---------------------------------------------------------------------------
# Pixel-level metric helpers
# ---------------------------------------------------------------------------

def _to_binary(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    return (torch.sigmoid(logits) > threshold).cpu().numpy().astype(np.uint8)


def compute_metrics_batch(
    logits: torch.Tensor,
    masks: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute segmentation metrics across the full batch."""
    preds = _to_binary(logits, threshold).flatten()
    trues = masks.cpu().numpy().astype(np.uint8).flatten()

    tp = np.logical_and(preds == 1, trues == 1).sum()
    tn = np.logical_and(preds == 0, trues == 0).sum()
    fp = np.logical_and(preds == 1, trues == 0).sum()
    fn = np.logical_and(preds == 0, trues == 1).sum()

    eps = 1e-7
    iou       = tp / (tp + fp + fn + eps)
    dice      = 2 * tp / (2 * tp + fp + fn + eps)
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)

    return {
        "iou":        float(iou),
        "dice":       float(dice),
        "pixel_acc":  float(pixel_acc),
        "precision":  float(precision),
        "recall":     float(recall),
    }


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace) -> None:
    log.info("━━━ Week 4 — Segmentation Evaluation (Test Set) ━━━")

    metrics_dir = Path("week4/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ────────────────────────────────────────────────────────────────
    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    log.info("Evaluation device: %s", device)

    # ── Load manifest → test split ─────────────────────────────────────────────
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    df = pd.read_csv(manifest_path)
    test_records = df[df["split"] == "test"].to_dict(orient="records")
    log.info("Test set size: %d samples", len(test_records))

    # ── Dataset & loader ──────────────────────────────────────────────────────
    test_dataset = SegmentationDataset(test_records, augment=False, image_size=args.image_size)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Load model ────────────────────────────────────────────────────────────
    weights_path = Path(args.weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"Weights not found: {weights_path}. Train the model first!"
        )

    model = get_segmentation_model(
        architecture=args.architecture,
        encoder_name=args.encoder,
        encoder_weights=None,       # weights already loaded below
        in_channels=3,
        num_classes=1,
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    log.info("Loaded weights from: %s", weights_path)

    # ── Inference ──────────────────────────────────────────────────────────────
    all_logits, all_masks = [], []

    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs   = imgs.to(device)
            logits = model(imgs)
            all_logits.append(logits.cpu())
            all_masks.append(masks.cpu())

    all_logits = torch.cat(all_logits)
    all_masks  = torch.cat(all_masks)

    # ── Compute metrics ────────────────────────────────────────────────────────
    metrics = compute_metrics_batch(all_logits, all_masks, threshold=args.threshold)

    # ── Console report ─────────────────────────────────────────────────────────
    print("\n" + "=" * 58)
    print("       WEEK 4 — SEGMENTATION EVALUATION REPORT         ")
    print("=" * 58)
    print(f"  IoU (Jaccard Index)  : {metrics['iou']:.4f}  ({metrics['iou']*100:.2f}%)")
    print(f"  Dice Coefficient     : {metrics['dice']:.4f}  ({metrics['dice']*100:.2f}%)")
    print(f"  Pixel Accuracy       : {metrics['pixel_acc']:.4f}  ({metrics['pixel_acc']*100:.2f}%)")
    print(f"  Precision (pixel)    : {metrics['precision']:.4f}  ({metrics['precision']*100:.2f}%)")
    print(f"  Recall    (pixel)    : {metrics['recall']:.4f}  ({metrics['recall']*100:.2f}%)")
    print("-" * 58)
    print(f"  Test samples         : {len(test_records)}")
    print(f"  Threshold            : {args.threshold}")
    print("=" * 58 + "\n")

    # ── Save JSON report ───────────────────────────────────────────────────────
    report = {
        "test_samples":  len(test_records),
        "threshold":     args.threshold,
        "model_weights": str(weights_path),
        **metrics,
    }
    report_path = metrics_dir / "segmentation_evaluation.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=4)
    log.info("Evaluation report → %s", report_path)

    # ── Bar chart of key metrics ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))

    metric_names  = ["IoU", "Dice", "Pixel Acc", "Precision", "Recall"]
    metric_values = [
        metrics["iou"], metrics["dice"], metrics["pixel_acc"],
        metrics["precision"], metrics["recall"],
    ]
    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#e74c3c"]

    bars = ax.bar(metric_names, [v * 100 for v in metric_values], color=colors, width=0.5, edgecolor="white")

    for bar, val in zip(bars, metric_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.8,
            f"{val*100:.1f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    ax.set_ylim(0, 115)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Wound Segmentation — Test Set Metrics", fontsize=14, fontweight="bold", pad=12)
    ax.grid(axis="y", alpha=0.4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    chart_path = metrics_dir / "evaluation_metrics_chart.png"
    plt.savefig(chart_path, dpi=300)
    plt.close()
    log.info("Metrics chart → %s", chart_path)
    log.info("━━━ Evaluation Complete ━━━")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Week 4 — Wound Segmentation Evaluation")

    parser.add_argument("--manifest",     type=str,   default="data/cleaned_week4/segmentation_manifest.csv")
    parser.add_argument("--weights",      type=str,   default="week4/weights/best_seg_model.pth")
    parser.add_argument("--architecture", type=str,   default="unet")
    parser.add_argument("--encoder",      type=str,   default="resnet34")
    parser.add_argument("--batch-size",   type=int,   default=8)
    parser.add_argument("--image-size",   type=int,   default=256)
    parser.add_argument("--threshold",    type=float, default=0.5)
    parser.add_argument("--device",       type=str,   default="cuda")

    args = parser.parse_args()

    if not torch.cuda.is_available():
        args.device = "cpu"

    evaluate(args)
