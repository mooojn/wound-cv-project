"""
week4/run_week4_pipeline.py
===========================
End-to-end runner for the Week 4 segmentation pipeline.

Runs the following steps in order:
    Step A : prepare_segmentation_dataset.py   — pair images/masks, 70/15/15 split
    Step B : train.py                          — train U-Net segmentation model
    Step C : evaluate.py                       — evaluate on test set (IoU, Dice, etc.)
    Step D : predict.py                        — generate visualisations & prediction grid

Usage (from project root):
    python week4/run_week4_pipeline.py
    python week4/run_week4_pipeline.py --device cpu --epochs 20
    python week4/run_week4_pipeline.py --skip-prepare   # skip dataset prep if already done
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n{'='*60}")
    print(f"[RUN] {' '.join(cmd)}")
    print(f"{'='*60}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Week 4 — Full Segmentation Pipeline Runner"
    )
    parser.add_argument("--device",        type=str, default="auto",
                        help="cuda | cpu | auto")
    parser.add_argument("--epochs",        type=int, default=25)
    parser.add_argument("--batch-size",    type=int, default=8)
    parser.add_argument("--image-size",    type=int, default=256)
    parser.add_argument("--architecture",  type=str, default="unet")
    parser.add_argument("--encoder",       type=str, default="resnet34")
    parser.add_argument("--num-samples",   type=int, default=15,
                        help="Number of test samples to visualise in Step D")
    parser.add_argument("--skip-prepare",  action="store_true",
                        help="Skip dataset preparation (Step A)")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    return device_arg


def main() -> None:
    args   = parse_args()
    py     = sys.executable
    root   = Path(__file__).resolve().parents[1]
    w4_dir = root / "week4"

    device = resolve_device(args.device)
    print(f"\n{'━'*60}")
    print(f"  Week 4 Segmentation Pipeline")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {args.epochs}")
    print(f"  Batch Size : {args.batch_size}")
    print(f"  Image Size : {args.image_size}x{args.image_size}")
    print(f"  Arch       : {args.architecture} / {args.encoder}")
    print(f"{'━'*60}\n")

    # ── Step A: Dataset Preparation ─────────────────────────────────────────
    if not args.skip_prepare:
        run([py, str(w4_dir / "prepare_segmentation_dataset.py")])
    else:
        print("[SKIP] Step A: Dataset preparation (--skip-prepare set)")

    # ── Step B: Training ────────────────────────────────────────────────────
    run([
        py, str(w4_dir / "train.py"),
        "--epochs",       str(args.epochs),
        "--batch-size",   str(args.batch_size),
        "--image-size",   str(args.image_size),
        "--architecture", args.architecture,
        "--encoder",      args.encoder,
        "--device",       device,
    ])

    # ── Step C: Evaluation ──────────────────────────────────────────────────
    run([
        py, str(w4_dir / "evaluate.py"),
        "--architecture", args.architecture,
        "--encoder",      args.encoder,
        "--image-size",   str(args.image_size),
        "--device",       device,
    ])

    # ── Step D: Predictions & Visualisations ────────────────────────────────
    run([
        py, str(w4_dir / "predict.py"),
        "--architecture", args.architecture,
        "--encoder",      args.encoder,
        "--image-size",   str(args.image_size),
        "--num-samples",  str(args.num_samples),
        "--device",       device,
    ])

    print(f"\n{'━'*60}")
    print("  ✅  Week 4 Pipeline Complete!")
    print(f"  Weights   → week4/weights/best_seg_model.pth")
    print(f"  Metrics   → week4/metrics/")
    print(f"  Previews  → week4/predictions/")
    print(f"{'━'*60}\n")


if __name__ == "__main__":
    main()
