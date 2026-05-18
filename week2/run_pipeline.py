"""
week2/run_pipeline.py
=====================
Orchestrates the entire Week 2 pipeline end-to-end:
1. Prepares the 1,500-image dataset (if manifest is not present or --force-prep).
2. Trains the transfer learning classification model on CPU (with frozen backbone).
3. Evaluates the model on the validation split and generates performance plots.
"""

import argparse
import logging
import os
from pathlib import Path

# Programmatic imports from other modules
from prepare_dataset import main as prepare_main
from train import train_model
from evaluate import evaluate_model

# ---------------------------------------------------------------------------
# Setup Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Week 2 Classification Pipeline Orchestrator")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs (5 is perfect for frozen backbone on CPU)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--model-name", type=str, default="mobilenet_v3_large", help="resnet18 or mobilenet_v3_large")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--force-prep", action="store_true", default=False, help="Force rebuild/resize dataset from raw")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    log.info("🚀 STARTING WEEK 2 END-TO-END MEDICAL CV PIPELINE")
    
    # Paths
    manifest_path = Path("data/cleaned_week2/annotated_manifest_week2.csv")
    
    # Step 1: Dataset Preparation
    if not manifest_path.exists() or args.force_prep:
        log.info("Step 1: Manifest not found or --force-prep specified. Building dataset...")
        prepare_main()
    else:
        log.info("Step 1: 1,500-image manifest already exists at %s. Skipping preparation.", manifest_path)
        
    # Step 2: Model Training
    log.info("Step 2: Training transfer learning model '%s' on device '%s'...", args.model_name, args.device)
    # Package arguments for training script
    train_args = argparse.Namespace(
        manifest=str(manifest_path),
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=1e-4,
        model_name=args.model_name,
        freeze_backbone=True,  # Always freeze backbone for CPU training
        val_split=0.2,
        device=args.device,
        seed=args.seed
    )
    train_model(train_args)
    
    # Step 3: Model Evaluation
    log.info("Step 3: Evaluating the best model checkpoint on the validation split...")
    eval_args = argparse.Namespace(
        manifest=str(manifest_path),
        weights="week2/weights/best_classifier.pth",
        batch_size=args.batch_size,
        model_name=args.model_name,
        val_split=0.2,
        device=args.device,
        seed=args.seed
    )
    evaluate_model(eval_args)
    
    log.info("🎉 WEEK 2 END-TO-END PIPELINE SUCCESSFULLY COMPLETED!")
    log.info("You can view model weights in: week2/weights/")
    log.info("You can view loss curves and confusion matrices in: week2/metrics/")

if __name__ == "__main__":
    main()
