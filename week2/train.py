"""
week2/train.py
==============
Trains the binary classifier using transfer learning on the 1,500 preprocessed
images. Saves the best model checkpoint and plots training history curves.
"""

import argparse
import csv
import logging
import os
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import WoundDataset, get_transforms
from model import get_model

# ---------------------------------------------------------------------------
# Setup Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

def train_model(args):
    log.info("━━━ Starting Week 2 Training Pipeline ━━━")
    
    # 1. Create Output Directories
    weights_dir = Path("week2/weights")
    metrics_dir = Path("week2/metrics")
    weights_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. Check Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    log.info("Training device selected: %s", device)
    
    # 3. Load Dataset Manifest
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}. Run prepare_dataset.py first!")
        
    df = pd.read_csv(manifest_path)
    records = df.to_dict(orient="records")
    log.info("Loaded %d images from manifest.", len(records))
    
    # 4. Stratified Split (80% Train, 20% Val)
    labels = [r["annotated_label"] for r in records]
    train_records, val_records = train_test_split(
        records,
        test_size=args.val_split,
        stratify=labels,
        random_state=args.seed
    )
    
    log.info("Stratified split: %d train images, %d validation images.", len(train_records), len(val_records))
    
    # 5. Initialize PyTorch Datasets and Dataloaders
    train_transform, val_transform = get_transforms()
    train_dataset = WoundDataset(train_records, transform=train_transform)
    val_dataset = WoundDataset(val_records, transform=val_transform)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    # 6. Initialize Model
    model = get_model(
        model_name=args.model_name,
        num_classes=2,
        freeze_backbone=args.freeze_backbone
    )
    model.to(device)
    
    # Define Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Only optimize parameters that require gradients (classifier head if frozen)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    log.info("Number of trainable parameter tensors: %d", len(trainable_params))
    
    optimizer = optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # 7. Training Loop
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    best_weights_path = weights_dir / "best_classifier.pth"
    
    for epoch in range(1, args.epochs + 1):
        # ─── Training Stage ───
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
        train_loss = running_loss / total_train
        train_acc = correct_train / total_train
        
        # ─── Validation Stage ───
        model.eval()
        running_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                running_val_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
                
        val_loss = running_val_loss / total_val
        val_acc = correct_val / total_val
        
        # Update Scheduler
        scheduler.step()
        
        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        
        log.info(
            "Epoch %2d/%2d  |  Train Loss: %.4f  Train Acc: %6.2f%%  |  Val Loss: %.4f  Val Acc: %6.2f%%",
            epoch, args.epochs, train_loss, train_acc * 100, val_loss, val_acc * 100
        )
        
        # Save check point if Validation Accuracy improves
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_weights_path)
            log.info(" >>> Saved new best model checkpoint! (Val Acc: %.2f%%)", val_acc * 100)
            
    log.info("━━━ Training Complete! Best Val Accuracy: %.2f%% ━━━", best_val_acc * 100)
    
    # 8. Save History Logs
    history_csv = metrics_dir / "training_history.csv"
    history_df = pd.DataFrame(history)
    history_df.to_csv(history_csv, index_label="epoch")
    log.info("Saved training history to %s", history_csv)
    
    # 9. Plot Loss and Accuracy curves
    log.info("Generating and saving training curves...")
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs_range = range(1, args.epochs + 1)
    
    # Plot Loss Curve
    axes[0].plot(epochs_range, history["train_loss"], label="Train Loss", color="#e74c3c", linewidth=2.5, marker="o")
    axes[0].plot(epochs_range, history["val_loss"], label="Val Loss", color="#2c3e50", linewidth=2.5, linestyle="--", marker="x")
    axes[0].set_title("Training & Validation Loss", fontsize=14, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=12)
    axes[0].set_ylabel("Loss", fontsize=12)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.5)
    
    # Plot Accuracy Curve
    axes[1].plot(epochs_range, [acc * 100 for acc in history["train_acc"]], label="Train Acc", color="#2ecc71", linewidth=2.5, marker="o")
    axes[1].plot(epochs_range, [acc * 100 for acc in history["val_acc"]], label="Val Acc", color="#3498db", linewidth=2.5, linestyle="--", marker="x")
    axes[1].set_title("Training & Validation Accuracy", fontsize=14, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=12)
    axes[1].set_ylabel("Accuracy (%)", fontsize=12)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(metrics_dir / "training_curves.png", dpi=300)
    plt.close()
    log.info("Saved training curve figure → %s", metrics_dir / "training_curves.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Wound Classifier")
    parser.add_argument("--manifest", type=str, default="data/cleaned_week2/annotated_manifest_week2.csv", help="Path to manifest CSV")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--model-name", type=str, default="mobilenet_v3_large", help="resnet18 or mobilenet_v3_large")
    parser.add_argument("--freeze-backbone", action="store_true", default=True, help="Freeze backbone for fast CPU training")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    # We allow running directly or passing arguments
    parsed_args = parser.parse_args()
    
    # Enforce backbone freezing since we are on CPU (unless cuda is explicitly available and requested)
    if not torch.cuda.is_available():
        parsed_args.freeze_backbone = True
        parsed_args.device = "cpu"
        
    train_model(parsed_args)
