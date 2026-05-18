"""
week2/evaluate.py
=================
Evaluates the best trained binary classifier model on the validation split.
Computes Accuracy, Precision, Recall, F1-Score and generates a high-quality
Confusion Matrix plot.
"""

import argparse
import json
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

import torch
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

def evaluate_model(args):
    log.info("━━━ Starting Week 2 Evaluation Pipeline ━━━")
    
    weights_path = Path(args.weights)
    metrics_dir = Path("week2/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    
    if not weights_path.exists():
        raise FileNotFoundError(f"Trained model checkpoint not found: {weights_path}. Train the model first!")
        
    # 1. Check Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    log.info("Evaluation device: %s", device)
    
    # 2. Re-create Stratified Split (Same seed to avoid data leakage)
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
    df = pd.read_csv(manifest_path)
    records = df.to_dict(orient="records")
    labels = [r["annotated_label"] for r in records]
    
    _, val_records = train_test_split(
        records,
        test_size=args.val_split,
        stratify=labels,
        random_state=args.seed
    )
    
    log.info("Evaluating on validation split of size: %d", len(val_records))
    
    # 3. Create loader
    _, val_transform = get_transforms()
    val_dataset = WoundDataset(val_records, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # 4. Load Model and Weights
    model = get_model(
        model_name=args.model_name,
        num_classes=2,
        freeze_backbone=True
    )
    
    # Load state dict safely
    log.info("Loading best weights from checkpoint: %s...", weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    
    # 5. Run Inference
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for idx, (images, targets) in enumerate(val_loader, start=1):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())
            
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # 6. Compute Metrics
    class_names = ["normal", "wound"]
    
    # Classification Report
    report_dict = classification_report(
        all_labels, 
        all_preds, 
        target_names=class_names, 
        output_dict=True
    )
    
    accuracy = np.mean(all_preds == all_labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, 
        all_preds, 
        average="binary"
    )
    
    log.info("Accuracy:  %6.2f%%", accuracy * 100)
    log.info("Precision: %6.2f%%", precision * 100)
    log.info("Recall:    %6.2f%%", recall * 100)
    log.info("F1-Score:  %6.2f%%", f1 * 100)
    
    # Save Report to JSON
    eval_report = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "detailed_report": report_dict
    }
    report_json_path = metrics_dir / "evaluation_report.json"
    with open(report_json_path, "w") as f:
        json.dump(eval_report, f, indent=4)
    log.info("Saved evaluation JSON report to %s", report_json_path)
    
    # 7. Print Professional Console Report
    print("\n" + "=" * 60)
    print("           WEEK 2 CLASSIFICATION PERFORMANCE REPORT          ")
    print("=" * 60)
    print(f"  Overall Accuracy :  {accuracy * 100:6.2f}%")
    print(f"  Precision (Wound):  {precision * 100:6.2f}%")
    print(f"  Recall (Wound)   :  {recall * 100:6.2f}%")
    print(f"  F1-Score (Wound) :  {f1 * 100:6.2f}%")
    print("-" * 60)
    print("  Per-Class Metrics Detail:")
    print("  " + f"{'Class':<12} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("  " + "-" * 45)
    for c in class_names:
        p_c = report_dict[c]["precision"] * 100
        r_c = report_dict[c]["recall"] * 100
        f_c = report_dict[c]["f1-score"] * 100
        print("  " + f"{c:<12} | {p_c:8.2f}% | {r_c:8.2f}% | {f_c:8.2f}%")
    print("=" * 60 + "\n")
    
    # 8. Generate and Plot Confusion Matrix
    log.info("Plotting Confusion Matrix...")
    cm = confusion_matrix(all_labels, all_preds)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    # Elegant blue-gradient color mapping
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax, shrink=0.7)
    
    # Show all ticks and label them with class names
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, 
           yticklabels=class_names,
           ylabel="Ground Truth",
           xlabel="Predicted Label")
    
    ax.set_title("Wound Classifier — Confusion Matrix", fontsize=13, fontweight="bold", pad=15)
    
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0, ha="center", fontsize=11)
    plt.setp(ax.get_yticklabels(), fontsize=11)
    
    # Loop over data dimensions and create text annotations.
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight="bold")
            
    fig.tight_layout()
    cm_path = metrics_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()
    log.info("Saved Confusion Matrix figure → %s", cm_path)
    log.info("━━━ Week 2 Evaluation Pipeline Complete ━━━")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Wound Classifier")
    parser.add_argument("--manifest", type=str, default="data/cleaned_week2/annotated_manifest_week2.csv", help="Path to manifest CSV")
    parser.add_argument("--weights", type=str, default="week2/weights/best_classifier.pth", help="Path to trained weights file")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--model-name", type=str, default="mobilenet_v3_large", help="resnet18 or mobilenet_v3_large")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    parsed_args = parser.parse_args()
    
    if not torch.cuda.is_available():
        parsed_args.device = "cpu"
        
    evaluate_model(parsed_args)
