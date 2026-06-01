# Week 4 - U-Net Wound Segmentation Pipeline

This module implements the **Week 4 Segmentation** task for clinical foot wound images.  
The model learns to produce a pixel-level binary mask that precisely delineates wound tissue from healthy skin.

---

## 🔬 What Was Implemented in Week 4

- Paired **1000 wound images** (`data/wound_main/`) with **1000 binary masks** (`data/wound_mask/`).
- Built a complete segmentation pipeline: dataset prep → training → evaluation → visualisation → Flask API.
- Trained a **U-Net** model with a **ResNet-34 ImageNet encoder** using `segmentation-models-pytorch`.
- Used a **Combined Loss** (50% BCE + 50% Dice Loss) for robust training on potentially imbalanced masks.
- Evaluated with **IoU, Dice Coefficient, Pixel Accuracy, Precision, and Recall** on a held-out test split.
- Saved 4-panel prediction visualisations (Original | GT Mask | Predicted Mask | Overlay).
- Served live inference via a **Flask REST API** on port `5002`.

---

## ⚙️ Technical Details

| Component         | Choice                                        |
|-------------------|-----------------------------------------------|
| Architecture      | `U-Net`                                       |
| Encoder Backbone  | `ResNet-34` (pretrained ImageNet)             |
| Loss Function     | `50% BCEWithLogitsLoss + 50% Dice Loss`       |
| Optimiser         | `AdamW` (lr=1e-3, weight_decay=1e-4)          |
| LR Scheduler      | `CosineAnnealingLR`                           |
| Input Image Size  | `256 × 256` px                                |
| Mask Type         | Binary (wound=1, background=0)                |
| Data Split        | 70% train / 15% val / 15% test                |
| Augmentation      | H-flip, V-flip, 90° rotation, colour jitter  |
| Normalisation     | ImageNet mean/std (consistent with Week 2/3)  |
| Inference Threshold | 0.5 (sigmoid activation)                   |

---

## 📂 Directory Structure

```text
week4/
├── weights/
│   └── best_seg_model.pth          # Best checkpoint (saved by highest Val IoU)
├── metrics/
│   ├── training_history.csv        # Loss / IoU / Dice per epoch
│   ├── training_curves.png         # Loss, IoU, Dice curves (3 panels)
│   ├── segmentation_evaluation.json # Test-set metrics report
│   └── evaluation_metrics_chart.png # Bar chart of test metrics
├── predictions/
│   ├── sample_XXXX.png             # Per-sample 4-panel visualisation
│   └── prediction_grid.png         # Combined grid overview
├── app_server.py                   # Flask REST API  (port 5002)
├── dataset.py                      # SegmentationDataset + augmentation
├── evaluate.py                     # Test-set evaluation script
├── model.py                        # U-Net model factory (smp)
├── predict.py                      # Visualisation / prediction script
├── prepare_segmentation_dataset.py # Dataset preparation + manifest CSV
├── run_week4_pipeline.py           # End-to-end pipeline runner
└── README.md
```

---

## 📊 Dataset

| Detail          | Value                                |
|-----------------|--------------------------------------|
| Total pairs     | 1000 (image ↔ mask)                 |
| Train split     | ~700 samples (70%)                  |
| Val split       | ~150 samples (15%)                  |
| Test split      | ~150 samples (15%)                  |
| Images dir      | `data/wound_main/`                  |
| Masks dir       | `data/wound_mask/`                  |
| Manifest CSV    | `data/cleaned_week4/segmentation_manifest.csv` |

---

## 🚀 Run Commands

### End-to-End Pipeline (recommended)
```bash
python week4/run_week4_pipeline.py
```

### Step-by-step

**Step A — Prepare dataset manifest:**
```bash
python week4/prepare_segmentation_dataset.py
```

**Step B — Train segmentation model:**
```bash
python week4/train.py --epochs 25 --batch-size 8 --device cuda
```

**Step C — Evaluate on test set:**
```bash
python week4/evaluate.py --device cuda
```

**Step D — Generate prediction visualisations:**
```bash
python week4/predict.py --num-samples 15 --device cuda
```

**CPU fallback (no GPU):**
```bash
python week4/run_week4_pipeline.py --device cpu --epochs 15 --batch-size 4
```

---

## ⚡ Flask API — Live Inference

Start the server:
```bash
python week4/app_server.py
# → http://127.0.0.1:5002
```

### Endpoints

| Method | Endpoint           | Description                              |
|--------|--------------------|------------------------------------------|
| POST   | `/week4/segment`   | Upload an image → get overlay + metrics  |
| GET    | `/week4/results`   | Fetch latest test-set evaluation metrics |
| GET    | `/week4/health`    | Server + weights health check            |

**Example POST request:**
```bash
curl -X POST http://127.0.0.1:5002/week4/segment \
     -F "file=@your_wound_image.jpg"
```

**Response fields:**
```json
{
  "status": "success",
  "wound_coverage": 23.45,
  "mean_confidence": 78.12,
  "overlay_b64": "<base64-encoded PNG>",
  "mask_b64":    "<base64-encoded PNG>"
}
```

---

## 📈 Evaluation Metrics

Computed pixel-by-pixel on the held-out **test split**:

| Metric           | Formula                                            |
|------------------|----------------------------------------------------|
| **IoU** (Jaccard)| TP / (TP + FP + FN)                               |
| **Dice** (F1)    | 2·TP / (2·TP + FP + FN)                           |
| **Pixel Accuracy**| (TP + TN) / Total pixels                         |
| **Precision**    | TP / (TP + FP)                                    |
| **Recall**       | TP / (TP + FN)                                    |
