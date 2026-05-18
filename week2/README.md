# Week 2 - Clinical Image Classification Pipeline

This module implements the Week 2 classification task for clinical foot scans. The model predicts two classes:
- `wound`
- `normal`

Manual annotation and annotator-tool work is covered in Week 1.

## What Was Implemented in Week 2

- Built a complete binary medical image classification pipeline for wound screening.
- Created a cleaned classification dataset from raw sources and exported a labeled manifest CSV.
- Trained a transfer-learning classifier and saved the best checkpoint.
- Evaluated performance with standard classification metrics and confusion matrix visualization.
- Integrated inference support via `app_server.py` for local testing.

## Technical Details

- Primary model used: `MobileNetV3-Large` (transfer learning backbone).
- Alternative supported in code: `ResNet18` (selectable through `--model-name`).
- Classification head output: `2` classes (`normal`, `wound`).
- Backbone strategy: frozen feature extractor during training for faster and stable CPU training.
- Input image size: `331 x 331` pixels.
- Optimizer: `AdamW` (default `lr=0.003`, `weight_decay=1e-4`).
- LR scheduler: Cosine Annealing.
- Loss function: `CrossEntropyLoss`.
- Data split: stratified `80/20` train/validation.
- Data augmentation (train): horizontal/vertical flips, random rotation, color jitter.
- Normalization: ImageNet mean/std.

## Directory Structure

```text
week2/
|-- metrics/                 # Training/evaluation outputs
|-- weights/                 # Best saved model checkpoint
|-- app_server.py            # Flask inference server
|-- dataset.py               # Dataset class + transforms
|-- evaluate.py              # Evaluation script + confusion matrix
|-- model.py                 # Model factory
|-- prepare_dataset.py       # Dataset preparation script
|-- run_pipeline.py          # End-to-end runner (prepare + train + eval)
|-- train.py                 # Training script
`-- README.md
```

## Dataset (Week 2)

- Total samples: `1500`
- `wound`: `1000`
- `normal`: `500`
- Prepared output:
  - `data/cleaned_week2/`
  - `data/cleaned_week2/annotated_manifest_week2.csv`

## Training and Evaluation Outputs

- Model checkpoint: `week2/weights/best_classifier.pth`
- Training logs/plots:
  - `week2/metrics/training_history.csv`
  - `week2/metrics/training_curves.png`
- Evaluation outputs:
  - `week2/metrics/evaluation_report.json`
  - `week2/metrics/confusion_matrix.png`

## Run Commands

1. Prepare dataset:
```bash
python week2/prepare_dataset.py
```

2. Train classifier:
```bash
python week2/train.py --epochs 10 --batch-size 32 --device cpu
```

3. Evaluate model:
```bash
python week2/evaluate.py --device cpu
```

4. Run full Week 2 pipeline:
```bash
python week2/run_pipeline.py
```

## App Launch

- Auto launch:
  - Run `start_week2_app.bat` from project root
- Manual launch:
  - Backend: `python week2/app_server.py`
  - Frontend: `cd app && npm run dev`
