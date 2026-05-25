# Week 3 Execution Plan (Due: 26-05-2026)

This plan is designed so that if every step is executed, Week 3 deliverables will be complete:
- Detection model
- Prediction outputs
- mAP results

## Scope
- Use Week 2 data pipeline outputs as the base.
- Ignore Week 1 code quality (as requested).
- Implement Week 3 detection pipeline in a new `week3/` module and connect outputs to `app/`.

## Required Deliverables (Week 3)
1. Trained object detection model weights.
2. Test/inference predictions saved as images + labels.
3. Evaluation report with mAP (especially `mAP50` and `mAP50-95`).

## Hardware Profile (Applied to Plan)
- GPU: GTX 1650 Ti (4GB VRAM)
- CPU: i7 10th Gen
- RAM: 16GB

For this setup, default training target should be stability first:
- Model: `yolov8n.pt` (start here, not `yolov8s.pt`)
- Image size: `imgsz=512` (drop to `448` if OOM)
- Batch: `batch=8` (drop to `4` if OOM)
- Workers: `workers=2`
- AMP: enabled (`amp=True`)

## 3-Part Execution Plan

## Part 1: Data + Project Setup

1. Create Week 3 structure.
```powershell
mkdir week3
mkdir week3\configs
mkdir week3\data
mkdir week3\scripts
mkdir week3\runs
mkdir week3\results
mkdir week3\results\predictions
```

2. Prepare YOLO dataset format from existing annotations.
- Input: Week 2/annotation outputs or exported annotation files from `app`.
- Output target:
  - `week3\data\images\train`
  - `week3\data\images\val`
  - `week3\data\images\test`
  - `week3\data\labels\train`
  - `week3\data\labels\val`
  - `week3\data\labels\test`
- Required label format per image file:
```txt
class_id x_center y_center width height
```
Normalized to [0,1].

3. Add YOLO dataset config file: `week3\configs\wound_detection.yaml`
```yaml
path: week3/data
train: images/train
val: images/val
test: images/test
names:
  0: wound
  1: foot
  2: limb
  3: peri-wound
  4: healthy skin
```

4. Implement dataset prep script: `week3\scripts\prepare_detection_dataset.py`
- Include split generation (recommended: 70% train / 20% val / 10% test).
- Validate that each image has a matching label file.
- Print per-class object counts before training.

5. Run Part 1 commands.
```powershell
pip install -r requirements.txt
python week3\scripts\prepare_detection_dataset.py
```

Part 1 exit criteria:
- `week3\data\images\{train,val,test}` created
- `week3\data\labels\{train,val,test}` created
- `week3\configs\wound_detection.yaml` valid

## Part 2: Detection Training (GTX 1650 Ti Optimized)

1. Implement training script: `week3\scripts\train_detection.py`
- Use `ultralytics` YOLOv8 (`yolov8n.pt` or `yolov8s.pt`).
- Use this baseline for your machine:
  - `model=yolov8n.pt`
  - `imgsz=512`
  - `epochs=60`
  - `batch=8`
  - `workers=2`
  - `cache=False`
  - `amp=True`
  - `patience=15`
  - `project=week3/runs`
  - `name=wound_detection`

2. Run training.
```powershell
python week3\scripts\train_detection.py
```

3. Fallback settings if CUDA OOM occurs:
- Retry with `batch=4`
- Then retry with `imgsz=448`
- Keep `yolov8n.pt` until baseline mAP is stable

Part 2 exit criteria:
- `week3/runs/wound_detection/weights/best.pt` exists
- training logs and curves are generated

## Part 3: Evaluation, Predictions, and App Integration

1. Implement evaluation script: `week3\scripts\evaluate_detection.py`
- Run YOLO validation on `val` and `test`.
- Save summary metrics to:
  - `week3\results\map_results.json`
  - `week3\results\map_results.csv`
- Must include:
  - `mAP50`
  - `mAP50-95`
  - precision
  - recall

2. Generate prediction outputs.
- Implement `week3\scripts\predict_detection.py`
- Run inference on test images and save:
  - plotted images to `week3\results\predictions\images\`
  - txt labels/confidences to `week3\results\predictions\labels\`

3. Create execution wrapper: `week3\run_week3_pipeline.py`
- Order:
  1. Data prep
  2. Train
  3. Evaluate
  4. Predict
- This becomes the single Week 3 command:
```powershell
python week3\run_week3_pipeline.py
```

4. Connect Week 3 outputs to app UI (`app/src/App.jsx`).
- Replace Week 3 placeholder section with:
  - model info card (best weights path)
  - metric cards (`mAP50`, `mAP50-95`, precision, recall)
  - prediction gallery from `week3/results/predictions/images`
- Keep Week 2 unchanged.

5. Final Week 3 packaging.
- Ensure these exist:
  - `week3/runs/wound_detection/weights/best.pt`
  - `week3/results/map_results.csv`
  - `week3/results/map_results.json`
  - `week3/results/predictions/images/*`
- Add a short `week3/README.md` with exact commands used and final metrics.

## Acceptance Checklist (Definition of Done)
- [ ] Training completes without runtime errors.
- [ ] Best detection weights file is produced.
- [ ] Test predictions are exported as visual outputs.
- [ ] mAP metrics are generated and saved in CSV/JSON.
- [ ] App Week 3 tab displays real Week 3 outputs (not placeholder text).
- [ ] Week 3 README documents reproducible commands.

## Recommended Command Sequence
```powershell
pip install -r requirements.txt
python week3\scripts\prepare_detection_dataset.py
python week3\scripts\train_detection.py
python week3\scripts\evaluate_detection.py
python week3\scripts\predict_detection.py
```

## Risk Controls
- If GPU memory fails, reduce `batch` and/or `imgsz`.
- If dataset is imbalanced, monitor per-class AP and oversample underrepresented classes.
- If mAP is weak, increase epochs and apply augmentation tuning before changing architecture.
