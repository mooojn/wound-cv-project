# Week 3 - YOLOv8 Wound Object Detection

Welcome to the **Week 3 Object Detection** module! This section implements real-time localized wound bounding box detection using an optimized **YOLOv8** workflow configured for GPU and CPU environments.

---

## 📂 Module Structure

The object detection pipeline is organized within the `week3/` directory:

```text
week3/
├── configs/
│   └── wound_detection.yaml      # YOLOv8 dataset structure configuration
├── results/
│   ├── map_results.csv           # Evaluation metrics in CSV format
│   ├── map_results.json          # Evaluation metrics in JSON format
│   └── part1_dataset_stats.json  # Pre-training dataset splits statistics
├── scripts/
│   ├── prepare_detection_dataset.py  # Splits annotations and images (70% train / 20% val / 10% test)
│   ├── train_detection.py            # Model training routine using YOLOv8
│   ├── evaluate_detection.py         # Validation and testing metrics generator
│   ├── predict_detection.py          # Visual verification & bounding box overlay generator
│   └── export_week3_for_app.py       # Package outputs for the React front-end
├── app_server.py                 # Flask REST API server for Week 3 model inference
├── README.md                     # Documentation & reproducible instructions (this file)
└── run_week3_pipeline.py         # End-to-end execution wrapper script
```

---

## ⚙️ Hardware Profile & Hyperparameters
Optimized to execute reliably on workstations (e.g. GTX 1650 Ti GPU or standard multicore CPU):
* **Model Backbone:** `yolov8n.pt` (Ultralytics YOLOv8 Nano)
* **Image Size (`imgsz`):** `512px` (optimal resolution/performance ratio)
* **Batch Size:** `8` (reduced memory footprint to prevent CUDA OOM)
* **AMP (Automatic Mixed Precision):** Enabled (`amp=True`)
* **Patience:** `15` (early stopping criterion)

---

## 🚀 Execution Instructions

### 1. Environment Verification
First, verify that your dependencies and the base YOLOv8 model are ready:
```bash
pip install -r requirements.txt
```

### 2. End-to-End Pipeline Execution
Run the entire dataset preparation, training, evaluation, and test prediction sequence with a single command:
```bash
python week3/run_week3_pipeline.py
```

### 3. Step-by-Step Execution (Manual Mode)

If you prefer to run the pipeline stages individually:

* **Step A: Prepare Dataset & Labels**
  Converts annotations to normalized YOLO format `[class_id x_center y_center width height]` and splits data into splits:
  ```bash
  python week3/scripts/prepare_detection_dataset.py
  ```

* **Step B: Train YOLOv8 Model**
  Initiates transfer learning from `yolov8n.pt` using local data configuration:
  ```bash
  python week3/scripts/train_detection.py --device auto --name wound_detection_cpu
  ```

* **Step C: Evaluate Performance & Metrics**
  Calculates precision, recall, `mAP50`, and `mAP50-95` on test splits:
  ```bash
  python week3/scripts/evaluate_detection.py --weights week3/runs/wound_detection_cpu/weights/best.pt --device auto
  ```

* **Step D: Run Visual Predictions**
  Infers on test subset and saves predicted bounding boxes:
  ```bash
  python week3/scripts/predict_detection.py --weights week3/runs/wound_detection_cpu/weights/best.pt --device auto
  ```

* **Step E: Export Results for App Integration**
  Packages evaluation summary stats and prediction outputs into React public folder assets:
  ```bash
  python week3/scripts/export_week3_for_app.py
  ```

---

## ⚡ Live Bounding-Box Inference & REST API

To enable live patient image upload bounding box inference in your React front-end, start the local REST server:

```bash
python week3/app_server.py
```
* **Host Address:** `http://127.0.0.1:5001`
* **Features:** Loads the local custom model weights, services live `/week3/detect` inference POST requests, and delivers dynamic charts/predictions directly to the web dashboard.
