# Week 2 — Clinical Image Classification Pipeline

This repository implements a lightweight, high-performance deep transfer learning workflow for the binary screening of clinical foot scans. The model classifies scans into two categories: **`Wound`** (representing active skin lesions, tissue breakdowns, and ulcerations) and **`Normal`** (representing intact, healthy skin control tissues).

---

## 📂 Directory Architecture

```markdown
week2/
├── weights/
│   └── best_classifier.pth     # Trained MobileNetV3-Large model weights checkpoint
├── metrics/
│   ├── confusion_matrix.png    # Graphical breakdown of classification sensitivity
│   └── training_curves.png     # Loss convergence & validation accuracy tracking curves
├── app_server.py                # Flask CPU inference API server (Port 5000)
├── dataset.py                  # PyTorch custom dataset class & data loader pipeline
├── train.py                    # Transfer learning training script with augmentations
├── evaluate.py                 # Stratified model evaluation & metric generation script
├── prepare_dataset.py          # Automates raw dataset downloading & preprocessing
└── README.md                   # This documentation file
```

---

## 🔬 Dataset Curation & Preprocessing

The classification head is trained on a curated collection of **1,500 clinical scans** with a stratified distribution:
1. **Positive Class (`wound`):** 1,000 images representing distressed clinical states, active tissue breakdowns, and active foot wounds.
2. **Negative Class (`normal`):** 500 images representing healthy controls with completely intact tissue.

### Curation Pipeline:
* **Lanczos Interpolation Resizing**: Images are cropped and resized to a standardized dimension of **`331x331` pixels**. We utilize Lanczos resizing (8x8 pixel neighborhood) to preserve high-frequency spatial details (granular borders, dry tissue flakes, skin textures) without introducing pixelation or interpolation blur.
* **Stratified Split**: The preprocessed corpus is partitioned into an **80% training set (1,200 images)** and a **20% validation set (300 images)**. A stratified split is applied to guarantee that the 2:1 positive-to-negative class distribution is maintained perfectly across both partitions, preventing training bias.

---

## 🧠 Neural Network & Transfer Learning Methodology

To support real-time clinical screening on edge devices and standard consumer CPU systems, the pipeline implements **Transfer Learning** using a pre-trained **`MobileNetV3-Large`** backbone.

### Architecture Configuration:
* **Frozen Feature Extractor**: The pre-trained ImageNet parameters of the backbone are frozen to preserve generalized visual representations and eliminate gradient dissipation during CPU execution:
  $$\theta_{\text{backbone}} \leftarrow \text{ImageNet Weights (frozen)}$$
* **Custom Classifier Head**: A custom linear projection layer maps the 960 features from the MobileNetV3 bottleneck directly to the 2 binary diagnostic classes:
  $$\text{Logits} = W \cdot x + b, \quad W \in \mathbb{R}^{2 \times 960}, \ b \in \mathbb{R}^2$$

### Hyperparameters & Regularization:
* **Data Augmentations**: Random horizontal/vertical flips ($p=0.5$), random rotations (up to 20°), and color jittering (brightness, contrast, saturation) are applied to the training split.
* **Optimization**: `AdamW` optimizer ($\eta = 3 \times 10^{-3}$, weight decay $= 10^{-4}$) scheduled via a Cosine Annealing learning rate policy.
* **Loss Function**: Binary Cross-Entropy.

---

## 📈 Model Convergence & Evaluation Metrics

The classifier converges rapidly, reaching a training loss of **0.0086** by epoch 5. Evaluating the frozen weights checkpoint on the independent, stratified validation split of 300 clinical samples yields near-perfect diagnostic capability:

| Metric | Screening Score | Clinical Interpretation |
| :--- | :---: | :--- |
| **Overall Accuracy** | **99.67%** | 299 out of 300 validation splits correctly categorized. |
| **Sensitivity / Recall** | **100.00%** | **0% False Negatives.** The model caught every single wound without a single missed diagnosis. |
| **Precision** | **99.50%** | **0.5% False Positives.** Only 1 healthy control scan flagged as distressed. |
| **F1-Score** | **99.75%** | Highly stable harmonic balance showing high generalization. |

### Confusion Matrix Verification:
* **True Negatives (Normal Control):** 99
* **False Positives (False Alarm):** 1
* **False Negatives (Missed Wound):** 0
* **True Positives (Active Wound):** 200

---

## 💻 Web App & Interactive Inference Suite

The frontend application provides a stunning clinical interface to run live local tests and directory inspections:

1. **Live Classifier Sandbox**:
   - Drag & drop or upload medical scans for instant, real-time CPU model predictions.
   - Shows interactive confidence sliders, standardized diagnostic outcome tags (`Wound` / `Normal`), and rich clinician recommendations based on model outcomes.

2. **1-Click Dataset Quick Loaders**:
   - Quick-load random controls (`data/Nomal`) or active scans (`data/wound_main`) directly from your clinical workspace in a single click, instantly running CPU inference.

3. **Folder Batch Inference Inspector**:
   - Scans any selected local directory, immediately loading base64 preview thumbnails to compile an interactive carousel.
   - Runs progressive background classification queues on the Flask server, dynamically updating individual image slides with animated loading skeletons as results arrive.

4. **Model Performance Visualizations**:
   - Both the **Confusion Matrix** and **Training Loss Convergence Curves** are rendered side-by-side in full resolution at the bottom of the dashboard for complete diagnostic reporting.

---

## 🚀 How To Run The Application

### Prerequisites:
Make sure you have your dependencies installed inside a virtual environment:
```bash
pip install -r requirements.txt
```

### 1-Click Dual App Launch:
Double-click the **`start_week2_app.bat`** file located in the project root. This orchestrates:
1. Running the **Flask API server** (`week2/app_server.py`) in the background on port `5000`.
2. Launching the **React Vite development server** on port `5173`.
3. Automatically opening your default web browser to the interactive dashboard.

### Manual Launching:
If you prefer running the processes manually, launch two separate terminal shells:

**Terminal 1 (Flask API Server):**
```bash
python week2/app_server.py
```

**Terminal 2 (React Frontend Web App):**
```bash
cd app
npm run dev
```
Navigate to `http://localhost:5173/` in your browser.
