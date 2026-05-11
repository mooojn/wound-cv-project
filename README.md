# 🩹 Wound Type Classification, Detection & Segmentation
### A Medical Computer Vision Pipeline for Automated Wound Assessment

> **Course Project:** Computer Vision 

---

## 📋 Project Overview

This project builds an **end-to-end medical computer vision pipeline** that automatically analyzes lower limb and foot wound images to:

1. **Classify** the wound type (8 categories) using a ResNet/ViT backbone
2. **Detect** the wound region with bounding boxes using YOLOv8
3. **Segment** the wound area (including peri-wound tissue) using U-Net / SAM
4. **Output** a fused wound severity dashboard combining all three predictions

The clinical motivation is real: wound care nurses spend significant time manually categorizing and measuring wounds. This pipeline automates that process using computer vision, supporting faster triage and treatment planning.

---

## 🗄️ Dataset

| Field | Details |
|---|---|
| **Name** | Lower Limb and Feet Wound Image Dataset |
| **Journal** | **Data in Brief** (Elsevier) — Vol. 66, 2026 |
| **DOI** | `10.1016/j.dib.2026.112730` |
| **License** | CC BY 4.0 (Open Access) |

### 🔗 Links

- 📄 **Journal Paper:** https://www.sciencedirect.com/science/article/pii/S2352340926002830
- 💾 **Dataset Download:** https://data.mendeley.com/datasets/hsj38fwnvr/3

### Dataset Statistics

| Split | Count | Description |
|---|---|---|
| Wound images | 2,686 | Annotated with segmentation masks |
| Healthy feet | 2,757 | Normal class (negative control) |
| **Total** | **5,443** | 331×331 px, JPG format |

### Wound Classes (8 types)

```
diabetic  ·  pressure  ·  trauma  ·  venous  ·  surgical  ·  arterial  ·  cellulitis  ·  miscellaneous
```

## 🗓️ Project Timeline

| Week | Due Date | Tasks | Key Deliverables |
|---|---|---|---|
| **Week 1** | 12-05-2026 | Dataset selection, cleaning, annotator development | Dataset proposal, annotator code, 20 sample annotations |
| **Week 2** | 19-05-2026 | Full annotation + classification training | Annotated dataset, ResNet classifier, metrics |
| **Week 3** | 26-05-2026 | Object detection training & testing | YOLOv8 model, mAP results, prediction outputs |
| **Week 4** | 02-06-2026 | Segmentation + paper writing + video | U-Net model, research paper, video demo |
| **Week 5** | 09-06-2026 | Final evaluation + video recording | Final package, video demo |

---

## 🏗️ Project Structure

```
wound-cv-project/
│
├── README.md                        ← You are here
│
├── data/                            ← All data (gitignored, download separately)
│   ├── raw/
│   ├── cleaned/
│   └── samples/
│
├── week1/
│   ├── week1_dataset_clean.py       ← Dataset cleaning & sample selection
│   ├── week1_annotator.py           ← Custom Tkinter annotation tool
│   └── annotations/
│       ├── annotations.csv          ← 20 sample annotations (CSV)
│       └── annotations.json         ← 20 sample annotations (JSON/COCO-style)
│
├── week2/
│   ├── week2_classification.py      ← ResNet/ViT training script
│   └── results/
│       └── metrics.csv
│
├── week3/
│   ├── week3_detection.py           ← YOLOv8 training & evaluation
│   └── results/
│       └── map_results.csv
│
├── week4/
│   ├── week4_segmentation.py        ← U-Net / SAM segmentation
│   ├── paper/
│   │   └── research_paper.pdf
│   └── demo/
│       └── video_demo.mp4
│
└── requirements.txt
```

---

## ⚙️ Setup & Installation

### 1. Clone the repository

```bash
git clone https://github.com/mooojn/wound-cv-project.git
cd wound-cv-project
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Go to: https://data.mendeley.com/datasets/hsj38fwnvr/3

Download all files and place them in `data/raw/` with this structure:
```
data/raw/Normal/
data/raw/Wound_Main/
data/raw/Wound_Masked/
```

### 4. Run Week 1 pipeline

```bash
# Step 1: Clean dataset and select 20 annotation samples
python week1/week1_dataset_clean.py

# Step 2: Launch the annotation tool
python week1/week1_annotator.py --images data/samples
```

---

## 🧰 Week 1 — Custom Annotation Tool

A fully custom **Tkinter-based GUI annotator** was developed (no external annotation tools used).

### Features
- Draw bounding boxes on wound regions with mouse drag
- Label each box with a wound Action Region (wound bed, peri-wound, healthy margin)
- Assign overall wound type label (8 classes)
- Assign severity score (0–7 scale)
- Add clinical notes per image
- Export annotations to **CSV** and **JSON (COCO-style)**
- Persistent save — resume annotation across sessions

### Running the annotator

```bash
python week1/week1_annotator.py --images data/samples
```

**Controls:**
- `Draw` — Click and drag on canvas to draw a bounding box
- `Save & Next` — Save current annotation and move to next image
- `Export CSV` — Save all annotations to `annotations.csv`
- `Export JSON` — Save all annotations to `annotations.json`

---


## 📦 Requirements

```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0        # YOLOv8
Pillow>=9.0.0
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
opencv-python>=4.7.0
segmentation-models-pytorch>=0.3.0   # U-Net
```

Install all:
```bash
pip install torch torchvision ultralytics pillow pandas numpy scikit-learn matplotlib opencv-python segmentation-models-pytorch
```

## 👤 Author

Munees Tariq
Department of Computer Science
University of Engineering and Technology
2023-CS-32

---

> *Dataset sourced from Data in Brief journal (Elsevier) — DOI: 10.1016/j.dib.2026.112730*
> *Project submitted for Computer Vision course, 2026*