<div align="center">
  
# UNIVERSITY OF ENGINEERING AND TECHNOLOGY
### Department of Computer Science

<br>

## COURSE PROJECT PROPOSAL: WEEK 1
### CS-421: Computer Vision (Medical Imaging Track)

<br>

---

# 🩹 Wound Type Classification, Detection & Segmentation
**An Automated Deep Learning Pipeline for Clinical Wound Assessment**

---

<br>

**Submitted By:**
**Name:** Munees Tariq
**Roll No:** 2023-CS-32
**Section:** A

<br>

**Date of Submission:** 12-05-2026

<br>

---

</div>

<div style="page-break-after: always;"></div>

## 1. Project Overview
This project aims to develop a robust Computer Vision pipeline for the automated analysis of lower limb and foot wounds. The system will integrate three core CV tasks:
1.  **Image Classification:** Identifying the wound etiology (8 distinct classes).
2.  **Object Detection:** Localizing the wound area using bounding boxes.
3.  **Semantic Segmentation:** Delineating precise wound boundaries and peri-wound tissue.

The final output will be a "Patient Risk Score Dashboard" that visualizes predictions from all three models to assist clinical staff in triage.

## 2. Dataset Selection (Journal Compliant)
In accordance with the course requirements (Healthcare/Medical Imaging data from *Data in Brief* or *Scientific Data*), the following dataset has been selected:

*   **Dataset Name:** Lower Limb and Feet Wound Image Dataset
*   **Journal:** **Data in Brief** (Elsevier) — Vol. 66, 2026
*   **DOI:** `10.1016/j.dib.2026.112730`
*   **Source:** Mendeley Data Repository
*   **License:** CC BY 4.0 (Open Access)

### 2.1 Dataset Statistics
| Category | Image Count | Description |
| :--- | :--- | :--- |
| **Wound Images** | 2,686 | Pathological cases with various etiologies |
| **Healthy Feet** | 2,757 | Normal control group |
| **Total** | **5,443** | Cleaned and normalized to 331x331 pixels |

### 2.2 Taxonomy (8 Wound Classes)
The dataset includes the following wound types for multi-class classification:
- Diabetic Ulcers
- Pressure Ulcers
- Venous Ulcers
- Arterial Ulcers
- Surgical Wounds
- Traumatic Wounds
- Cellulitis
- Miscellaneous

## 3. Data Cleaning & Pre-processing (Week 1)
A custom automated cleaning pipeline (`week1_dataset_clean.py`) was developed to:
- **Validate Integrity:** Filtered out truncated or corrupted image headers.
- **Deduplication:** Performed MD5 checksum analysis to remove duplicate entries.
- **Normalization:** Standardized all images to a consistent 331x331 RGB format.
- **Stratification:** Performed stratified sampling to select 20 diverse images for initial annotation.

## 4. Custom Annotation Tool (Week 1)
To satisfy the requirement of developing a **"Mini Annotation Tool"**, a custom GUI was built using Python's **Tkinter** library. 

### 4.1 Features
- **Interactive Canvas:** Drawing bounding boxes via mouse drag-and-drop.
- **Multi-Level Labeling:**
    - Wound Type Classification (8 classes).
    - Action Region Segmentation Labels (Wound Bed, Peri-wound, Healthy Margin).
    - Clinical Severity Scoring (0-7 scale).
- **Persistence:** Real-time autosave to JSON to prevent data loss.
- **Exports:** Native support for CSV and COCO-style JSON formats.

## 5. Sample Annotations
A total of **20 sample annotations** have been generated using the custom tool. These include a mix of normal feet and wound images to establish the baseline for the upcoming Detection and Segmentation tasks.

## 6. Project Timeline & Deliverables
| Phase | Task | Status |
| :--- | :--- | :--- |
| **Week 1** | Dataset Proposal, Cleaning, Annotator, 20 Samples | **COMPLETED** |
| **Week 2** | Full Annotation & ResNet Classification | Pending |
| **Week 3** | YOLOv8 Object Detection Training | Pending |
| **Week 4** | U-Net Segmentation & Research Paper | Pending |
| **Week 5** | Final Evaluation & Video Demo | Pending |
