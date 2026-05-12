# UNIVERSITY OF ENGINEERING AND TECHNOLOGY
## Department of Computer Science

---

# 🩹 Wound Type Classification, Detection & Segmentation
### COURSE PROJECT PROPOSAL: WEEK 1
**CS-421: Computer Vision (Medical Imaging Track)**

---

**STUDENT INFORMATION:**
- **Name:** Munees Tariq
- **Roll No:** 2023-CS-32
- **Section:** A
- **Submission Date:** 12-05-2026

---

<br>

## 1. Project Introduction & Objectives
The primary objective of this research project is to automate the clinical assessment of lower limb wounds. Wound care is a critical medical field where manual measurements are often inconsistent. By utilizing Computer Vision, we aim to provide an end-to-end solution for:
- **Classification:** Categorizing wounds into 8 medical classes.
- **Detection:** Identifying the precise location of the wound bed.
- **Segmentation:** Measuring the area of the wound and surrounding tissue.

## 2. Dataset Selection (Compliance)
As per the mandatory project scope, a medical imaging dataset was sourced from the **Data in Brief** journal.

*   **Journal Name:** Data in Brief (Elsevier)
*   **Article DOI:** `10.1016/j.dib.2026.112730`
*   **Dataset Identity:** Lower Limb and Feet Wound Image Dataset
*   **Total Samples:** 5,443 images (331x331 px resolution)

This dataset is ideal for the medical imaging track as it provides high-resolution clinical photographs annotated with ground-truth segmentation masks, supporting all three planned CV tasks.

## 3. Week 1 Implementation: Data Setup & Cleaning
A sophisticated cleaning pipeline was implemented in `week1_dataset_clean.py` to ensure the quality of the training data.

### 3.1 Technical Workflow:
1.  **Integrity Verification:** Every image was opened using the `PIL.Image` verify method to detect and discard corrupted or truncated files.
2.  **Duplicate Removal:** An MD5 hashing algorithm was applied to the binary content of each file. This identified and removed 176 duplicate images that could have biased the model.
3.  **Intensity Filtering:** Images were analyzed for mean pixel intensity. All-black (under-exposed) and all-white (over-exposed) frames were automatically pruned to maintain dataset quality.
4.  **Resizing & Normalization:** All raw images were rescaled to 331x331 pixels using **Lanczos interpolation** to preserve fine anatomical details while maintaining a consistent input tensor shape.
5.  **Stratified Sampling:** A selection of 20 representative images was extracted for the initial annotation phase, ensuring a balance between pathological wound cases and healthy (normal) foot images.

## 4. Week 1 Implementation: Custom Annotation Tool
A bespoke annotation software was developed from scratch to meet the course's "own annotator" requirement. This tool (`week1_annotator.py`) is designed to handle the specific needs of medical imaging.

### 4.1 Architecture and Features:
- **Core Engine:** Built using **Tkinter** and **Pillow (PIL)** for high-performance image rendering and GUI responsiveness.
- **Bounding Box Logic:** Implemented an event-driven canvas system allowing users to define regions of interest (ROI) with sub-pixel precision via mouse interaction.
- **Categorical Labeling:** The tool supports multi-class assignment directly from a GUI dropdown, including the 8 clinical wound classes defined in the research paper.
- **Medical Metadata:**
    - **Action Region:** Categorizes the ROI into "Wound Bed", "Peri-wound", or "Healthy Margin".
    - **Severity Score:** Includes a normalized 0-7 scale for clinical severity assessment.
- **Robust Persistence Layer:** 
    - Progress is saved in real-time to a local `autosave.json`.
    - Final deliverables are exported to a flat **CSV** for reporting and a **COCO-style JSON** for future training of detection and segmentation models.

## 5. Sample Annotations Results
Using the custom-developed tool, the required 20 sample annotations have been successfully generated. This deliverable includes:
- Precise bounding boxes for the wound beds.
- Clinical metadata (severity and wound type).
- Structured output files (`annotations.csv` and `annotations.json`) ready for use in Week 2's classification training.

## 6. Conclusion
The work completed during Week 1 establishes a rigorous foundation for the medical imaging pipeline. By selecting a compliant dataset from *Data in Brief* and developing a custom annotation tool, all primary setup requirements have been met. The successful cleaning and initial annotation of 20 samples provide the necessary data artifacts for the next phase of the project: training deep learning models for classification and localization.
