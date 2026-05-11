# Dataset Proposal (Week 1)

## Project Title
Wound Type Classification, Detection, and Segmentation for Lower Limb and Foot Images

## Student
Munees Tariq (2023-CS-32)

## Course
Computer Vision Project (Medical Imaging Track)

## Proposal Date
12-05-2026

## Problem Statement
Manual wound assessment in clinical settings is slow, subjective, and labor intensive. This project proposes a computer vision pipeline to automatically classify wound type, localize wound regions, and segment wound/peri-wound tissue from medical images.

## Dataset Source (Compliant with Course Scope)
- Dataset: Lower Limb and Feet Wound Image Dataset
- Journal: Data in Brief (Elsevier), Vol. 66, 2026
- DOI: 10.1016/j.dib.2026.112730
- Download: https://data.mendeley.com/datasets/hsj38fwnvr/3
- License: CC BY 4.0

This satisfies the requirement to use healthcare/medical imaging data from Data in Brief or Scientific Data.

## Dataset Summary
- Wound images: 2,686
- Healthy (normal) images: 2,757
- Total: 5,443
- Resolution: 331 x 331
- Wound classes (target taxonomy): diabetic, pressure, trauma, venous, surgical, arterial, cellulitis, miscellaneous

## Local Data Layout Used
- data/Nomal/ (healthy images)
- data/wound_main/ (flat wound images)
- data/wound_mask/ (segmentation masks)

## Week 1 Work Completed
1. Dataset cleaning and validation script implemented (`week1/week1_dataset_clean.py`).
2. Deduplication, image validation, resize normalization, and manifest export completed.
3. 20-sample selection pipeline completed with mixed labels from wound and normal pools.
4. Week 1 annotation artifacts prepared (`week1/week1_sample_annotations.csv`).

## Annotation Tool Plan
A custom mini annotator is used/developed for manual wound class assignment and region marking on selected samples, fulfilling the “own annotator” requirement.

## Planned Technical Pipeline (Weeks 2-4)
- Classification: ResNet/ViT for wound-type prediction.
- Detection: YOLOv8 or RT-DETR for wound localization.
- Segmentation: U-Net and/or SAM for wound boundary delineation.
- Fusion: Unified risk/severity dashboard from all model outputs.

## Week 1 Deliverables Checklist
- Dataset proposal: Completed
- Dataset cleaning code: Completed
- Custom annotator code: Included in repository workflow
- 20 sample annotations: Completed

## Risks and Mitigation
- Risk: Flat wound directory lacks class folders.
- Mitigation: Assign temporary `miscellaneous` at cleaning stage; apply manual class labels during annotation.

## Conclusion
The selected dataset and Week 1 pipeline satisfy the course medical-imaging scope and establish a solid base for classification, detection, and segmentation in upcoming weeks.
