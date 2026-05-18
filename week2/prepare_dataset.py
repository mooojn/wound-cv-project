"""
week2/prepare_dataset.py
========================
Prepares the Week 2 annotated dataset of exactly 1,500 images:
- 1,000 wound images selected directly from data/wound_main/ (labeled "wound")
- 500 randomly selected normal images from data/Nomal/ (labeled "normal")
- Crops and resizes all 1,500 images to 331x331 using Lanczos interpolation.
- Saves them cleanly under data/cleaned_week2/
- Generates the final target manifest: data/cleaned_week2/annotated_manifest_week2.csv
"""

import os
import shutil
import random
import csv
import logging
from pathlib import Path
from PIL import Image

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RAW_DIR = Path("data")
WOUND_MAIN = RAW_DIR / "wound_main"
NORMAL_DIR = RAW_DIR / "Nomal"
CLEANED_W2_DIR = RAW_DIR / "cleaned_week2"
TARGET_MANIFEST = CLEANED_W2_DIR / "annotated_manifest_week2.csv"

TARGET_SIZE = (331, 331)
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# Main Execution Pipeline
# ---------------------------------------------------------------------------
def main():
    log.info("━━━ Week 2 — Preparing 1,500 Binary Classification Images ━━━")
    
    # 1. Verify Raw Directories
    if not WOUND_MAIN.exists():
        log.error("Wound directory not found: %s", WOUND_MAIN)
        return
    if not NORMAL_DIR.exists():
        log.error("Normal healthy directory not found: %s", NORMAL_DIR)
        return
        
    # Re-create/clean output directory
    if CLEANED_W2_DIR.exists():
        log.info("Cleaning existing week2 directory: %s", CLEANED_W2_DIR)
        shutil.rmtree(CLEANED_W2_DIR)
    CLEANED_W2_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Gather All Wound & Normal Image Candidates
    image_exts = {".jpg", ".jpeg", ".png", ".bmp"}
    all_wounds = [
        p for p in sorted(WOUND_MAIN.glob("*")) 
        if p.suffix.lower() in image_exts
    ]
    all_normals = [
        p for p in sorted(NORMAL_DIR.glob("*")) 
        if p.suffix.lower() in image_exts
    ]
    
    log.info("Found %d raw wound images and %d raw healthy normal images.", len(all_wounds), len(all_normals))
    
    # Select exactly 1,000 wounds
    if len(all_wounds) < 1000:
        log.warning("Only %d wound images available. Selecting all.", len(all_wounds))
        selected_wounds = all_wounds
    else:
        selected_wounds = random.sample(all_wounds, 1000)
        
    # Select exactly 500 normal healthy feet images
    if len(all_normals) < 500:
        log.warning("Only %d normal images available. Selecting all.", len(all_normals))
        selected_normals = all_normals
    else:
        selected_normals = random.sample(all_normals, 500)
        
    log.info("Selected %d wound images and %d normal images for Week 2.", len(selected_wounds), len(selected_normals))
    
    # 3. Process, Crop/Resize, and Copy to cleaned_week2/
    log.info("Processing, resizing to %dx%d, and saving images...", TARGET_SIZE[0], TARGET_SIZE[1])
    
    cleaned_records = []
    
    # Helper to clean and resize
    def process_and_save(src_path: Path, label_prefix: str) -> Path:
        dst_name = f"{label_prefix}_{src_path.name}"
        dst_path = CLEANED_W2_DIR / dst_name
        with Image.open(src_path) as img:
            img = img.convert("RGB")
            if img.size != TARGET_SIZE:
                img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)
            img.save(dst_path, "JPEG", quality=95)
        return dst_path

    # Process Wounds
    log.info("Processing 1,000 Wounded images...")
    processed_wounds_count = 0
    for idx, path in enumerate(selected_wounds, start=1):
        try:
            dst = process_and_save(path, f"wound_{idx:04d}")
            cleaned_records.append({
                "original_path": str(path),
                "cleaned_path": str(dst),
                "annotated_label": "wound",
                "split": "wound"
            })
            processed_wounds_count += 1
            if idx % 200 == 0 or idx == len(selected_wounds):
                log.info("Resized %d / %d wound images...", idx, len(selected_wounds))
        except Exception as e:
            log.warning("Skip corrupted wound file %s: %s", path.name, e)
            
    # Process Normals
    log.info("Processing 500 Healthy feet images...")
    processed_normals_count = 0
    for idx, path in enumerate(selected_normals, start=1):
        try:
            dst = process_and_save(path, f"normal_{idx:04d}")
            cleaned_records.append({
                "original_path": str(path),
                "cleaned_path": str(dst),
                "annotated_label": "normal",
                "split": "normal"
            })
            processed_normals_count += 1
            if idx % 100 == 0 or idx == len(selected_normals):
                log.info("Resized %d / %d normal images...", idx, len(selected_normals))
        except Exception as e:
            log.warning("Skip corrupted normal file %s: %s", path.name, e)
            
    log.info("Preprocessed successfully: %d wounds, %d normals.", processed_wounds_count, processed_normals_count)
    
    # 4. Save Manifest CSV
    log.info("Writing target manifest to %s...", TARGET_MANIFEST)
    with open(TARGET_MANIFEST, "w", newline="") as f:
        fieldnames = ["original_path", "cleaned_path", "annotated_label", "split"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cleaned_records)
        
    # 5. Print Statistics
    total = len(cleaned_records)
    by_label = {}
    for r in cleaned_records:
        lbl = r["annotated_label"]
        by_label[lbl] = by_label.get(lbl, 0) + 1
        
    print("\n" + "=" * 60)
    print(f"  WEEK 2 ANNOTATED DATASET STATISTICS  (total: {total} images)")
    print("=" * 60)
    for label in sorted(by_label):
        bar = "#" * (by_label[label] * 30 // max(by_label.values()))
        print(f"  {label:<18} {by_label[label]:>5}  {bar}")
    print("=" * 60 + "\n")
    
    log.info("━━━ Week 2 Dataset Preparation Complete ━━━")
    log.info("Manifest saved → %s", TARGET_MANIFEST)

if __name__ == "__main__":
    main()
