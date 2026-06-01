"""
week4/prepare_segmentation_dataset.py
======================================
Scans data/wound_main/ (images) and data/wound_mask/ (binary masks),
pairs them by index, performs a stratified 70/15/15 train/val/test split,
and writes a manifest CSV for the segmentation pipeline.

Output:
    data/cleaned_week4/segmentation_manifest.csv
    data/cleaned_week4/dataset_stats.json
"""

import json
import logging
import re
from pathlib import Path
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
IMAGES_DIR   = ROOT / "data" / "wound_main"
MASKS_DIR    = ROOT / "data" / "wound_mask"
OUT_DIR      = ROOT / "data" / "cleaned_week4"
MANIFEST_CSV = OUT_DIR / "segmentation_manifest.csv"
STATS_JSON   = OUT_DIR / "dataset_stats.json"


def extract_index(filename: str) -> int:
    """Extract the numeric index from filenames like wound_main-0001.jpg."""
    match = re.search(r"-(\d+)\.", filename)
    return int(match.group(1)) if match else -1


def main() -> None:
    log.info("━━━ Week 4 — Segmentation Dataset Preparation ━━━")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Collect and pair images ↔ masks by numeric index
    image_files = sorted(IMAGES_DIR.glob("*.jpg"), key=lambda p: extract_index(p.name))
    mask_files  = sorted(MASKS_DIR.glob("*.jpg"),  key=lambda p: extract_index(p.name))

    image_idx = {extract_index(p.name): p for p in image_files}
    mask_idx  = {extract_index(p.name): p for p in mask_files}

    common_indices = sorted(set(image_idx.keys()) & set(mask_idx.keys()))

    if not common_indices:
        raise RuntimeError(
            f"No matching image/mask pairs found!\n"
            f"  Images dir : {IMAGES_DIR}\n"
            f"  Masks dir  : {MASKS_DIR}"
        )

    records = [
        {
            "image_path": str(image_idx[i]),
            "mask_path":  str(mask_idx[i]),
            "index":      i,
        }
        for i in common_indices
    ]
    log.info("Paired %d image-mask samples.", len(records))

    # 2. 70 / 15 / 15 split  (train → val → test)
    train_val, test = train_test_split(records, test_size=0.15, random_state=42)
    train,     val  = train_test_split(train_val, test_size=0.15 / 0.85, random_state=42)

    for r in train: r["split"] = "train"
    for r in val:   r["split"] = "val"
    for r in test:  r["split"] = "test"

    all_records = train + val + test
    log.info("Split → train: %d | val: %d | test: %d", len(train), len(val), len(test))

    # 3. Write manifest CSV
    import csv
    fieldnames = ["index", "split", "image_path", "mask_path"]
    with open(MANIFEST_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_records)
    log.info("Manifest saved → %s", MANIFEST_CSV)

    # 4. Write stats JSON
    stats = {
        "total_pairs":  len(all_records),
        "train_count":  len(train),
        "val_count":    len(val),
        "test_count":   len(test),
        "images_dir":   str(IMAGES_DIR),
        "masks_dir":    str(MASKS_DIR),
        "manifest_csv": str(MANIFEST_CSV),
    }
    with open(STATS_JSON, "w") as f:
        json.dump(stats, f, indent=4)
    log.info("Stats saved → %s", STATS_JSON)
    log.info("━━━ Dataset Preparation Complete ━━━")


if __name__ == "__main__":
    main()
