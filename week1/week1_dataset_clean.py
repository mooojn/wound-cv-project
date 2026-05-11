"""
week1_dataset_clean.py
======================
Dataset cleaning and 20-sample selection for the Wound CV Project.

Expected raw data layout (after downloading from Mendeley):
    data/raw/Normal/          ← healthy feet images
    data/raw/Wound_Main/      ← wound images, one sub-folder per class
    data/raw/Wound_Masked/    ← corresponding segmentation masks

Outputs:
    data/cleaned/             ← cleaned, renamed, verified images
    data/samples/             ← 20 representative images for annotation
    data/cleaned/dataset_manifest.csv   ← full manifest of cleaned data
    data/cleaned/sample_manifest.csv    ← manifest of the 20 samples
"""

import os
import shutil
import hashlib
import random
import csv
import json
import logging
from pathlib import Path
from collections import defaultdict

from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RAW_DIR       = Path("data")
CLEANED_DIR   = Path("data/cleaned")
SAMPLES_DIR   = Path("data/samples")

WOUND_MAIN    = RAW_DIR / "wound_main"
WOUND_MASKED  = RAW_DIR / "wound_mask"
NORMAL_DIR    = RAW_DIR / "Nomal"

TARGET_SIZE   = (331, 331)          # dataset native resolution
SAMPLES_COUNT = 20                  # annotation quota for Week 1
RANDOM_SEED   = 42

WOUND_CLASSES = [
    "diabetic", "pressure", "trauma", "venous",
    "surgical", "arterial", "cellulitis", "miscellaneous"
]

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def md5(path: Path) -> str:
    """Return MD5 hex digest of a file (duplicate detection)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: Path) -> tuple[bool, str]:
    """
    Open with Pillow and run basic sanity checks.
    Returns (True, '') on success or (False, reason) on failure.
    """
    try:
        with Image.open(path) as img:
            img.verify()                        # catches truncated files
        with Image.open(path) as img:
            arr = np.array(img.convert("RGB"))
            if arr.size == 0:
                return False, "empty pixel array"
            if arr.mean() < 2:
                return False, "nearly all-black image"
            if arr.mean() > 253:
                return False, "nearly all-white image"
        return True, ""
    except Exception as exc:
        return False, str(exc)


def clean_filename(original: str) -> str:
    """Lower-case, replace spaces/special chars with underscores."""
    stem = Path(original).stem
    stem = stem.lower().replace(" ", "_")
    # keep only alphanumeric, dash, underscore
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)
    return stem + ".jpg"

# ---------------------------------------------------------------------------
# Step 1 — Collect all raw image paths with metadata
# ---------------------------------------------------------------------------

def collect_raw_images() -> list[dict]:
    """Walk raw directories and return a list of record dicts."""
    records = []

    # ── Wound images ────────────────────────────────────────────────────────
    if WOUND_MAIN.exists():
        for img_path in sorted(WOUND_MAIN.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                records.append({
                    "original_path": str(img_path),
                    "label": "miscellaneous",   # no class sub-folders, assign manually in annotator
                    "split": "wound",
                })
    else:
        log.warning("wound_main directory not found: %s", WOUND_MAIN)

    # ── Normal (healthy) images ──────────────────────────────────────────────
    if NORMAL_DIR.exists():
        for img_path in sorted(NORMAL_DIR.glob("*")):
            if img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                records.append({
                    "original_path": str(img_path),
                    "label": "normal",
                    "split": "normal",
                })
    else:
        log.warning("Normal directory not found: %s", NORMAL_DIR)

    log.info("Collected %d raw files", len(records))
    return records

# ---------------------------------------------------------------------------
# Step 2 — Validate + deduplicate + copy to cleaned/
# ---------------------------------------------------------------------------

def clean_dataset(records: list[dict]) -> list[dict]:
    """
    Validate each image, deduplicate by MD5, resize if needed,
    copy to CLEANED_DIR/<label>/, and return cleaned record list.
    """
    CLEANED_DIR.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[str] = set()
    cleaned: list[dict] = []
    stats = defaultdict(int)

    for rec in records:
        src = Path(rec["original_path"])
        label = rec["label"]

        # ── Validate ────────────────────────────────────────────────────────
        ok, reason = is_valid_image(src)
        if not ok:
            log.debug("SKIP (invalid) %s — %s", src.name, reason)
            stats["invalid"] += 1
            continue

        # ── Deduplicate ─────────────────────────────────────────────────────
        digest = md5(src)
        if digest in seen_hashes:
            log.debug("SKIP (duplicate) %s", src.name)
            stats["duplicate"] += 1
            continue
        seen_hashes.add(digest)

        # ── Destination ─────────────────────────────────────────────────────
        dst_dir = CLEANED_DIR / label
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_name = clean_filename(src.name)
        # avoid name collision in same label folder
        counter = 0
        dst = dst_dir / dst_name
        while dst.exists():
            counter += 1
            dst = dst_dir / (Path(dst_name).stem + f"_{counter}.jpg")

        # ── Copy / resize ────────────────────────────────────────────────────
        try:
            with Image.open(src) as img:
                img = img.convert("RGB")
                if img.size != TARGET_SIZE:
                    img = img.resize(TARGET_SIZE, Image.LANCZOS)
                img.save(dst, "JPEG", quality=95)
        except Exception as exc:
            log.warning("SKIP (save error) %s — %s", src.name, exc)
            stats["save_error"] += 1
            continue

        cleaned.append({
            "original_path": str(src),
            "cleaned_path":  str(dst),
            "label":         label,
            "split":         rec["split"],
            "md5":           digest,
            "width":         TARGET_SIZE[0],
            "height":        TARGET_SIZE[1],
        })
        stats["kept"] += 1

    log.info(
        "Cleaning done — kept: %d | duplicates: %d | invalid: %d | errors: %d",
        stats["kept"], stats["duplicate"], stats["invalid"], stats["save_error"]
    )
    return cleaned

# ---------------------------------------------------------------------------
# Step 3 — Save full manifest CSV
# ---------------------------------------------------------------------------

def save_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not records:
        log.warning("No records to write to %s", path)
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)
    log.info("Manifest saved → %s (%d rows)", path, len(records))

# ---------------------------------------------------------------------------
# Step 4 — Stratified sample of 20 images for annotation
# ---------------------------------------------------------------------------

def select_samples(cleaned: list[dict], n: int = SAMPLES_COUNT) -> list[dict]:
    """
    Pick ~equal images from each wound class (+ a few normal) so the
    annotator sees a representative spread across all 8 wound types.
    """
    random.seed(RANDOM_SEED)

    by_label: dict[str, list[dict]] = defaultdict(list)
    for rec in cleaned:
        by_label[rec["label"]].append(rec)

    # All labels present in the cleaned set (wound classes + normal)
    labels = sorted(by_label.keys())
    per_label = max(1, n // len(labels))

    selected: list[dict] = []
    for label in labels:
        pool = by_label[label]
        k = min(per_label, len(pool))
        selected.extend(random.sample(pool, k))

    # If we still need more, fill from largest classes
    if len(selected) < n:
        all_remaining = [
            r for r in cleaned if r not in selected
        ]
        random.shuffle(all_remaining)
        selected.extend(all_remaining[: n - len(selected)])

    selected = selected[:n]
    random.shuffle(selected)

    log.info(
        "Selected %d samples — label distribution: %s",
        len(selected),
        {lbl: sum(1 for r in selected if r["label"] == lbl) for lbl in labels},
    )
    return selected

# ---------------------------------------------------------------------------
# Step 5 — Copy samples to data/samples/ and return updated records
# ---------------------------------------------------------------------------

def copy_samples(samples: list[dict]) -> list[dict]:
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    updated = []
    for i, rec in enumerate(samples, start=1):
        src = Path(rec["cleaned_path"])
        dst = SAMPLES_DIR / f"sample_{i:02d}_{rec['label']}.jpg"
        shutil.copy2(src, dst)
        updated.append({**rec, "sample_path": str(dst), "sample_id": i})

    log.info("Samples copied → %s", SAMPLES_DIR)
    return updated

# ---------------------------------------------------------------------------
# Step 6 — Print dataset statistics
# ---------------------------------------------------------------------------

def print_statistics(cleaned: list[dict]) -> None:
    total = len(cleaned)
    by_label: dict[str, int] = defaultdict(int)
    for r in cleaned:
        by_label[r["label"]] += 1

    print("\n" + "=" * 60)
    print(f"  DATASET STATISTICS  (total: {total} images)")
    print("=" * 60)
    for label in sorted(by_label):
        bar = "#" * (by_label[label] * 30 // max(by_label.values()))
        print(f"  {label:<18} {by_label[label]:>5}  {bar}")
    print("=" * 60 + "\n")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log.info("━━━ Week 1 — Dataset Cleaning & Sample Selection ━━━")

    # 1. Collect
    raw = collect_raw_images()
    if not raw:
        log.error(
            "No images found. Make sure data/raw/Wound_Main and data/raw/Normal exist."
        )
        return

    # 2. Clean
    cleaned = clean_dataset(raw)
    if not cleaned:
        log.error("No images survived cleaning. Check your raw data.")
        return

    # 3. Full manifest
    save_manifest(cleaned, CLEANED_DIR / "dataset_manifest.csv")

    # 4. Statistics
    print_statistics(cleaned)

    # 5. Stratified sample
    samples = select_samples(cleaned, n=SAMPLES_COUNT)

    # 6. Copy samples
    samples = copy_samples(samples)

    # 7. Sample manifest
    save_manifest(samples, CLEANED_DIR / "sample_manifest.csv")

    log.info("━━━ Week 1 cleaning complete ━━━")
    log.info("Next step → python week1/week1_annotator.py --images data/samples")


if __name__ == "__main__":
    main()
