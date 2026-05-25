"""
Prepare Week 3 YOLO detection dataset from wound images + wound masks.

Inputs:
- data/wound_main/*.jpg
- data/wound_mask/*.jpg

Outputs:
- week3/data/images/{train,val,test}
- week3/data/labels/{train,val,test}
- week3/data/dataset_manifest.csv
- week3/results/part1_dataset_stats.json
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


RANDOM_SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass
class Sample:
    sample_id: str
    image_path: Path
    mask_path: Path
    width: int
    height: int
    bbox_xyxy: tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", type=Path, default=Path("data/wound_main"))
    parser.add_argument("--masks", type=Path, default=Path("data/wound_mask"))
    parser.add_argument("--out", type=Path, default=Path("week3/data"))
    parser.add_argument("--results", type=Path, default=Path("week3/results"))
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--mask-threshold", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="0 means use all pairs")
    return parser.parse_args()


def numeric_id(path: Path) -> str | None:
    # Matches names like wound_main-0103.jpg or wound_mask-0103.jpg
    m = re.search(r"(\d+)", path.stem)
    return m.group(1) if m else None


def yolo_from_xyxy(x1: int, y1: int, x2: int, y2: int, w: int, h: int) -> tuple[float, float, float, float]:
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = x1 + bw / 2.0
    cy = y1 + bh / 2.0
    return cx / w, cy / h, bw / w, bh / h


def bbox_from_mask(mask_path: Path, threshold: int) -> tuple[int, int, int, int] | None:
    with Image.open(mask_path) as m:
        arr = np.asarray(m.convert("L"))
    ys, xs = np.where(arr > threshold)
    if len(xs) == 0 or len(ys) == 0:
        return None
    x1, y1 = int(xs.min()), int(ys.min())
    x2, y2 = int(xs.max()), int(ys.max())
    return (x1, y1, x2, y2)


def ensure_clean_dirs(out_root: Path) -> None:
    for split in ("train", "val", "test"):
        image_split = out_root / "images" / split
        label_split = out_root / "labels" / split
        if image_split.exists():
            shutil.rmtree(image_split)
        if label_split.exists():
            shutil.rmtree(label_split)
        image_split.mkdir(parents=True, exist_ok=True)
        label_split.mkdir(parents=True, exist_ok=True)


def build_samples(image_dir: Path, mask_dir: Path, threshold: int) -> list[Sample]:
    image_map: dict[str, Path] = {}
    mask_map: dict[str, Path] = {}

    for p in sorted(image_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            sid = numeric_id(p)
            if sid:
                image_map[sid] = p

    for p in sorted(mask_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            sid = numeric_id(p)
            if sid:
                mask_map[sid] = p

    pairs = sorted(set(image_map.keys()) & set(mask_map.keys()))
    samples: list[Sample] = []

    for sid in pairs:
        img_path = image_map[sid]
        mask_path = mask_map[sid]
        with Image.open(img_path) as img:
            w, h = img.size
        bbox = bbox_from_mask(mask_path, threshold)
        if bbox is None:
            continue
        samples.append(
            Sample(
                sample_id=sid,
                image_path=img_path,
                mask_path=mask_path,
                width=w,
                height=h,
                bbox_xyxy=bbox,
            )
        )
    return samples


def split_samples(samples: list[Sample], train_ratio: float, val_ratio: float, seed: int) -> dict[str, list[Sample]]:
    if not (0 < train_ratio < 1 and 0 < val_ratio < 1 and train_ratio + val_ratio < 1):
        raise ValueError("Invalid split ratios. Must satisfy train>0, val>0, train+val<1.")

    rng = random.Random(seed)
    shuffled = samples[:]
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val : n_train + n_val + n_test],
    }


def write_split(split_name: str, split_samples_: list[Sample], out_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for s in split_samples_:
        img_dst = out_root / "images" / split_name / s.image_path.name
        label_dst = out_root / "labels" / split_name / f"{s.image_path.stem}.txt"

        shutil.copy2(s.image_path, img_dst)

        x1, y1, x2, y2 = s.bbox_xyxy
        cx, cy, bw, bh = yolo_from_xyxy(x1, y1, x2, y2, s.width, s.height)
        # single class: wound => 0
        label_dst.write_text(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n", encoding="utf-8")

        rows.append(
            {
                "sample_id": s.sample_id,
                "split": split_name,
                "image_path": str(img_dst).replace("\\", "/"),
                "label_path": str(label_dst).replace("\\", "/"),
                "width": str(s.width),
                "height": str(s.height),
                "bbox_x1": str(x1),
                "bbox_y1": str(y1),
                "bbox_x2": str(x2),
                "bbox_y2": str(y2),
                "class_name": "wound",
                "class_id": "0",
            }
        )
    return rows


def main() -> None:
    args = parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    args.results.mkdir(parents=True, exist_ok=True)
    ensure_clean_dirs(args.out)

    samples = build_samples(args.images, args.masks, args.mask_threshold)
    if args.limit and args.limit > 0:
        samples = samples[: args.limit]

    if not samples:
        raise RuntimeError("No valid image-mask pairs with detectable wound region were found.")

    split_map = split_samples(samples, args.train_ratio, args.val_ratio, args.seed)

    manifest_rows: list[dict[str, str]] = []
    for split_name in ("train", "val", "test"):
        manifest_rows.extend(write_split(split_name, split_map[split_name], args.out))

    manifest_path = args.out / "dataset_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "split",
            "image_path",
            "label_path",
            "width",
            "height",
            "bbox_x1",
            "bbox_y1",
            "bbox_x2",
            "bbox_y2",
            "class_name",
            "class_id",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(manifest_rows)

    split_counts = Counter(row["split"] for row in manifest_rows)
    class_counts = Counter(row["class_name"] for row in manifest_rows)
    stats = {
        "total_samples": len(manifest_rows),
        "split_counts": dict(split_counts),
        "class_counts": dict(class_counts),
        "images_dir": str((args.out / "images").resolve()),
        "labels_dir": str((args.out / "labels").resolve()),
        "manifest": str(manifest_path.resolve()),
    }
    stats_path = args.results / "part1_dataset_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print("\n=== Week 3 Part 1 Dataset Preparation Complete ===")
    print(f"Total labeled samples: {stats['total_samples']}")
    print(f"Split counts: {stats['split_counts']}")
    print(f"Class counts: {stats['class_counts']}")
    print(f"Manifest: {manifest_path}")
    print(f"Stats: {stats_path}")


if __name__ == "__main__":
    main()
