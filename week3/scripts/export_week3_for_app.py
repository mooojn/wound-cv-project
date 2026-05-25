import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Week 3 summary + prediction assets into app/public.")
    parser.add_argument("--results", type=Path, default=Path("week3/results"))
    parser.add_argument("--public", type=Path, default=Path("app/public"))
    parser.add_argument("--max-images", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.public.mkdir(parents=True, exist_ok=True)

    csv_path = args.results / "map_results.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {csv_path}")

    rows = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    summary_path = args.public / "week3_summary.json"
    summary_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    pred_src = args.results / "predictions" / "images"
    pred_dst = args.public / "week3_predictions"
    if pred_dst.exists():
        shutil.rmtree(pred_dst)
    pred_dst.mkdir(parents=True, exist_ok=True)

    copied = []
    if pred_src.exists():
        for img in sorted(pred_src.glob("*"))[: args.max_images]:
            if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                target = pred_dst / img.name
                shutil.copy2(img, target)
                copied.append(f"/week3_predictions/{img.name}")

    pred_index = args.public / "week3_predictions.json"
    pred_index.write_text(json.dumps(copied, indent=2), encoding="utf-8")

    print(f"Saved: {summary_path}")
    print(f"Saved: {pred_index}")
    print(f"Copied predictions: {len(copied)}")


if __name__ == "__main__":
    main()
