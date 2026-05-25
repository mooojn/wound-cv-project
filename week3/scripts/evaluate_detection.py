import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Week 3 detection model and export summary metrics.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--data", type=Path, default=Path("week3/configs/wound_detection.yaml"))
    parser.add_argument("--project", type=Path, default=Path("week3/results"))
    parser.add_argument("--name", default="eval")
    parser.add_argument("--device", default="auto", help="auto, 0, cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    device = "0" if args.device == "auto" else args.device

    rows = []
    for split in ("val", "test"):
        metrics = model.val(
            data=str(args.data),
            split=split,
            project=str(args.project),
            name=f"{args.name}_{split}",
            device=device,
            save_json=True,
            verbose=False,
        )
        rows.append(
            {
                "split": split,
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "mAP50": float(metrics.box.map50),
                "mAP50-95": float(metrics.box.map),
            }
        )

    csv_path = args.project / "map_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "precision", "recall", "mAP50", "mAP50-95"])
        writer.writeheader()
        writer.writerows(rows)

    json_path = args.project / "map_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
