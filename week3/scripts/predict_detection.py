import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Week 3 prediction images and YOLO label txt outputs.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to best.pt")
    parser.add_argument("--source", type=Path, default=Path("week3/data/images/test"))
    parser.add_argument("--project", type=Path, default=Path("week3/results/predictions"))
    parser.add_argument("--name", default="test_preds")
    parser.add_argument("--device", default="auto", help="auto, 0, cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.project.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    device = "0" if args.device == "auto" else args.device

    model.predict(
        source=str(args.source),
        project=str(args.project),
        name=args.name,
        device=device,
        save=True,
        save_txt=True,
        save_conf=True,
        verbose=False,
    )

    run_dir = args.project / args.name
    image_dir = args.project / "images"
    label_dir = args.project / "labels"
    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    for img in run_dir.glob("*"):
        if img.is_file() and img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            shutil.copy2(img, image_dir / img.name)

    labels_src = run_dir / "labels"
    if labels_src.exists():
        for txt in labels_src.glob("*.txt"):
            shutil.copy2(txt, label_dir / txt.name)

    print(f"Prediction images: {image_dir}")
    print(f"Prediction labels: {label_dir}")


if __name__ == "__main__":
    main()
