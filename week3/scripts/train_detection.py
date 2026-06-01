import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 wound detection model")
    default_model = Path(__file__).resolve().parents[1] / "weights" / "pretrained" / "yolov8n.pt"
    parser.add_argument("--model", default=str(default_model), help="Base YOLOv8 model weights")
    parser.add_argument("--imgsz", type=int, default=512, help="Training image size")
    parser.add_argument("--epochs", type=int, default=60, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--amp", action="store_true", default=True, help="Enable AMP")
    parser.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--name", default="wound_detection", help="Run name")
    parser.add_argument(
        "--device",
        default="auto",
        help="Training device (auto, 0, 0,1, cpu). Auto uses CUDA:0 if available.",
    )
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic in final epochs")
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    return "0" if torch.cuda.is_available() else "cpu"


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    data_config = repo_root / "week3" / "configs" / "wound_detection.yaml"
    project_dir = repo_root / "week3" / "runs"
    device = resolve_device(args.device)

    print(f"[Week3-Part2] Training on device: {device}")
    print(f"[Week3-Part2] Data config: {data_config}")
    print(f"[Week3-Part2] Run name: {args.name}")

    model = YOLO(args.model)
    model.train(
        data=str(data_config),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        cache=False,
        amp=args.amp,
        patience=args.patience,
        project=str(project_dir),
        name=args.name,
        device=device,
        close_mosaic=args.close_mosaic,
        exist_ok=False,
    )


if __name__ == "__main__":
    main()
