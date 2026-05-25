import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str]) -> None:
    print(f"\n[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full Week 3 pipeline: prepare -> train -> eval -> predict")
    parser.add_argument("--device", default="auto", help="auto, 0, cpu")
    parser.add_argument("--run-name", default="wound_detection")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    py = sys.executable
    root = Path(__file__).resolve().parents[1]

    run([py, str(root / "scripts" / "prepare_detection_dataset.py")])
    run([py, str(root / "scripts" / "train_detection.py"), "--device", args.device, "--name", args.run_name])

    weights = root / "runs" / args.run_name / "weights" / "best.pt"
    run([py, str(root / "scripts" / "evaluate_detection.py"), "--weights", str(weights), "--device", args.device])
    run([py, str(root / "scripts" / "predict_detection.py"), "--weights", str(weights), "--device", args.device])

    print("\nWeek 3 pipeline complete.")


if __name__ == "__main__":
    main()
