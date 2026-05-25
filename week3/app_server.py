"""
Week 3 backend server for detection metrics, artifacts, and live inference.
Run:
  python week3/app_server.py
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from ultralytics import YOLO

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

ROOT = Path(__file__).resolve().parents[1]
WEEK3_DIR = ROOT / "week3"
RESULTS_DIR = WEEK3_DIR / "results"
PRED_IMAGES_DIR = RESULTS_DIR / "predictions" / "images"

GRAPH_FILES = [
    "BoxPR_curve.png",
    "BoxP_curve.png",
    "BoxR_curve.png",
    "BoxF1_curve.png",
    "confusion_matrix.png",
    "confusion_matrix_normalized.png",
]


def resolve_best_weights() -> Path:
    candidates = [
        WEEK3_DIR / "runs" / "wound_detection_cpu" / "weights" / "best.pt",
        WEEK3_DIR / "runs" / "wound_detection_gpu" / "weights" / "best.pt",
        WEEK3_DIR / "runs" / "wound_detection" / "weights" / "best.pt",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("No Week 3 best.pt found in known run directories.")


MODEL: YOLO | None = None
MODEL_PATH: Path | None = None


def get_model() -> YOLO:
    global MODEL, MODEL_PATH
    if MODEL is None:
        MODEL_PATH = resolve_best_weights()
        MODEL = YOLO(str(MODEL_PATH))
    return MODEL


@app.route("/week3/health", methods=["GET"])
def health() -> Any:
    model_ready = False
    model_path = None
    try:
        _ = get_model()
        model_ready = True
        model_path = str(MODEL_PATH)
    except Exception:
        model_ready = False
    return jsonify(
        {
            "status": "ok",
            "model_loaded": model_ready,
            "model_path": model_path,
            "results_dir": str(RESULTS_DIR),
        }
    )


@app.route("/week3/summary", methods=["GET"])
def summary() -> Any:
    path = RESULTS_DIR / "map_results.json"
    if not path.exists():
        return jsonify({"error": f"Missing file: {path}"}), 404
    return send_file(path, mimetype="application/json")


@app.route("/week3/graphs", methods=["GET"])
def graphs() -> Any:
    payload: dict[str, list[str]] = {"eval_val": [], "eval_test": []}
    for split in ("eval_val", "eval_test"):
        split_dir = RESULTS_DIR / split
        for name in GRAPH_FILES:
            file_path = split_dir / name
            if file_path.exists():
                payload[split].append(f"/week3/artifacts/{split}/{name}")
    return jsonify(payload)


@app.route("/week3/predictions", methods=["GET"])
def predictions() -> Any:
    limit = int(request.args.get("limit", 24))
    if not PRED_IMAGES_DIR.exists():
        return jsonify({"images": []})
    images = []
    for p in sorted(PRED_IMAGES_DIR.glob("*")):
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            images.append(f"/week3/artifacts/predictions/images/{p.name}")
    return jsonify({"images": images[:limit], "total": len(images)})


@app.route("/week3/artifacts/<path:subpath>", methods=["GET"])
def artifacts(subpath: str) -> Any:
    safe_base = RESULTS_DIR.resolve()
    candidate = (RESULTS_DIR / subpath).resolve()
    if safe_base not in candidate.parents and candidate != safe_base:
        return jsonify({"error": "Invalid artifact path"}), 400
    if not candidate.exists() or not candidate.is_file():
        return jsonify({"error": "Artifact not found"}), 404
    return send_file(candidate)


@app.route("/week3/detect", methods=["POST"])
def detect() -> Any:
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded in field 'image'"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    conf = float(request.form.get("conf", 0.25))
    img_bytes = file.read()
    np_img = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
    if np_img is None:
        return jsonify({"error": "Unable to decode image"}), 400

    model = get_model()
    results = model.predict(source=np_img, conf=conf, save=False, verbose=False)
    r = results[0]
    plotted = r.plot()

    ok, enc = cv2.imencode(".jpg", plotted)
    if not ok:
        return jsonify({"error": "Failed to encode detection result"}), 500

    detections = []
    for box in r.boxes:
        xyxy = box.xyxy[0].tolist()
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())
        label = model.names.get(cls_id, str(cls_id))
        detections.append(
            {
                "class_id": cls_id,
                "label": label,
                "confidence": score,
                "bbox_xyxy": [round(v, 2) for v in xyxy],
            }
        )

    data_url = "data:image/jpeg;base64," + base64.b64encode(enc.tobytes()).decode("utf-8")
    return jsonify(
        {
            "status": "success",
            "detections": detections,
            "detection_count": len(detections),
            "result_image_data": data_url,
        }
    )


if __name__ == "__main__":
    print("Week 3 server running on http://127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
