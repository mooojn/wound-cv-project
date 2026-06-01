"""
week4/app_server.py
===================
Flask REST API for live wound segmentation inference.

Endpoints:
    POST /week4/segment
        Body : multipart/form-data  →  field "file" (image)
        Returns : JSON with base64-encoded overlay PNG + metrics

    GET  /week4/results
        Returns : Latest evaluation metrics from segmentation_evaluation.json

Usage (from project root):
    python week4/app_server.py
    # → http://127.0.0.1:5002
"""

import base64
import io
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from PIL import Image
from torchvision import transforms

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ROOT         = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = ROOT / "week4" / "weights" / "best_seg_model.pth"
METRICS_PATH = ROOT / "week4" / "metrics" / "segmentation_evaluation.json"
METRICS_DIR  = ROOT / "week4" / "metrics"
PRED_DIR     = ROOT / "week4" / "predictions"
IMAGE_SIZE   = 256
THRESHOLD    = 0.5

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Model loader (singleton)
# ---------------------------------------------------------------------------
_model  = None
_device = None


def _get_model():
    global _model, _device

    if _model is not None:
        return _model, _device

    from model import get_segmentation_model

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Loading segmentation model on %s …", _device)

    _model = get_segmentation_model(
        architecture="unet",
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        num_classes=1,
    )

    if not WEIGHTS_PATH.exists():
        raise FileNotFoundError(
            f"Model weights not found: {WEIGHTS_PATH}\n"
            "Run the training pipeline first."
        )

    _model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=_device))
    _model.to(_device)
    _model.eval()
    log.info("Model ready.")
    return _model, _device


# ---------------------------------------------------------------------------
# Image pre/post processing helpers
# ---------------------------------------------------------------------------

_preprocess = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _overlay_mask(
    image_np: np.ndarray,
    mask_np:  np.ndarray,
    color: tuple = (255, 60, 60),
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend binary prediction mask onto the original image."""
    overlay      = image_np.copy()
    wound_pixels = mask_np > 0.5
    overlay[wound_pixels] = (
        (1 - alpha) * image_np[wound_pixels] + alpha * np.array(color)
    ).astype(np.uint8)
    return overlay


def _pil_to_base64(img: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    img.save(buffer, format=fmt)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a normalised (C, H, W) tensor back to a PIL image."""
    img = tensor.permute(1, 2, 0).cpu().numpy()
    img = img * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(img)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/week4/segment", methods=["POST"])
def segment():
    """Accept an uploaded image and return segmentation results."""
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Use field name 'file'."}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    try:
        # 1. Load image
        pil_img = Image.open(file.stream).convert("RGB")
        original_size = pil_img.size

        # 2. Preprocess
        model, device = _get_model()
        input_tensor  = _preprocess(pil_img).unsqueeze(0).to(device)

        # 3. Inference
        with torch.no_grad():
            logits = model(input_tensor)
            prob   = torch.sigmoid(logits)
            pred   = (prob > THRESHOLD).float()

        # 4. Build outputs
        pred_mask_np = pred[0, 0].cpu().numpy()             # (H, W) binary

        # Resize mask back to original image size for overlay
        mask_pil    = Image.fromarray((pred_mask_np * 255).astype(np.uint8))
        mask_pil    = mask_pil.resize(original_size, Image.NEAREST)
        mask_np_full = np.array(mask_pil) > 127

        orig_np      = np.array(pil_img)
        overlay_np   = _overlay_mask(orig_np, mask_np_full.astype(np.uint8))

        overlay_pil = Image.fromarray(overlay_np)
        mask_pil_out = Image.fromarray((mask_np_full * 255).astype(np.uint8))

        # 5. Pixel-level metrics (on model-size prediction)
        wound_pixels  = pred_mask_np.sum()
        total_pixels  = pred_mask_np.size
        wound_percent = float(wound_pixels / total_pixels * 100)
        mean_prob     = float(prob[0, 0].mean().item())

        response = {
            "status":          "success",
            "original_size":   list(original_size),
            "wound_coverage":  round(wound_percent, 2),
            "mean_confidence": round(mean_prob * 100, 2),
            "threshold":       THRESHOLD,
            "overlay_b64":     _pil_to_base64(overlay_pil),
            "mask_b64":        _pil_to_base64(mask_pil_out),
        }

        return jsonify(response), 200

    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503
    except Exception as e:
        log.exception("Segmentation error")
        return jsonify({"error": str(e)}), 500


@app.route("/week4/results", methods=["GET"])
def get_results():
    """Return the latest evaluation metrics from the test set."""
    if not METRICS_PATH.exists():
        return jsonify({
            "error": "No evaluation results found. Run evaluate.py first.",
            "metrics_path": str(METRICS_PATH),
        }), 404

    with open(METRICS_PATH) as f:
        metrics = json.load(f)

    return jsonify({"status": "success", "metrics": metrics}), 200


@app.route("/week4/health", methods=["GET"])
def health():
    return jsonify({
        "status":         "ok",
        "weights_exists": WEIGHTS_PATH.exists(),
        "metrics_exists": METRICS_PATH.exists(),
        "device":         str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
    }), 200


@app.route("/week4/assets/<asset_type>/<path:filename>", methods=["GET"])
def get_asset(asset_type, filename):
    """Serve generated Week 4 charts and prediction visualisations."""
    asset_dirs = {
        "metrics": METRICS_DIR,
        "predictions": PRED_DIR,
    }
    directory = asset_dirs.get(asset_type)
    if directory is None:
        return jsonify({"error": "Unknown asset type."}), 404
    return send_from_directory(directory, filename)


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    log.info("Starting Week 4 Segmentation Server on http://127.0.0.1:5002")
    app.run(host="127.0.0.1", port=5002, debug=False)
