"""
week4/model.py
==============
Segmentation model factory using segmentation-models-pytorch (smp).

Default: U-Net with a ResNet-34 encoder pre-trained on ImageNet.
Alternative encoders selectable via --encoder argument.

U-Net was chosen because:
  - Strong encoder-decoder skip connections → good spatial precision
  - Well-established benchmark on medical image segmentation
  - Lightweight enough for CPU inference
"""

import logging

import torch
import torch.nn as nn

log = logging.getLogger(__name__)


def get_segmentation_model(
    architecture: str = "unet",
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    in_channels: int = 3,
    num_classes: int = 1,
) -> nn.Module:
    """
    Build a segmentation model from segmentation-models-pytorch.

    Args:
        architecture    : "unet" | "unetplusplus" | "fpn" | "deeplabv3plus"
        encoder_name    : Any smp-compatible encoder (resnet34, efficientnet-b0, …)
        encoder_weights : "imagenet" or None
        in_channels     : Input image channels (3 for RGB)
        num_classes     : Output mask channels (1 for binary segmentation)

    Returns:
        nn.Module ready to call with (B, C, H, W) input.
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation-models-pytorch is required. "
            "Install it with: pip install segmentation-models-pytorch"
        )

    arch = architecture.lower().strip()
    log.info(
        "Building segmentation model: %s | encoder: %s | weights: %s",
        arch, encoder_name, encoder_weights,
    )

    factory = {
        "unet":           smp.Unet,
        "unetplusplus":   smp.UnetPlusPlus,
        "fpn":            smp.FPN,
        "deeplabv3plus":  smp.DeepLabV3Plus,
    }

    if arch not in factory:
        raise ValueError(
            f"Unknown architecture '{arch}'. "
            f"Choose from: {list(factory.keys())}"
        )

    model = factory[arch](
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels,
        classes=num_classes,
        activation=None,          # Raw logits → BCEWithLogitsLoss handles sigmoid
    )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(
        "Model built. Total params: %s | Trainable params: %s",
        f"{total_params:,}", f"{trainable_params:,}",
    )

    return model


# ---------------------------------------------------------------------------
# Quick smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    m = get_segmentation_model()
    dummy = torch.randn(2, 3, 256, 256)
    out   = m(dummy)
    print(f"Input : {dummy.shape}")
    print(f"Output: {out.shape}")   # Expected: (2, 1, 256, 256)
