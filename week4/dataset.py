"""
week4/dataset.py
================
PyTorch Dataset for wound segmentation.

Each sample returns:
    image  : (3, H, W) float32 tensor, ImageNet-normalised
    mask   : (1, H, W) float32 tensor, values in {0.0, 1.0}
"""

import logging
from pathlib import Path

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

log = logging.getLogger(__name__)

# Default spatial resolution sent to the model
IMAGE_SIZE = 256


class SegmentationDataset(Dataset):
    """
    Dataset for paired wound image + binary mask segmentation.

    Args:
        records (list[dict]): Rows from the manifest CSV.
                              Each dict must contain keys:
                              "image_path", "mask_path".
        augment (bool): If True, applies random spatial augmentations
                        (only for the training split).
        image_size (int): Resize target for both image and mask.
    """

    # ImageNet stats used across Week 2 / Week 3 — keep consistent
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD  = [0.229, 0.224, 0.225]

    def __init__(
        self,
        records: list[dict],
        augment: bool = False,
        image_size: int = IMAGE_SIZE,
    ):
        self.records    = records
        self.augment    = augment
        self.image_size = image_size

        # Shared spatial transform (applied identically to image and mask)
        self._spatial_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
        ])

        # Image-only colour normalisation
        self._img_to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
        ])

        # Mask to binary float tensor (no normalisation)
        self._mask_to_tensor = transforms.Compose([
            transforms.ToTensor(),   # → (1, H, W) in [0, 1]
        ])

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rec = self.records[idx]

        # ── Load image ──────────────────────────────────────────────────
        try:
            img = Image.open(rec["image_path"]).convert("RGB")
        except Exception:
            log.warning("Failed to load image: %s — using blank.", rec["image_path"])
            img = Image.new("RGB", (self.image_size, self.image_size), 0)

        # ── Load mask ───────────────────────────────────────────────────
        try:
            mask = Image.open(rec["mask_path"]).convert("L")
        except Exception:
            log.warning("Failed to load mask: %s — using blank.", rec["mask_path"])
            mask = Image.new("L", (self.image_size, self.image_size), 0)

        # ── Resize both to the same spatial size ───────────────────────
        img  = img.resize((self.image_size, self.image_size), Image.BILINEAR)
        mask = mask.resize((self.image_size, self.image_size), Image.NEAREST)

        # ── Optional augmentation (train split only) ────────────────────
        if self.augment:
            img, mask = self._augment(img, mask)

        # ── Convert to tensors ──────────────────────────────────────────
        img_tensor  = self._img_to_tensor(img)
        mask_tensor = self._mask_to_tensor(mask)

        # Binarise mask: any pixel > 0.5 after ToTensor → 1
        mask_tensor = (mask_tensor > 0.5).float()

        return img_tensor, mask_tensor

    # ------------------------------------------------------------------
    def _augment(self, img: Image.Image, mask: Image.Image):
        """Apply identical random transforms to both image and mask."""
        import random

        # Random horizontal flip
        if random.random() > 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        # Random vertical flip
        if random.random() > 0.5:
            img  = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        # Random 90° rotation (0, 90, 180, 270)
        angle = random.choice([0, 90, 180, 270])
        if angle != 0:
            img  = img.rotate(angle)
            mask = mask.rotate(angle)

        # Colour jitter (image only)
        jitter = transforms.ColorJitter(
            brightness=0.2, contrast=0.2, saturation=0.1
        )
        img = jitter(img)

        return img, mask
