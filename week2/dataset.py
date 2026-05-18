"""
week2/dataset.py
================
Defines a custom PyTorch Dataset and standard torchvision transformations
for the Wound vs. Normal binary classification task.
"""

import csv
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class WoundDataset(Dataset):
    """Custom PyTorch Dataset for loading wound and normal healthy feet images."""
    
    def __init__(self, records: list[dict], transform=None):
        """
        Args:
            records (list of dict): List containing image path and label info.
                                    Format: [{"cleaned_path": str, "annotated_label": str}]
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.records = records
        self.transform = transform
        self.class_to_idx = {"normal": 0, "wound": 1}
        self.idx_to_class = {0: "normal", 1: "wound"}

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        record = self.records[idx]
        img_path = Path(record["cleaned_path"])
        
        # Load image safely
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
        except Exception as e:
            # Fallback to an empty image if loading fails
            img = Image.new("RGB", (331, 331), color=0)
            
        label_str = record["annotated_label"]
        label_idx = self.class_to_idx.get(label_str, 0)
        
        if self.transform:
            img_tensor = self.transform(img)
        else:
            # Default fallback transform if none provided
            fallback_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = fallback_transform(img)
            
        return img_tensor, label_idx

def get_transforms() -> tuple[transforms.Compose, transforms.Compose]:
    """
    Returns separate, robust pipelines for training and validation.
    
    - Train pipeline includes robust geometric & color augmentations to avoid overfitting.
    - Validation pipeline only handles tensor conversion and ImageNet normalization.
    """
    train_transform = transforms.Compose([
        # Data Augmentations
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
        
        # Core
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    return train_transform, val_transform
