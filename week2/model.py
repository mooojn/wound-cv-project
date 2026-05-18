"""
week2/model.py
==============
Defines the classification model factory for transfer learning on ResNet-18
and MobileNetV3-Large backbones.
"""

import logging
import torch
import torch.nn as nn
from torchvision import models

log = logging.getLogger(__name__)

def get_model(
    model_name: str = "mobilenet_v3_large", 
    num_classes: int = 2, 
    freeze_backbone: bool = True
) -> nn.Module:
    """
    Model factory to build transfer learning models with pre-trained backbones.
    
    Args:
        model_name (str): Either "mobilenet_v3_large" or "resnet18".
        num_classes (int): Number of target categories (2 for normal/wound).
        freeze_backbone (bool): If True, freezes feature extraction layers to allow
                                ultra-fast training on CPU.
    """
    model_name = model_name.lower().strip()
    log.info("Building model '%s' (classes: %d, freeze_backbone: %s)...", 
             model_name, num_classes, freeze_backbone)
    
    # 1. Safely parse torchvision weights configuration
    try:
        from torchvision.models import ResNet18_Weights, MobileNet_V3_Large_Weights
        resnet_weights = ResNet18_Weights.DEFAULT
        mobilenet_weights = MobileNet_V3_Large_Weights.DEFAULT
    except ImportError:
        resnet_weights = None
        mobilenet_weights = None

    # 2. Build Backbone
    if model_name == "resnet18":
        if resnet_weights is not None:
            model = models.resnet18(weights=resnet_weights)
        else:
            model = models.resnet18(pretrained=True)
            
        # Freeze Backbone if requested
        if freeze_backbone:
            for name, param in model.named_parameters():
                if "fc" not in name:
                    param.requires_grad = False
                    
        # Replace the final linear head
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        
    elif model_name == "mobilenet_v3_large":
        if mobilenet_weights is not None:
            model = models.mobilenet_v3_large(weights=mobilenet_weights)
        else:
            model = models.mobilenet_v3_large(pretrained=True)
            
        # Freeze Backbone if requested
        if freeze_backbone:
            for param in model.features.parameters():
                param.requires_grad = False
                
        # MobileNetV3-Large final classifier has index 3 as the linear out layer
        in_features = model.classifier[3].in_features
        # Replace only the final classification linear layer to keep the dropout and activation
        model.classifier[3] = nn.Linear(in_features, num_classes)
        
    else:
        raise ValueError(f"Unsupported model backbone: {model_name}. Use 'resnet18' or 'mobilenet_v3_large'.")
        
    return model

if __name__ == "__main__":
    # Test building models
    logging.basicConfig(level=logging.INFO)
    m1 = get_model("resnet18")
    print(f"ResNet-18 head: {m1.fc}")
    m2 = get_model("mobilenet_v3_large")
    print(f"MobileNet head: {m2.classifier}")
