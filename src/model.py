from typing import Optional

import torch
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def create_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Create a ResNet18 model for classification.

    Args:
        num_classes: number of output classes
        pretrained: whether to use ImageNet pre-trained weights

    Returns:
        A PyTorch nn.Module model.
    """
    if pretrained:
        weights: Optional[ResNet18_Weights] = ResNet18_Weights.DEFAULT
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


def load_model_for_inference(
    model_path: str,
    device: torch.device,
    num_classes_fallback: int,
) -> tuple[nn.Module, list]:
    """
    Load model and class names from a checkpoint.

    The checkpoint can be either:
    - a full dict with keys 'model_state_dict' and 'class_names'
    - or a raw state_dict (for backward compatibility)

    Args:
        model_path: path to checkpoint
        device: torch.device
        num_classes_fallback: used if no class info is found

    Returns:
        (model, class_names)
    """
    checkpoint = torch.load(model_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        class_names = checkpoint.get("class_names")
        if class_names is None:
            class_names = [str(i) for i in range(num_classes_fallback)]
    else:
        # Raw state dict fallback
        state_dict = checkpoint
        class_names = [str(i) for i in range(num_classes_fallback)]

    num_classes = len(class_names)
    model = create_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, class_names
