import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image

from .config import DEFAULT_MODEL_PATH, PROCESSED_DATA_DIR, PROJECT_CLASS_NAMES
from .data_loader import get_transforms
from .model import load_model_for_inference
from .utils import get_device, setup_logging, to_python_float


def _load_checkpoint_model(
    model_path: Path,
) -> tuple[torch.nn.Module, List[str], torch.device]:
    device = get_device()
    model, class_names = load_model_for_inference(
        model_path=str(model_path),
        device=device,
        num_classes_fallback=len(PROJECT_CLASS_NAMES),
    )
    return model, class_names, device


def _predict_tensor(
    image_tensor: torch.Tensor,
    model: torch.nn.Module,
    class_names: List[str],
    device: torch.device,
) -> Dict[str, Any]:
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.unsqueeze(0).to(device)
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        conf, pred_idx = torch.max(probs, 1)

    confidence = to_python_float(conf)
    pred_idx_int = int(pred_idx.item())
    predicted_class = class_names[pred_idx_int]

    probs_np = probs.cpu().numpy()[0]
    all_probs = {class_names[i]: float(probs_np[i]) for i in range(len(class_names))}

    return {
        "class_name": predicted_class,
        "confidence": confidence,
        "all_probs": all_probs,
    }


def predict_image(
    image_path: str,
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> Dict[str, Any]:
    """
    Predict disease class for a single image path.

    Args:
        image_path: path to the image file
        model_path: path to the trained model checkpoint

    Returns:
        dict with keys: class_name, confidence, all_probs
    """
    setup_logging()
    logger = logging.getLogger("predict")
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path_obj}")

    model, class_names, device = _load_checkpoint_model(model_path_obj)

    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(train=False)
    image_tensor = transform(image)

    result = _predict_tensor(image_tensor, model, class_names, device)

    logger.info(
        "Predicted %s with confidence %.4f",
        result["class_name"],
        result["confidence"],
    )
    return result


def predict_pil_image(
    image: Image.Image,
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> Dict[str, Any]:
    """
    Predict disease class for a PIL image.

    Args:
        image: PIL Image
        model_path: path to the trained model checkpoint

    Returns:
        dict with keys: class_name, confidence, all_probs
    """
    model_path_obj = Path(model_path)
    if not model_path_obj.exists():
        raise FileNotFoundError(f"Model file not found: {model_path_obj}")

    model, class_names, device = _load_checkpoint_model(model_path_obj)

    image = image.convert("RGB")
    transform = get_transforms(train=False)
    image_tensor = transform(image)

    result = _predict_tensor(image_tensor, model, class_names, device)
    return result


def predict_bytes(
    file_bytes: bytes,
    model_path: str = str(DEFAULT_MODEL_PATH),
) -> Dict[str, Any]:
    """
    Predict disease class for image bytes (e.g. from an uploaded file).
    """
    image = Image.open(BytesIO(file_bytes)).convert("RGB")
    return predict_pil_image(image=image, model_path=model_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict papaya leaf disease for a single image"
    )
    parser.add_argument(
        "--image-path",
        type=str,
        required=True,
        help="Path to input image",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model checkpoint",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = predict_image(args.image_path, model_path=args.model_path)

    print(f"Predicted class: {result['class_name']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print("Class probabilities:")
    for name, prob in result["all_probs"].items():
        print(f"  {name}: {prob:.4f}")


if __name__ == "__main__":
    main()
