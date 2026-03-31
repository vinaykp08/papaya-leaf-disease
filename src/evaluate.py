import argparse
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets

from .config import (
    DEFAULT_MODEL_PATH,
    METRICS_CSV_PATH,
    METRICS_JSON_PATH,
    PROCESSED_DATA_DIR,
)
from .data_loader import get_transforms
from .model import load_model_for_inference
from .utils import get_device, setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Papaya Leaf Disease Detection Model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers",
    )
    return parser.parse_args()


def create_test_loader(
    processed_data_dir: Path,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, List[str]]:
    test_dir = processed_data_dir / "test"
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))
    class_names = test_dataset.classes
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return test_loader, class_names


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    class_names: List[str],
) -> Tuple[float, Dict[str, float], np.ndarray]:
    model.eval()
    total = 0
    correct = 0
    all_labels: List[int] = []
    all_preds: List[int] = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy().tolist())
            all_preds.extend(preds.cpu().numpy().tolist())

    overall_acc = correct / max(total, 1)

    per_class_correct: Dict[int, int] = defaultdict(int)
    per_class_total: Dict[int, int] = Counter(all_labels)

    for true_label, pred_label in zip(all_labels, all_preds):
        if true_label == pred_label:
            per_class_correct[true_label] += 1

    per_class_acc: Dict[str, float] = {}
    for idx, name in enumerate(class_names):
        total_for_class = per_class_total.get(idx, 0)
        correct_for_class = per_class_correct.get(idx, 0)
        acc = correct_for_class / max(total_for_class, 1)
        per_class_acc[name] = acc

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    return overall_acc, per_class_acc, cm


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("evaluate")

    device = get_device()
    logger.info("Using device: %s", device)

    processed_dir = Path(args.data_dir)
    test_loader, class_names = create_test_loader(
        processed_data_dir=processed_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model, ckpt_class_names = load_model_for_inference(
        model_path=args.model_path,
        device=device,
        num_classes_fallback=len(class_names),
    )

    # Prefer checkpoint class names if lengths match
    if len(ckpt_class_names) == len(class_names):
        class_names = ckpt_class_names

    overall_acc, per_class_acc, cm = evaluate_model(
        model=model,
        dataloader=test_loader,
        device=device,
        class_names=class_names,
    )

    logger.info("Overall Test Accuracy: %.4f", overall_acc)
    logger.info("Per-class Accuracy:")
    for name, acc in per_class_acc.items():
        logger.info("  %s: %.4f", name, acc)

    logger.info("Confusion Matrix:")
    logger.info("\n%s", cm)

    # Save metrics
    metrics_rows = []
    for name, acc in per_class_acc.items():
        metrics_rows.append({"class": name, "accuracy": acc})
    df = pd.DataFrame(metrics_rows)
    df.loc[len(df)] = {"class": "overall", "accuracy": overall_acc}

    df.to_csv(METRICS_CSV_PATH, index=False)
    df.to_json(METRICS_JSON_PATH, orient="records", indent=2)
    logger.info("Metrics saved to %s and %s", METRICS_CSV_PATH, METRICS_JSON_PATH)


if __name__ == "__main__":
    main()
