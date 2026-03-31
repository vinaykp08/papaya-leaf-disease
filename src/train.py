import argparse
import logging
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from .config import (
    DEFAULT_MODEL_PATH,
    PROCESSED_DATA_DIR,
    PROJECT_CLASS_NAMES,
)
from .data_loader import create_dataloaders
from .model import create_model
from .utils import ensure_dir, get_device, set_seed, setup_logging, to_python_float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Papaya Leaf Disease Detection Model"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(PROCESSED_DATA_DIR),
        help="Path to processed data directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--output-model-path",
        type=str,
        default=str(DEFAULT_MODEL_PATH),
        help="Path to save best model",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of dataloader workers",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Train", leave=False)
    for inputs, labels in loop:
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += to_python_float(loss) * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

    epoch_loss = total_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def evaluate(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += to_python_float(loss) * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == labels).item()
            total += labels.size(0)

    epoch_loss = total_loss / max(total, 1)
    epoch_acc = correct / max(total, 1)
    return epoch_loss, epoch_acc


def main() -> None:
    args = parse_args()
    setup_logging()
    logger = logging.getLogger("train")

    set_seed(args.seed)
    device = get_device()
    logger.info("Using device: %s", device)

    data_dir = Path(args.data_dir)
    train_loader, val_loader, _, class_names = create_dataloaders(
        processed_data_dir=data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    logger.info("Detected classes from data: %s", class_names)
    if len(class_names) != len(PROJECT_CLASS_NAMES):
        logger.warning(
            "Number of classes in data (%d) does not match PROJECT_CLASS_NAMES (%d). "
            "Using data-derived class names.",
            len(class_names),
            len(PROJECT_CLASS_NAMES),
        )

    num_classes = len(class_names)
    model = create_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=7, gamma=0.1)

    best_val_acc = 0.0
    output_model_path = Path(args.output_model_path)
    ensure_dir(output_model_path.parent)

    for epoch in range(1, args.epochs + 1):
        logger.info("Epoch %d/%d", epoch, args.epochs)

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        logger.info(
            "Epoch %d | Train Loss: %.4f | Train Acc: %.4f | "
            "Val Loss: %.4f | Val Acc: %.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            logger.info("New best val acc: %.4f. Saving model.", best_val_acc)
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": class_names,
            }
            torch.save(checkpoint, output_model_path)

        scheduler.step()

    logger.info("Training finished. Best val acc: %.4f", best_val_acc)


if __name__ == "__main__":
    main()
