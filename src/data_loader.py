from pathlib import Path
from typing import Tuple

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from .config import IMAGE_SIZE


def get_transforms(train: bool = True) -> transforms.Compose:
    """
    Get torchvision transforms for training or evaluation.
    """
    if train:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )


def create_dataloaders(
    processed_data_dir: Path,
    batch_size: int = 32,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """
    Create train, val, and test dataloaders from processed data directory.

    Args:
        processed_data_dir: Path to data/processed
        batch_size: batch size
        num_workers: number of subprocesses for data loading

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    train_dir = processed_data_dir / "train"
    val_dir = processed_data_dir / "val"
    test_dir = processed_data_dir / "test"

    train_dataset = datasets.ImageFolder(train_dir, transform=get_transforms(train=True))
    val_dataset = datasets.ImageFolder(val_dir, transform=get_transforms(train=False))
    test_dataset = datasets.ImageFolder(test_dir, transform=get_transforms(train=False))

    class_names = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, test_loader, class_names
