import random
import shutil
from pathlib import Path
from typing import Dict, List

from src.config import (
    DATASET_TO_PROJECT_LABELS,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    SAMPLE_DATA_DIR,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)
from src.utils import ensure_dir, set_seed


def collect_image_paths(
    src_dir: Path,
) -> List[Path]:
    """Collect all image file paths in a directory (non-recursive)."""
    image_paths: List[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        image_paths.extend(src_dir.glob(ext))
    return image_paths


def split_dataset(
    image_paths: List[Path],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> tuple[List[Path], List[Path], List[Path]]:
    """Split paths into train/val/test according to ratios."""
    total = len(image_paths)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_paths = image_paths[:train_end]
    val_paths = image_paths[train_end:val_end]
    test_paths = image_paths[val_end:]
    return train_paths, val_paths, test_paths


def copy_files(paths: List[Path], dst_dir: Path) -> None:
    """Copy image files into destination directory."""
    ensure_dir(dst_dir)
    for src_path in paths:
        dst_path = dst_dir / src_path.name
        shutil.copy2(src_path, dst_path)


def main() -> None:
    set_seed(42)

    # Ensure output dirs exist
    ensure_dir(PROCESSED_DATA_DIR)
    ensure_dir(SAMPLE_DATA_DIR)

    print("=== Preparing processed dataset ===")
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")

    if not RAW_DATA_DIR.exists():
        raise FileNotFoundError(
            f"RAW_DATA_DIR does not exist: {RAW_DATA_DIR}. "
            "Please place the raw dataset there or run scripts/download_dataset.py "
            "for instructions."
        )

    # project_classes will be unique values from mapping
    project_classes = sorted(set(DATASET_TO_PROJECT_LABELS.values()))
    print("Project classes:", project_classes)

    # Create output subdirectories
    for split in ("train", "val", "test"):
        for cls in project_classes:
            ensure_dir(PROCESSED_DATA_DIR / split / cls)

    # Mapping from project class to all file paths that should be used
    project_paths: Dict[str, List[Path]] = {cls: [] for cls in project_classes}

    # Collect paths from selected dataset folders
    for dataset_folder, project_label in DATASET_TO_PROJECT_LABELS.items():
        src_dir = RAW_DATA_DIR / dataset_folder
        if not src_dir.exists():
            raise FileNotFoundError(
                f"Expected raw folder does not exist: {src_dir}. "
                "Check your dataset extraction and/or folder names."
            )

        image_paths = collect_image_paths(src_dir)
        print(f"Found {len(image_paths)} images in {src_dir}")

        project_paths[project_label].extend(image_paths)

    # Shuffle and split per project class
    for project_label, paths in project_paths.items():
        print(f"\nProcessing class '{project_label}' with {len(paths)} images")
        random.shuffle(paths)
        train_paths, val_paths, test_paths = split_dataset(
            paths, TRAIN_RATIO, VAL_RATIO, TEST_RATIO
        )

        print(
            f"  Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}"
        )

        copy_files(train_paths, PROCESSED_DATA_DIR / "train" / project_label)
        copy_files(val_paths, PROCESSED_DATA_DIR / "val" / project_label)
        copy_files(test_paths, PROCESSED_DATA_DIR / "test" / project_label)

        # Also copy a few sample images for quick inspection
        sample_subset = paths[: min(5, len(paths))]
        sample_class_dir = SAMPLE_DATA_DIR / project_label
        copy_files(sample_subset, sample_class_dir)

    print("\nDone. Processed dataset created in:", PROCESSED_DATA_DIR)


if __name__ == "__main__":
    main()
