from pathlib import Path
from typing import Dict, List

# Base directories
BASE_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = BASE_DIR / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"
SAMPLE_DATA_DIR: Path = DATA_DIR / "sample"
MODEL_DIR: Path = BASE_DIR / "models"
DEFAULT_MODEL_PATH: Path = MODEL_DIR / "best_model.pth"
METRICS_CSV_PATH: Path = BASE_DIR / "metrics.csv"
METRICS_JSON_PATH: Path = BASE_DIR / "metrics.json"

# Project-level class names (order used for reporting / UI if checkpoint does not override)
PROJECT_CLASS_NAMES: List[str] = [
    "healthy",
    "leaf_curl",
    "mosaic",
    "black_spot",
    "powdery_mildew",
]

# Mapping from raw dataset folder names to project labels
# Adjust if your raw folder names differ.
DATASET_TO_PROJECT_LABELS: Dict[str, str] = {
    "Healthy": "healthy",
    "Curl": "leaf_curl",
    "Mosaic": "mosaic",
    "Bacterial_Spot": "black_spot",
    "Ringspot": "powdery_mildew",
}

# Train/val/test split ratios
TRAIN_RATIO: float = 0.7
VAL_RATIO: float = 0.15
TEST_RATIO: float = 0.15

# Image settings
IMAGE_SIZE: int = 224
