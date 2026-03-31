import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def setup_logging(level: int = logging.INFO) -> None:
    """Configure basic logging."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    """Return CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def ensure_dir(path: Path) -> None:
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def to_python_float(value: Any) -> float:
    """Convert torch / numpy scalar to Python float."""
    if isinstance(value, (float, int)):
        return float(value)
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)
