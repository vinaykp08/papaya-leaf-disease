"""
Simple smoke test to ensure imports and basic model construction work.

Run with:
    python -m tests.test_imports
"""

from pathlib import Path

import torch

from src.config import PROCESSED_DATA_DIR
from src.data_loader import create_dataloaders
from src.model import create_model


def main() -> None:
    print("=== Running import and construction tests ===")

    # Test model creation
    model = create_model(num_classes=5, pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        output = model(dummy_input)
    assert output.shape[1] == 5
    print("Model creation test passed.")

    # Test dataloader creation (only if processed data exists)
    if PROCESSED_DATA_DIR.exists():
        try:
            train_loader, val_loader, test_loader, class_names = create_dataloaders(
                processed_data_dir=PROCESSED_DATA_DIR,
                batch_size=2,
                num_workers=0,
            )
            print("Detected classes:", class_names)
            _ = next(iter(train_loader))
            print("Dataloader test passed.")
        except Exception as exc:
            print("Dataloader test failed:", exc)
    else:
        print(
            f"Processed data directory not found: {PROCESSED_DATA_DIR}. "
            "Skipping dataloader test."
        )

    print("All tests completed.")


if __name__ == "__main__":
    main()
