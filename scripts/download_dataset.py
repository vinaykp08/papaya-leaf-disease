"""
Helper script for downloading / preparing the Papaya Leaf Disease Image Dataset.

Because the dataset is hosted on Mendeley Data and may require manual login,
this script does NOT attempt to download it automatically.

Instead, it prints clear instructions to the user.
"""

from pathlib import Path

from src.config import RAW_DATA_DIR
from src.utils import ensure_dir


def main() -> None:
    ensure_dir(RAW_DATA_DIR)

    print("=== Papaya Leaf Disease Dataset Download Helper ===\n")
    print("1. Open your browser and search for:")
    print("   'Papaya Leaf Disease Image Dataset Mendeley'")
    print("   and go to the Mendeley Data page.")
    print()
    print("2. Download the dataset archive (e.g., .zip).")
    print()
    print(f"3. Extract the contents so that you have class folders inside:\n   {RAW_DATA_DIR}")
    print("   For example, the structure should look like:")
    print("   data/raw/")
    print("     Anthracnose/")
    print("     Bacterial_Spot/")
    print("     Curl/")
    print("     Healthy/")
    print("     Mealybug/")
    print("     Mite_Disease/")
    print("     Mosaic/")
    print("     Ringspot/")
    print()
    print(
        "4. If your extracted folder names differ, either rename them to match the above "
        "or update the DATASET_TO_PROJECT_LABELS mapping in src/config.py."
    )
    print()
    print("Once this is done, run:")
    print("   python scripts/prepare_data.py")
    print("to create train/val/test splits in data/processed/.")


if __name__ == "__main__":
    main()
