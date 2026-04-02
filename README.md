# Papaya Leaf Disease Detection System

An end-to-end **Papaya Leaf Disease Detection System** using deep learning (PyTorch) and a Streamlit web UI.

The system:

- Trains a CNN (ResNet18) on papaya leaf images.
- Evaluates model performance (accuracy, per-class metrics, confusion matrix).
- Provides a web UI where you can upload papaya leaf images and get predicted disease labels.

---

## 1. Project Overview

### Goal

Detect papaya leaf diseases from images for **5 classes**:

- `healthy`
- `leaf_curl`
- `mosaic`
- `black_spot`
- `powdery_mildew`

> **Note on label mapping (important)**  
> We use a public dataset that does **not** have exactly these class names.  
> We map the dataset’s classes to our project labels as follows:

| Project label      | Dataset folder used              | Real disease meaning in dataset |
|--------------------|----------------------------------|---------------------------------|
| `healthy`          | `Healthy`                        | Healthy papaya leaves           |
| `leaf_curl`        | `Curl`                           | Curl disease                    |
| `mosaic`           | `Mosaic`                         | Mosaic disease                  |
| `black_spot`       | `Bacterial_Spot`                 | Bacterial spot                  |
| `powdery_mildew`   | `Ringspot`                       | Ringspot disease                |

So **class names in your model output follow the assignment** (healthy, leaf_curl, mosaic, black_spot, powdery_mildew), but internally:

- `black_spot` → bacterial spot images
- `powdery_mildew` → ringspot images

This limitation is purely due to the available public dataset.

---

## 2. Tech Stack

- **Language**: Python 3.10+
- **Deep Learning**: PyTorch, torchvision
- **Data Handling**: torchvision `ImageFolder`, PIL
- **UI**: Streamlit
- **Utilities**: numpy, pandas, scikit-learn, tqdm, logging
- **OS**: Windows or Linux

---

## 3. Dataset

We use the **Papaya Leaf Disease Image Dataset** (Mendeley Data):

"https://data.mendeley.com/datasets/3kwgxg4stb/1"

- Name: **Papaya Leaf Disease Image Dataset**
- Host: Mendeley Data
- Classes: Anthracnose, Bacterial Spot, Curl, Healthy, Mealybug, Mite Disease, Mosaic, Ringspot
- Approx. images: 3626 raw + 18130 augmented images

Search for **"Papaya Leaf Disease Image Dataset Mendeley"** to reach the dataset page, or use the DOI / link provided on Mendeley Data.

### 3.1. Downloading the dataset

1. Go to the dataset page.
2. Download the images archive (`.zip`).
3. Extract it so that you end up with subfolders per class inside:


# 1. Clone the repo (if using git) or copy the project folder
cd papaya-leaf-disease-detector

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
venv\Scripts\activate

# 3. Install dependencies
pip install --upgrade pip

pip install -r requirements.txt

# From project root
python scripts/prepare_data.py

#train dataset 
python -m src.train

#evaluate 
python -m src.evaluate

#run project 
streamlit run app/ui.py
