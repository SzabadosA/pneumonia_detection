# Pneumonia Detection System Documentation

## Overview
The Pneumonia Detection System is a machine learning-based application designed to classify chest X-ray images as either **Normal** or **Pneumonia-positive**. It uses a deep learning model trained on medical datasets to provide accurate and reliable predictions.

## Installation
### Prerequisites
Ensure that you have the following installed:
- Python 3.8+
- Pip (Python package manager)
- Virtual environment (optional but recommended)

### Setup Instructions
To install the dependencies, run:
```bash
python install.py
```
Alternatively, you can manually install the required dependencies using:
```bash
pip install -r requirements.txt
```

## Project Structure
```
├── README.md                # Project overview
├── Research_Report.pdf      # Detailed research documentation
├── install.py               # Installation script
├── interface.py             # Main interface for user interaction
├── code/                    # Source code directory
│   ├── __init__.py          # Package initialization
│   ├── batch_process.py     # Batch processing for multiple images
│   ├── classifier.py        # Core classification logic
│   ├── classify_random_images.py # Script to classify random test images
│   ├── custom_checkpoint.py # Custom model checkpoint logic
│   ├── dataloader.py        # Data preprocessing and loading
│   ├── model.py             # Deep learning model definition
│   ├── train.py             # Model training script
│   ├── visualize.py         # Visualization utilities
│   ├── grad_cam.py          # Grad-CAM implementation for explainability
└── dataset/                 # Directory for input data (not included in repo)
```

## Code Documentation

### `install.py`
**Purpose:** Installs all required dependencies for the project using Conda and Pip.

**Code Summary:**
```python
import os

packages = [
    "pytorch torchvision torchaudio pytorch-cuda=11.8",
    "pytorch-lightning",
    "torchmetrics",
    "tensorboard",
    "grad-cam",
    "albumentations",
    "opencv",
    "matplotlib",
    "torchcam",
    "segmentation-models-pytorch",
    "transformers",
]

command = "conda install -y -c pytorch -c nvidia -c conda-forge " + " ".join(packages)
os.system(command)
os.system("pip install gradio")
```

**Explanation:**
- Defines a list of required packages for deep learning, data augmentation, and visualization.
- Uses `os.system` to install packages via Conda.
- Installs Gradio separately using Pip for the web-based interface.

**Usage:**
```bash
python install.py
```

### `interface.py`
**Purpose:** Provides a user interface for interacting with the pneumonia detection system.

### `batch_process.py`
**Purpose:** Allows processing of multiple images at once.

### `classifier.py`
**Purpose:** Implements the core classification logic.
- `predict(image_path: str) -> str`: Classifies an input image as "Normal" or "Pneumonia".
- `load_model()`: Loads the trained deep learning model.

### `dataloader.py`
**Purpose:** Handles data loading and preprocessing.
- `load_dataset(dataset_path: str) -> torch.utils.data.Dataset`: Loads and preprocesses the dataset.

### `train.py`
**Purpose:** Provides functionality for training the deep learning model.
- `train_model()`: Initiates model training.

## Usage
### Running the Interface
To start the pneumonia detection system, run:
```bash
python interface.py
```
This will launch a command-line or graphical interface (depending on implementation) to allow users to upload X-ray images and receive predictions.

### Running Batch Processing
To classify a batch of X-ray images stored in a folder:
```bash
python code/batch_process.py --input_folder /path/to/images
```

## Model Training
If you want to retrain the model using new data, use the training script:
```bash
python code/train.py --epochs 10 --batch_size 32
```
Modify the hyperparameters as needed.

## Explainability (Grad-CAM)
To generate Grad-CAM visualizations for model interpretability:
```bash
python code/grad_cam.py --image_path /path/to/image.jpg
```

## Contributions
- To contribute to this project, fork the repository and create a pull request.

## License
This project is licensed under [MIT License](LICENSE).

