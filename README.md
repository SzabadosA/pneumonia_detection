# Pneumonia Detection 
This project aims to build a classifier utilizing transfer learning for Pneumonia Detection and comparing between popular pretrained models such as ResetNet18 and ImageNet.

# Dataset
The project is based on the public chest-xray-pneumonia kaggle dataset:
```
curl -L -o ./data/dataset.zip\ https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia
```
Unzip the test, train and val folder into data/raw
```
project/
├── data/                  # Dataset
│   ├── raw/
│   │   ├── test/
│   │   ├── val/
│   │   ├── train/
│   ├── processed/
```
# Installation

The recommended installation is to set up a conda environment with the following dependencies:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install conda-forge::pytorch-lightning
conda install conda-forge::torchmetrics
conda install conda-forge::tensorboard
conda install conda-forge::grad-cam
conda install conda-forge::albumentations
conda install conda-forge::opencv
conda install conda-forge::matplotlib
```

# Project Structure
```
project/
├── notebooks/             # For interactive exploration and demonstration
│   ├── checkpoints
│   ├── tb_logs
│   │   ├── pneumonia_classifier
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb
│   ├── gradcam_visualization.ipynb
│   ├── inference_demo.ipynb
├── code/                  # Reusable code modules
│   ├── data_preprocessing.py
│   ├── models.py
│   ├── training.py
│   ├── gradcam_visualization.py
│   ├── inference.py
├── data/                  # Dataset
│   ├── raw/
│   ├── processed/
├── checkpoints/           # Model checkpoints
│   ├── resnet50_best.ckpt
│   ├── custom_model_best.ckpt
├── results/               # Results and visualizations
│   ├── gradcam/
│   ├── metrics_summary.csv
│   ├── plots/
├── report/                # Final report
│   ├── main.tex
│   ├── references.bib
│   └── final_report.pdf
└── README.md
```