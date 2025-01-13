# Pneumonia Detection 
Medical image classification plays a vital role in healthcare, aiding in precise and timely disease diagnosis. This project focuses on developing a flexible machine learning-based classification system to evaluate and compare the performance of various models on the Chest X-ray Images (Pneumonia) dataset from Kaggle. The dataset consists of 5,863 labeled chest X-ray images divided into two categories: Normal and  Pneumonia.

The primary objective is to identify the best-performing model for pneumonia detection by leveraging transfer learning with pre-trained convolutional neural networks such as DenseNet131, EfficientNet, ResNet18, ResNet50, and exploring transformer-based methods like Visual Transformers (ViT). The project emphasizes fine-tuning these models to ensure optimal performance on the dataset.

To comprehensively evaluate model performance, metrics including Accuracy, F1-Score, Precision, Recall, Specificity, AUC/ROC curves, and confusion matrices are employed. Additionally, Grad-CAM is utilized for visualization and explainability, highlighting the critical lung regions influencing the models' predictions.

The ultimate goal is to build a robust and adaptable classifier written with Pytorch Lightning, capable of reliably distinguishing between healthy lungs and pneumonia-affected lungs. This project underscores the potential of deep learning models in advancing medical image analysis and supporting radiologists in early disease diagnosis through systematic model comparison and explainability.

![Description of Image](misc/plotresults.PNG)

# Dataset
The project is based on the public chest-xray-pneumonia kaggle dataset:
```
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
```
All images, pretrained models and misc can be downloaded at:
```
https://drive.google.com/drive/folders/1MIcgEqpcMU24N4lEFqmBIhDUXR6jAzv2?usp=sharing
```

All downloaded folders should be placed in the root of the project directory.
```
project/
├── checkpoints/                 
├── code/ 
├── data/ 
├── models/
├── notebooks/  
```
# Installation

It is recommended to set up a conda environment with the following dependencies:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install conda-forge::pytorch-lightning
conda install conda-forge::torchmetrics
conda install conda-forge::tensorboard
conda install conda-forge::grad-cam
conda install conda-forge::albumentations
conda install conda-forge::opencv
conda install conda-forge::matplotlib
conda install conda-forge::torchcam
conda install conda-forge::segmentation-models-pytorch
conda install conda-forge::transformers
```
