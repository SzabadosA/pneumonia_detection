Training Structure
===========================

This example demonstrates the full training pipeline for pneumonia classification. The workflow is demonstrated utilizing the CNNPneumoniaClassifier with a ResNet50-based classifier with gradual unfreezing. Other models and classifiers (e.g. the ViTClassifier) can be adopted accordingly.

**Contents:**
- Import necessary libraries
- Check GPU availability
- Define model architecture and configurations
- Visualize sample images from datasets
- Train the model with gradual unfreezing
- Evaluate model performance using multiple metrics

.. note::
   This notebook is part of the Pneumonia Detection pipeline. It uses `torchvision`, `torch`, `matplotlib`, and `scikit-learn` for deep learning and visualization.

----

Import Libraries
----------------
This section loads required libraries for data preprocessing, model training, and visualization.

.. code-block:: python

   import os
   from IPython import get_ipython
   from torchvision.transforms import Compose, Resize, InterpolationMode, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip
   import torch
   from code.classifier import CNNPneumoniaClassifier, Config
   from code.plot_random_image import plot_random_image_from_loader
   from code.classify_random_images import plot_random_images_with_labels
   from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
   import matplotlib.pyplot as plt

----

Check GPU Availability
----------------------
Ensures that the system is using a **CUDA-enabled GPU** for acceleration.

.. code-block:: python

   gpu_info = !nvidia-smi
   gpu_info = '\n'.join(gpu_info)
   if gpu_info.find('failed') >= 0:
       print('Not connected to a GPU')
   else:
       print(gpu_info)

----

Setup Model
-----------
Initializes the **ResNet50-based pneumonia classifier** with a **custom training configuration**.

.. code-block:: python

   config = Config(
       backbone_name="resnet50",
       transfer_learning=True,
       learning_rate=1e-4,
       batch_size=20,
       max_epochs=50,
       weight_decay=1e-4,
       dropout=0.2,
       num_workers=16,
       model_name="ResNet50_gradual_unfreeze",
       version="001",
       optimizer_name="sgd",
       use_class_weights=True,
       image_res=224,
       patience=10,
       image_type=3,
       gradually_unfreeze=True,
       unfreeze_interval=5,
       num_layers_to_unfreeze=2,
       frozen_lr=1e-6,  # Learning rate for frozen layers
       unfrozen_lr=1e-5  # Learning rate for unfrozen layers
   )

   model = CNNPneumoniaClassifier(config)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = model.to(device)

----

Plot Sample Images
------------------
Displays random images from different datasets to ensure data is loaded correctly.

.. code-block:: python

   plot_random_image_from_loader(train_loader)

----

Train Model
-----------
Trains the model using **gradual unfreezing**.

.. code-block:: python

   trainer.fit(model, train_dataloader, val_dataloader)

.. note::
   Gradual unfreezing starts by training only the final layers, then progressively unfreezing earlier layers every **5 epochs**.

----

Load Model
----------
Loads a **previously trained model checkpoint**.

.. code-block:: python

   model.load_from_checkpoint("checkpoints/resnet50_unfreeze_epoch10.ckpt")

----

Test Model
----------
Evaluates the model on the **test dataset** and computes accuracy.

.. code-block:: python

   test_accuracy = evaluate(model, test_loader)
   print(f"Test Accuracy: {test_accuracy:.2f}%")

----

Plot Confusion Matrix
---------------------
Visualizes classification performance using a **confusion matrix**.

.. code-block:: python

   y_true, y_pred = get_predictions(model, test_loader)
   cm = confusion_matrix(y_true, y_pred)
   ConfusionMatrixDisplay(cm).plot()

----

Plot ROC/AUC Score
------------------
Calculates the **ROC-AUC** score for model performance.

.. code-block:: python

   roc_auc = compute_roc_auc(model, test_loader)
   print(f"ROC-AUC Score: {roc_auc:.4f}")

----

Plot GradCAM
------------
Generates a **Grad-CAM heatmap** for model explainability.

.. code-block:: python

   plot_gradcam(model, sample_image)

----

Evaluate Metrics in TensorBoard
-------------------------------
Logs training metrics in **TensorBoard** for better visualization.

.. code-block:: python

   %reload_ext tensorboard
   %load_ext tensorboard
   %tensorboard --logdir logs/

----

Classify Random Images
----------------------
Runs model predictions on random test images.

.. code-block:: python

   plot_random_images_with_labels(model, test_loader)
