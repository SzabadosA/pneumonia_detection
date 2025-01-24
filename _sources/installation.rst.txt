Installation Guide
==================

This guide provides step-by-step instructions for installing and setting up the Pneumonia Detection system.

---

Prerequisites
-------------
Ensure the following dependencies are installed before proceeding:

- `Miniconda` or `Anaconda` (recommended for environment management)
- Python **3.12.3** (installed via Conda)
- A **GPU with CUDA support** (optional but recommended for deep learning)

---

Cloning the Repository
----------------------
First, clone the repository and navigate to the project directory:

.. code-block:: bash

   git clone https://github.com/SzabadosA/pneumonia_detection.git
   cd pneumonia-detection

---

Setting Up the Conda Environment
--------------------------------
Create and activate a new Conda environment (replace `<custom_name>` with a name of your choice, e.g., `pneumonia_detection`):

.. code-block:: bash

   conda create --name <custom_name> python=3.12.3
   conda activate <custom_name>

---

Installing Dependencies
-----------------------
There are two ways to install the required dependencies: using an **install script** or **manual installation**.

### **Option 1: Install Using the Script**
Run the `install.py` script within the activated Conda environment:

.. code-block:: bash

   python install.py

---

### **Option 2: Manual Installation**
Alternatively, install the dependencies manually using Conda and pip:

.. code-block:: bash

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
   pip install gradio

---

Running the Interface
---------------------
To start the **Gradio interface** for pneumonia detection, run:

.. code-block:: bash

   python interface.py

This will launch a user-friendly web interface where users can upload chest X-ray images and receive pneumonia predictions.

---

Project Structure
-----------------
The project follows the structure below:

.. code-block:: bash

   pneumonia-detection/
   │── Research_Report.pdf    # Project Research Report
   │── interface.py           # Main entry point for running the interface
   │── install.py             # Install Script for dependencies
   │── README.md              # Documentation
   │── checkpoints/           # (To be downloaded) Training checkpoints
   │── code/                  # Source code
   │── data/                  # (To be downloaded) Dataset folder
   │── misc/                  # Readme Image Files
   │── models/                # (To be downloaded) Saved models
   │── notebooks/             # Jupyter notebooks for exploration

---

Next Steps
----------
- **Run Interface**: Start the model inference with `python interface.py`
- **Explore Notebooks**: The source code for all models is available in the `./notebooks/` directory.
- **Review Research Report**: Read the provided research report for model performance analysis.

---

Contact
-------
For issues or inquiries, open an [issue on GitHub](https://github.com/SzabadosA/pneumonia_detection.git).

---

License
-------
This project is licensed under the **MIT License**.
