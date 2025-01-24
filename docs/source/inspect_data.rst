Inspecting Dataset for Pneumonia Classification
===============================================

This notebook helps in **exploring the dataset** and **visualizing sample images** to ensure correctness before training a deep learning model.

**Objectives:**
- Load dataset images from different categories.
- Display sample images from both **infected** and **healthy** cases.
- Verify dataset integrity.

.. note::
   This notebook is part of the **Pneumonia Detection Pipeline**.

----

Import Libraries
----------------
This section loads necessary dependencies for **image processing and visualization**.

.. code-block:: python

   import os
   from IPython import get_ipython
   get_ipython().run_line_magic("matplotlib", "inline")
   import pathlib
   import cv2
   import matplotlib.pyplot as plt
   from code_pn.project_globals import DATADIR

   # Define dataset categories
   categories = ['train', 'val', 'test']

----

Plot Examples from Dataset
--------------------------
This function **randomly selects images** from different dataset categories.

.. code-block:: python

   import random

   def plot_random_images(category='train', class_name='PNEUMONIA'):
       path = DATADIR / category / class_name
       files = os.listdir(path)
       sample_files = random.sample(files, min(5, len(files)))  # Select up to 5 random images

       fig, axes = plt.subplots(1, len(sample_files), figsize=(15, 5))
       for idx, file in enumerate(sample_files):
           image = cv2.imread(os.path.join(path, file))
           axes[idx].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
           axes[idx].set_title(file)
           axes[idx].axis('off')

       plt.suptitle(f"Random Samples from {category} - {class_name}")
       plt.show()

----

Show Example of Infected Lung
-----------------------------
This section **displays a sample image** from the `PNEUMONIA` class.

.. code-block:: python

   path = DATADIR / 'train' / 'PNEUMONIA'
   sample_image = cv2.imread(os.path.join(path, os.listdir(path)[0]))
   plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
   plt.title(f'Sample from training data - PNEUMONIA')
   plt.axis('off')
   plt.show()

----

Show Example of Healthy Lung
----------------------------
This section **displays a sample image** from the `NORMAL` class.

.. code-block:: python

   path = DATADIR / 'train' / 'NORMAL'
   sample_image = cv2.imread(os.path.join(path, os.listdir(path)[0]))
   plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
   plt.title(f'Sample from training data - NORMAL')
   plt.axis('off')
   plt.show()

----

Conclusion
----------
- This notebook **verifies the dataset integrity** by displaying images from different categories.
- Helps to **ensure that images are correctly labeled** before training a deep learning model.
- **Next step:** Prepare dataset preprocessing for model training.

----

