Equalizing Medical Images
==========================

This notebook **applies histogram equalization** to **medical X-ray images**, enhancing contrast for better visibility. Histogram equalization is particularly useful for **medical imaging**, where subtle variations in brightness can be diagnostically significant.

**Objectives:**
- Load medical images from the dataset.
- Apply **histogram equalization** to enhance contrast.
- Save the equalized images in a structured directory.

.. note::
   This notebook is part of the **Pneumonia Detection Pipeline**.

----

Import Libraries
----------------
This section loads necessary dependencies for **image processing and file handling**.

.. code-block:: python

   import os
   from IPython import get_ipython
   get_ipython().run_line_magic("matplotlib", "inline")
   from PIL import Image, ImageOps

----

Defining the Equalization Function
----------------------------------
The function **iterates through the dataset**, applies **histogram equalization**, and saves the **enhanced images**.

**Processing Steps:**
1. **Load original image** – Convert to **RGB**.
2. **Apply histogram equalization** – Using `ImageOps.equalize()`.
3. **Save enhanced image** – Store it in an organized directory.

.. code-block:: python

   def equalize_image(input_folder, output_folder):
       # Ensure the output folder exists
       os.makedirs(output_folder, exist_ok=True)

       for root, _, files in os.walk(input_folder):
           for file in files:
               # Check for image files
               if file.endswith((".jpg", ".jpeg", ".png")):
                   input_path = os.path.join(root, file)

                   # Create the corresponding output path
                   relative_path = os.path.relpath(root, input_folder)
                   output_dir = os.path.join(output_folder, relative_path)
                   os.makedirs(output_dir, exist_ok=True)

                   # Load the image
                   image = Image.open(input_path).convert("RGB")

                   # Apply histogram equalization
                   equalized_image = ImageOps.equalize(image)

                   # Save the equalized image
                   output_path = os.path.join(output_dir, file)
                   equalized_image.save(output_path)

                   print(f"Processed: {input_path} -> {output_path}")

----

Processing the Dataset
----------------------
This section **initializes directory paths** and calls the `equalize_image` function.

.. code-block:: python

   # Define input and output folders
   input_folder = "../data/reordered"
   output_folder = "../data/equalized"

   # Apply histogram equalization to all images
   equalize_image(input_folder, output_folder)

**Expected Outcome:**
- The script **iterates through all images**.
- Each image undergoes **histogram equalization**.
- The **final images are saved** in an organized directory.

----

Saving and Organizing Output
----------------------------
The processed images are saved to './data/equalized', maintaining the **original folder structure**.


**File Naming Conventions:**
- Images retain **original filenames**.
- Output structure mirrors **original dataset hierarchy**.

----

Example Output
--------------
This visualization compares an **original image** with its **equalized version**.

.. code-block:: python

   import matplotlib.pyplot as plt

   def visualize_equalization(image_path):
       original = Image.open(image_path).convert("RGB")
       equalized = ImageOps.equalize(original)

       fig, axes = plt.subplots(1, 2, figsize=(10, 5))
       axes[0].imshow(original)
       axes[0].set_title("Original Image")
       axes[0].axis("off")

       axes[1].imshow(equalized)
       axes[1].set_title("Equalized Image")
       axes[1].axis("off")

       plt.tight_layout()
       plt.show()

----

Conclusion
----------
- The **equalized images** provide **improved contrast**, aiding visualization.
- The processed images are **stored efficiently** for further use in segmentation models.
- This **preprocessing step** enhances the quality of images used in training models.

----



