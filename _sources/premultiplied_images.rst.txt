Premultiplied Images Processing
===============================

This notebook processes **equalized medical images** by applying corresponding **segmentation masks**, using a **premultiplied alpha blending technique**. The final output ensures that only relevant regions of the image are visible.

**Objectives:**
- Load equalized images and corresponding segmentation masks.
- Resize masks to match input images.
- Apply **premultiplication** (pixel-wise multiplication).
- Save the resulting masked images in an organized directory.

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
   from PIL import Image
   import numpy as np

----

Defining the Processing Function
--------------------------------
The following function **iterates through a dataset**, applies **segmentation masks**, and saves the **premultiplied images**.

**Processing Steps:**
1. **Load equalized image** – Ensures input image is in **RGB format**.
2. **Locate and open corresponding mask** – Loads grayscale mask from the `masked` dataset.
3. **Resize mask** – Ensures it matches the input image resolution.
4. **Multiply image by mask** – Applies **pixel-wise multiplication**.
5. **Save output** – Stores masked image in an organized structure.

.. code-block:: python

   def process_images(equalized_dir, masked_dir, output_dir):
       for root, dirs, files in os.walk(equalized_dir):
           for file in files:
               if file.endswith(('.jpg', '.jpeg', '.png')):
                   # Get the relative path for maintaining folder structure
                   relative_path = os.path.relpath(root, equalized_dir)

                   # Paths to the equalized image and corresponding mask
                   equalized_path = os.path.join(root, file)
                   mask_path = os.path.join(masked_dir, relative_path, file)
                   mask_path = mask_path.replace(".jpeg", "_mask.png")  # Adjust for mask naming conventions

                   # Ensure the mask exists
                   if not os.path.exists(mask_path):
                       print(f"Mask not found for: {equalized_path}")
                       continue

                   # Open the equalized image and mask
                   equalized_image = Image.open(equalized_path).convert('RGB')
                   mask_image = Image.open(mask_path).convert('L')  # Convert mask to grayscale

                   # Resize the mask to match the image dimensions
                   mask_image = mask_image.resize(equalized_image.size, resample=Image.BILINEAR)

                   # Multiply the image and mask
                   equalized_array = np.array(equalized_image)
                   mask_array = np.array(mask_image) / 255.0  # Normalize mask to [0, 1]
                   masked_image = (equalized_array * mask_array[..., None]).astype(np.uint8)

                   # Create the output directory structure
                   output_folder = os.path.join(output_dir, relative_path)
                   os.makedirs(output_folder, exist_ok=True)

                   # Save the masked image
                   output_path = os.path.join(output_folder, file)
                   Image.fromarray(masked_image).save(output_path)

                   print(f"Processed and saved: {output_path}")

----

Processing the Dataset
----------------------
This section **initializes directory paths** and calls the `process_images` function.

.. code-block:: python

   # Define input and output directories
   equalized_dir = "../data/equalized"
   masked_dir = "../data/masked"
   output_dir = "../data/premultiplied"

   # Run the image processing pipeline
   process_images(equalized_dir, masked_dir, output_dir)

**Expected Outcome:**
- The script **iterates through all equalized images**.
- Masks are **applied correctly** to create **segmented versions**.
- The **final images are saved** in a structured directory.

----

Saving and Organizing Output
----------------------------
The processed images are saved to './data/premultiplied/':

**File Naming Conventions:**
- Images retain **original filenames**.
- Masked images are **automatically mapped**.

----

Example Output
--------------
The following visualization shows an **original image**, a **mask**, and the **final premultiplied result**.

.. code-block:: python

   def visualize_premultiplied(image_path, mask_path):
       image = Image.open(image_path).convert("RGB")
       mask = Image.open(mask_path).convert("L").resize(image.size, resample=Image.BILINEAR)

       # Apply mask
       mask_array = np.array(mask) / 255.0
       image_array = np.array(image)
       premultiplied = (image_array * mask_array[..., None]).astype(np.uint8)

       # Display results
       fig, axes = plt.subplots(1, 3, figsize=(12, 6))
       axes[0].imshow(image)
       axes[0].set_title("Original Image")
       axes[0].axis("off")

       axes[1].imshow(mask, cmap="gray")
       axes[1].set_title("Segmentation Mask")
       axes[1].axis("off")

       axes[2].imshow(premultiplied)
       axes[2].set_title("Premultiplied Image")
       axes[2].axis("off")

       plt.tight_layout()
       plt.show()

----


