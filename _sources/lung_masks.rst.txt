Lung Mask Segmentation
======================

This notebook demonstrates the **deep learning-based approach for lung segmentation** using a **UNet model**. The segmentation model is trained on medical images with **binary lung masks**, leveraging `torchvision`, `PIL`, and `torch`.

**Objectives:**
- Load and preprocess lung segmentation dataset.
- Implement a **UNet** model for medical image segmentation.
- Train the model with **Binary Cross-Entropy loss**.
- Visualize predictions on sample images.

.. note::
   This notebook is part of the Pneumonia Detection pipeline.

----

Import Libraries
----------------
This section loads necessary libraries for **deep learning, image processing, and dataset handling**.

.. code-block:: python

   import json
   import torch.nn as nn
   import torch
   import torch.optim as optim
   from torchvision import transforms
   from torch.utils.data import Dataset
   from PIL import Image, ImageDraw
   import numpy as np
   import matplotlib.pyplot as plt

----

Dataset Preparation
-------------------
This dataset consists of **lung X-ray images** with **manually annotated segmentation masks** stored in a JSON file ('.data/masked/annotated/annotations.json').

**Dataset format:**
- JSON file contains **polygon coordinates** outlining lung regions.

.. code-block:: javascript

   {
     "IM-0007-0001.jpeg408508": {
       "filename": "IM-0007-0001.jpeg",
       "size": 408508,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               217,
               361,
               523,
               795,
               954,
               1160,
               1345,
               1562,
               1728,
               1830,
               1830,
               1862,
               1806,
               217,
               153,
               183,
               219
             ],
             "all_points_y": [
               712,
               325,
               206,
               151,
               206,
               210,
               170,
               227,
               472,
               844,
               1186,
               1585,
               1815,
               1811,
               1486,
               984,
               708
             ]
           },
           "region_attributes": {
             "class_labels": "lungs"
           }
         }
       ],
       "file_attributes": {}
     },
     "IM-0010-0001.jpeg299115": {
       "filename": "IM-0010-0001.jpeg",
       "size": 299115,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               273,
               347,
               461,
               605,
               736,
               886,
               1083,
               1199,
               1317,
               1507,
               1621,
               1705,
               228,
               238,
               271
             ],
             "all_points_y": [
               696,
               470,
               247,
               133,
               112,
               171,
               145,
               138,
               195,
               506,
               866,
               1469,
               1470,
               1083,
               691
             ]
           },
           "region_attributes": {
             "class_labels": "lungs"
           }
         }
       ],
       "file_attributes": {}
     },
     "IM-0022-0001.jpeg82987": {
       "filename": "IM-0022-0001.jpeg",
       "size": 82987,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               176,
               191,
               225,
               272,
               320,
               383,
               444,
               552,
               619,
               690,
               736,
               889,
               994,
               1117,
               1178,
               1206,
               175
             ],
             "all_points_y": [
               609,
               424,
               304,
               198,
               98,
               29,
               6,
               6,
               18,
               8,
               1,
               3,
               105,
               290,
               457,
               615,
               615
             ]
           },
           "region_attributes": {
             "class_labels": "lungs\n"
           }
         },
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               340,
               330
             ],
             "all_points_y": [
               476,
               501
             ]
           },
           "region_attributes": {
             "class_labels": ""
           }
         }
       ],
       "file_attributes": {}
     },
     "IM-0023-0001.jpeg306362": {
       "filename": "IM-0023-0001.jpeg",
       "size": 306362,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               384,
               486,
               612,
               720,
               847,
               1122,
               1253,
               1433,
               1627,
               1747,
               1787,
               1807,
               272,
               290,
               322,
               380
             ],
             "all_points_y": [
               602,
               367,
               185,
               94,
               92,
               131,
               74,
               110,
               361,
               677,
               970,
               1381,
               1381,
               1070,
               847,
               596
             ]
           },
           "region_attributes": {
             "class_labels": "lungs\n"
           }
         }
       ],
       "file_attributes": {}
     },
     "IM-0049-0001.jpeg173441": {
       "filename": "IM-0049-0001.jpeg",
       "size": 173441,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               272,
               310,
               371,
               459,
               693,
               818,
               1001,
               1105,
               1350,
               1453,
               1489,
               1489,
               263,
               273
             ],
             "all_points_y": [
               681,
               517,
               383,
               264,
               94,
               120,
               113,
               125,
               322,
               544,
               837,
               1056,
               1055,
               676
             ]
           },
           "region_attributes": {
             "class_labels": "lungs"
           }
         }
       ],
       "file_attributes": {}
     },
     "IM-0065-0001.jpeg122201": {
       "filename": "IM-0065-0001.jpeg",
       "size": 122201,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               192,
               180,
               184,
               228,
               315,
               460,
               662,
               760,
               946,
               1124,
               1291,
               1342,
               1337,
               191
             ],
             "all_points_y": [
               1015,
               893,
               744,
               467,
               269,
               128,
               111,
               142,
               109,
               183,
               414,
               946,
               1017,
               1015
             ]
           },
           "region_attributes": {
             "class_labels": "lungs\n"
           }
         }
       ],
       "file_attributes": {}
     },
     "IM-0075-0001.jpeg157190": {
       "filename": "IM-0075-0001.jpeg",
       "size": 157190,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               290,
               417,
               526,
               596,
               764,
               861,
               975,
               1120,
               1250,
               1421,
               1531,
               1586,
               235,
               241,
               286
             ],
             "all_points_y": [
               488,
               239,
               94,
               42,
               41,
               80,
               54,
               39,
               104,
               315,
               608,
               1063,
               1062,
               747,
               486
             ]
           },
           "region_attributes": {
             "class_labels": "lungs\n"
           }
         }
       ],
       "file_attributes": {}
     },
     "IM-0086-0001.jpeg231577": {
       "filename": "IM-0086-0001.jpeg",
       "size": 231577,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               171,
               238,
               290,
               359,
               453,
               724,
               960,
               1123,
               1254,
               1500,
               1678,
               1763,
               1861,
               1888,
               128,
               164
             ],
             "all_points_y": [
               1121,
               830,
               574,
               390,
               239,
               73,
               96,
               109,
               68,
               120,
               246,
               424,
               892,
               1456,
               1452,
               1116
             ]
           },
           "region_attributes": {
             "class_labels": "lungs\n"
           }
         }
       ],
       "file_attributes": {}
     },
     "IM-0095-0001.jpeg176391": {
       "filename": "IM-0095-0001.jpeg",
       "size": 176391,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               264,
               301,
               423,
               578,
               715,
               828,
               926,
               1018,
               1162,
               1368,
               1535,
               1615,
               1641,
               1654,
               225,
               233,
               265
             ],
             "all_points_y": [
               725,
               561,
               362,
               187,
               135,
               133,
               150,
               106,
               106,
               319,
               554,
               834,
               1075,
               1303,
               1308,
               1069,
               724
             ]
           },
           "region_attributes": {
             "class_labels": "lungs\n"
           }
         }
       ],
       "file_attributes": {}
     },
     "IM-0145-0001.jpeg505771": {
       "filename": "IM-0145-0001.jpeg",
       "size": 505771,
       "regions": [
         {
           "shape_attributes": {
             "name": "polyline",
             "all_points_x": [
               239,
               281,
               376,
               509,
               638,
               737,
               871,
               958,
               1083,
               1252,
               1376,
               1451,
               1512,
               166,
               210,
               241
             ],
             "all_points_y": [
               647,
               538,
               365,
               214,
               185,
               211,
               197,
               145,
               177,
               376,
               607,
               835,
               1152,
               1152,
               753,
               641
             ]
           },
           "region_attributes": {
             "class_labels": "lungs\n"
           }
         }
       ],
       "file_attributes": {}
     }
   }

- Each image is mapped to a corresponding mask.
.. code-block:: python

   class LungSegmentationDataset(Dataset):
       def __init__(self, image_folder, json_path, transform=None, target_size=(256, 256)):
           self.image_folder = image_folder
           self.transform = transform
           self.target_size = target_size

           # Load JSON annotations
           with open(json_path, 'r') as f:
               self.annotations = json.load(f)

           self.image_files = list(self.annotations.keys())

       def __len__(self):
           return len(self.image_files)

       def __getitem__(self, idx):
           # Load image and corresponding annotation
           image_info = self.annotations[self.image_files[idx]]
           image_path = os.path.join(self.image_folder, image_info["filename"])
           image = Image.open(image_path).convert("RGB")

           # Create binary mask
           mask = Image.new("L", image.size)
           draw = ImageDraw.Draw(mask)
           for region in image_info["regions"]:
               points = list(zip(region["shape_attributes"]["all_points_x"],
                                 region["shape_attributes"]["all_points_y"]))
               draw.polygon(points, outline=1, fill=1)

           # Resize image and mask
           image = image.resize(self.target_size, Image.BILINEAR)
           mask = mask.resize(self.target_size, Image.NEAREST)

           # Apply transformations
           if self.transform:
               image = self.transform(image)

           mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)  # Add channel dimension
           return image, mask

----

UNet Model Architecture
-----------------------
This model is a **fully convolutional neural network** designed for segmentation tasks. It consists of:
- **Encoder (Downsampling Path):** Captures contextual information.
- **Bottleneck Layer:** Connects encoder and decoder.
- **Decoder (Upsampling Path):** Restores spatial resolution for segmentation.

.. code-block:: python

   class UNet(nn.Module):
       def __init__(self, in_channels=3, out_channels=1):
           super(UNet, self).__init__()

           def conv_block(in_channels, out_channels):
               return nn.Sequential(
                   nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
                   nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                   nn.ReLU(inplace=True),
               )

           self.enc1 = conv_block(in_channels, 64)
           self.enc2 = conv_block(64, 128)
           self.enc3 = conv_block(128, 256)
           self.enc4 = conv_block(256, 512)

           self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
           self.bottleneck = conv_block(512, 1024)

           self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
           self.dec4 = conv_block(1024, 512)
           self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
           self.dec3 = conv_block(512, 256)
           self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
           self.dec2 = conv_block(256, 128)
           self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
           self.dec1 = conv_block(128, 64)

           self.final = nn.Conv2d(64, out_channels, kernel_size=1)

       def forward(self, x):
           enc1 = self.enc1(x)
           enc2 = self.enc2(self.pool(enc1))
           enc3 = self.enc3(self.pool(enc2))
           enc4 = self.enc4(self.pool(enc3))

           bottleneck = self.bottleneck(self.pool(enc4))

           up4 = self.up4(bottleneck)
           dec4 = self.dec4(torch.cat([up4, enc4], dim=1))
           up3 = self.up3(dec4)
           dec3 = self.dec3(torch.cat([up3, enc3], dim=1))
           up2 = self.up2(dec3)
           dec2 = self.dec2(torch.cat([up2, enc2], dim=1))
           up1 = self.up1(dec2)
           dec1 = self.dec1(torch.cat([up1, enc1], dim=1))

           return torch.sigmoid(self.final(dec1))

----

Training Process
----------------
The model is trained using **Binary Cross-Entropy loss** for 200 epochs.

.. code-block:: python

   optimizer = optim.Adam(model.parameters(), lr=1e-4)
   criterion = nn.BCELoss()

   for epoch in range(200):
       model.train()
       epoch_loss = 0
       for images, masks in train_loader:
           images, masks = images.to(device), masks.to(device)
           outputs = model(images)
           loss = criterion(outputs, masks)

           optimizer.zero_grad()
           loss.backward()
           optimizer.step()

           epoch_loss += loss.item()

       print(f"Epoch {epoch + 1}/200, Loss: {epoch_loss:.4f}")

----

Visualizing Segmentation Results
--------------------------------
The predictions are **superimposed** on the original images.

.. code-block:: python

   def visualize_prediction(image, mask_pred):
       mask_pred = mask_pred / mask_pred.max()
       image_np = np.array(image) / 255.0  # Normalize to [0, 1]
       premultiplied_image = image_np * mask_pred[..., None]

       fig, axes = plt.subplots(1, 2, figsize=(12, 6))
       axes[0].imshow(image_np)
       axes[0].set_title("Original Image")
       axes[0].axis("off")

       axes[1].imshow(premultiplied_image)
       axes[1].set_title("Masked Image (Predicted Lung)")
       axes[1].axis("off")

       plt.tight_layout()
       plt.show()
