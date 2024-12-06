import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import random
import torch


import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import random
import torch
import os

import matplotlib.pyplot as plt
import torchvision.transforms.functional as F
import random
import torch
import os
import numpy as np


def plot_random_image_from_loader(dataset, dataset_name, normalize=True):
    import numpy as np

    # Select a random index
    random_idx = random.randint(0, len(dataset) - 1)

    # Fetch the image and label directly
    image, label, path = dataset[random_idx]


    # Convert tensor image to numpy for visualization
    if isinstance(image, torch.Tensor):
        # If normalized, reverse normalization (only if normalization is applied in transforms)
        # Uncomment if Normalize is used in transforms

        if normalize:
            mean = torch.tensor([0.485, 0.456, 0.406])
            std = torch.tensor([0.229, 0.224, 0.225])
        else:
            mean = torch.tensor([0.0, 0.0, 0.0])
            std = torch.tensor([1.0, 1.0, 1.0])
        image = image * std[:, None, None] + mean[:, None, None]
        # Convert to HWC format for plotting
        image = image.permute(1, 2, 0).cpu().numpy()
    # Clip values to [0, 1] for visualization
    image = np.clip(image, 0, 1)

    # Plot the image
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    name = "NORMAL" if label == 0 else "PNEUMONIA"
    plt.title(f"{dataset_name} - {os.path.basename(path)}\nLabel: {name}")
    plt.axis("off")
    plt.show()
