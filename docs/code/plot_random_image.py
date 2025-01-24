import matplotlib.pyplot as plt
import random
import torch
import os


def plot_random_image_from_loader(dataset, dataset_name, normalize=True):
    """
    Plots a random image from the given dataset with its label.

    Args:
        dataset (torch.utils.data.Dataset): The dataset to sample from.
        dataset_name (str): Name of the dataset (e.g., 'Train', 'Validation', 'Test') for display.
        normalize (bool): Whether to reverse normalization applied during preprocessing.

    """
    import numpy as np

    # Select a random index from the dataset.
    random_idx = random.randint(0, len(dataset) - 1)

    # Fetch the image, label, and file path using the random index.
    image, label, path = dataset[random_idx]

    # If the image is a PyTorch tensor, convert it to a NumPy array for visualization.
    if isinstance(image, torch.Tensor):
        if normalize:
            # Reverse normalization if the image was normalized during preprocessing.
            mean = torch.tensor([0.485, 0.456, 0.406])  # Mean used for normalization.
            std = torch.tensor([0.229, 0.224, 0.225])  # Std deviation used for normalization.
        else:
            # No normalization applied; use default mean and std.
            mean = torch.tensor([0.0, 0.0, 0.0])
            std = torch.tensor([1.0, 1.0, 1.0])

        # Reverse normalization to restore original pixel values.
        image = image * std[:, None, None] + mean[:, None, None]

        # Convert the image tensor from CHW (Channels, Height, Width) to HWC (Height, Width, Channels) format.
        image = image.permute(1, 2, 0).cpu().numpy()

    # Ensure pixel values are in the range [0, 1] for visualization.
    image = np.clip(image, 0, 1)

    # Plot the image using Matplotlib.
    plt.figure(figsize=(6, 6))
    plt.imshow(image)

    # Map the label to a class name (e.g., "NORMAL" or "PNEUMONIA").
    name = "NORMAL" if label == 0 else "PNEUMONIA"

    # Set the title with dataset name, file name, and label.
    plt.title(f"{dataset_name} - {os.path.basename(path)}\nLabel: {name}")

    # Remove axis lines for a cleaner display.
    plt.axis("off")

    # Display the plot.
    plt.show()
