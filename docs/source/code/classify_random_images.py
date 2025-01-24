import matplotlib.pyplot as plt
import random
import numpy as np
import torch

def plot_random_images_with_labels(model, dataloader, num_images=20, seed=42):
    """
    Plots random images from the dataset with their actual labels, predicted labels, and filenames.

    Args:
        model: The trained model for predictions.
        dataloader: The dataloader for fetching the images.
        num_images: Number of images to display.
        seed: Random seed for reproducibility.
    """
    # Set random seed for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    classes = ["NORMAL", "PNEUMONIA"]

    # Set model to evaluation mode
    model.eval()

    # Fetch a random subset of images from the dataloader
    dataset = dataloader.dataset
    random_indices = random.sample(range(len(dataset)), num_images)

    fig, axes = plt.subplots(nrows=4, ncols=5, figsize=(20, 16))
    axes = axes.flatten()

    for i, idx in enumerate(random_indices):
        # Retrieve the raw image, label, and filename
        img, label, filename = dataset[idx]
        filename = filename.replace("\\", "/")
        parts = filename.split('/')
        formatted_filename = "/".join(parts[-3:])

        # Convert to batch format for prediction
        img_batch = img.unsqueeze(0).to(model.device)
        label = classes[int(label)]

        # Get model prediction
        output = model(img_batch)
        pred_label = classes[torch.argmax(output, dim=1).item()]

        # Prepare image for plotting (convert tensor to numpy)
        img_np = img.permute(1, 2, 0).cpu().numpy()  # Convert CHW to HWC
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())  # Normalize to [0, 1]

        # Display the image with labels
        axes[i].imshow(img_np)
        axes[i].set_title(
            f"Filename: {formatted_filename}\nActual: {label}, Predicted: {pred_label}",
            fontsize=10
        )
        axes[i].axis("off")

    # Hide unused axes if fewer than num_images are plotted
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()