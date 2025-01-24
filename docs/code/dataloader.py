from torch.utils.data import Dataset
from PIL import Image
import os

class PneumoniaDataset(Dataset):
    """
    Custom dataset for loading pneumonia classification images.

    This dataset loads images from a directory structure where images are
    stored in subdirectories named according to their class labels (e.g.,
    'PNEUMONIA' and 'NORMAL').

    Args:
        root_dir (str): Root directory containing class subdirectories.
        transform (callable, optional): A function/transform to apply to the images.
        resolution (int, optional): Image resolution for resizing (default: 256).
    """
    def __init__(self, root_dir, transform=None, resolution=256):
        self.root_dir = root_dir
        self.transform = transform
        self.resolution = resolution

        # Gather all image paths and labels
        self.image_paths = []
        self.labels = []
        for class_name in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                label = 1 if class_name == "PNEUMONIA" else 0
                for img_name in os.listdir(class_path):
                    if img_name.endswith(('.jpg', '.jpeg', '.png')):  # Check valid image files
                        self.image_paths.append(os.path.join(class_path, img_name))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        img_path = self.image_paths[index]
        img = Image.open(img_path).convert('RGB')
        label = self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, label, img_path
