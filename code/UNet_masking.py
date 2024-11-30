import os
import json
import numpy as np
import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw


class LungSegmentationDataset(Dataset):
    def __init__(self, image_folder, json_path, transform=None, target_size=(256, 256)):
        self.image_folder = image_folder
        self.transform = transform
        self.target_size = target_size  # Define target size for resizing

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
        image = Image.open(image_path).convert("RGB")  # Load as RGB

        # Create mask from polyline
        mask = Image.new("L", image.size)
        draw = ImageDraw.Draw(mask)
        for region in image_info["regions"]:
            points = list(zip(region["shape_attributes"]["all_points_x"],
                              region["shape_attributes"]["all_points_y"]))
            draw.polygon(points, outline=1, fill=1)

        # Resize image and mask to target size
        image = image.resize(self.target_size, Image.BILINEAR)
        mask = mask.resize(self.target_size, Image.NEAREST)

        # Apply transformations (if any)
        if self.transform:
            image = self.transform(image)

        mask = torch.tensor(np.array(mask), dtype=torch.float32).unsqueeze(0)  # Add channel dimension

        return image, mask


# Define transformations
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# Dataset paths
image_folder = "../data/masked/annotated"
json_path = "../data/masked/annotated/annotations.json"

# Create dataset and DataLoader
dataset = LungSegmentationDataset(image_folder, json_path, transform=train_transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)



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

        dec4 = self.dec4(torch.cat([self.up4(bottleneck), enc4], dim=1))
        dec3 = self.dec3(torch.cat([self.up3(dec4), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self.up2(dec3), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self.up1(dec2), enc1], dim=1))

        return torch.sigmoid(self.final(dec1))



# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet().to(device)
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 100
for epoch in range(epochs):
    model.train()
    epoch_loss = 0
    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, masks)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

# Save model
torch.save(model.state_dict(), "lung_segmentation_model.pth")

import matplotlib.pyplot as plt


def predict_mask(model, image_path, transform):
    model.eval()
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        mask_pred = model(image_tensor).squeeze(0).squeeze(0).cpu().numpy()

    return image, mask_pred


# Predict masks for a new folder
test_folder = "../data/masked/test"
output_folder = "../data/masked/test/masks"

os.makedirs(output_folder, exist_ok=True)

for image_file in os.listdir(test_folder):
    image_path = os.path.join(test_folder, image_file)
    image, mask_pred = predict_mask(model, image_path, train_transform)

    # Save or visualize the mask
    mask_pred = (mask_pred > 0.5).astype(np.uint8)  # Binarize the mask
    mask_image = Image.fromarray((mask_pred * 255).astype(np.uint8))
    mask_image.save(os.path.join(output_folder, image_file.replace(".jpg", "_mask.png")))

    # Optional: visualize
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original Image")
    plt.subplot(1, 2, 2)
    plt.imshow(mask_pred, cmap="gray")
    plt.title("Predicted Mask")
    plt.show()