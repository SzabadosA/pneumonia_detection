import os
import sys
from transformers import ViTImageProcessor
sys.path.append(os.path.abspath("./code"))
import torch
import gradio as gr
import numpy as np
from torchvision import transforms
from PIL import Image
from classifier import PneumoniaClassifier, Config
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from classifier import CNNPneumoniaClassifier, ViTPneumoniaClassifier


# Load available models from the saved directory
MODEL_DIR = "models"
available_models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]  # Use .pth instead of .pt


import torch
import timm
import numpy as np
import cv2
from skimage import io


import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations

class GradCam:
    def __init__(self, model, target):
        self.model = model.eval()  # Set the model to evaluation mode
        self.feature = None  # To store the features from the target layer
        self.gradient = None  # To store the gradients from the target layer
        self.handlers = []  # List to keep track of hooks
        self.target = target  # Target layer for Grad-CAM
        self._get_hook()  # Register hooks to the target layer

    # Hook to get features from the forward pass
    def _get_features_hook(self, module, input, output):
        self.feature = self.reshape_transform(output)  # Store and reshape the output features

    # Hook to get gradients from the backward pass
    def _get_grads_hook(self, module, input_grad, output_grad):
        if output_grad.requires_grad:
            output_grad.register_hook(self._store_grad)  # Register hook to store gradients

    def _store_grad(self, grad):
        self.gradient = self.reshape_transform(grad)  # Store gradients for later use

    # Register forward hooks to the target layer
    def _get_hook(self):
        self.target.register_forward_hook(self._get_features_hook)
        self.target.register_forward_hook(self._get_grads_hook)

    # Function to reshape the tensor for visualization
    def reshape_transform(self, tensor, height=14, width=14):
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)  # Rearrange dimensions to (C, H, W)
        return result

    # Function to compute the Grad-CAM heatmap
    def __call__(self, inputs):
        self.model.zero_grad()  # Zero the gradients
        output = self.model(inputs)  # Forward pass

        # Get the index of the highest score in the output
        index = np.argmax(output.cpu().data.numpy())
        target = output[0][index]  # Get the target score
        target.backward()  # Backward pass to compute gradients

        # Get the gradients and features
        gradient = self.gradient[0].cpu().data.numpy()
        weight = np.mean(gradient, axis=(1, 2))  # Average the gradients
        feature = self.feature[0].cpu().data.numpy()

        # Compute the weighted sum of the features
        cam = feature * weight[:, np.newaxis, np.newaxis]
        cam = np.sum(cam, axis=0)  # Sum over the channels
        cam = np.maximum(cam, 0)  # Apply ReLU to remove negative values

        # Normalize the heatmap
        cam -= np.min(cam)
        cam /= np.max(cam)
        cam = cv2.resize(cam, (224, 224))  # Resize to match the input image size
        return cam  # Return the Grad-CAM heatmap

def apply_gradcam_CNN(model, image_tensor):
    """Apply Grad-CAM and plot the result."""
    model.eval()

    # Ensure gradients are enabled
    for param in model.parameters():
        param.requires_grad = True

    # Select the last convolutional layer dynamically
    target_layer = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module  # Keep overwriting to get the last Conv2D layer

    if target_layer is None:
        raise ValueError("No convolutional layers found in the model!")

    # Ensure the input tensor allows gradient computation
    image_tensor.requires_grad = True

    # Initialize Grad-CAM
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Check if gradients are being recorded
    for name, param in model.named_parameters():
        if param.grad is None:
            print(
                f"WARNING: No gradients found for {name}. Ensure the model is in train mode and doing a backward pass.")

    # Compute Grad-CAM heatmap
    grayscale_cam = cam(input_tensor=image_tensor, eigen_smooth=False)[0]

    # Normalize and overlay on image
    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    heatmap = show_cam_on_image(image_np.astype(np.float32), grayscale_cam, use_rgb=True)

    # Plot result
def prepare_input(image):
    img = np.array(image)
    img = cv2.resize(img, (224, 224))  # Resize to 224x224

    # Ensure 3-channel RGB format
    if len(img.shape) == 2:  # If grayscale (shape: H, W)
        img = np.stack([img] * 3, axis=-1)  # Convert to RGB (H, W, 3)

    img = np.float32(img) / 255  # Normalize to [0,1]


    #means = np.array([0.5, 0.5, 0.5])
    #stds = np.array([0.5, 0.5, 0.5])
    #img -= means
    #img /= stds

    img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))  # Convert to (C, H, W)
    img = img[np.newaxis, ...]  # Add batch dimension
    return torch.tensor(img, dtype=torch.float32, requires_grad=True)

def gen_cam(image, mask):
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255  # Normalize heatmap

    # Ensure image has 3 channels
    if len(image.shape) == 2:  # If grayscale (H, W)
        image = np.stack([image] * 3, axis=-1)  # Convert to (H, W, 3)

    cam = (1 - 0.5) * heatmap + 0.5 * image
    cam = cam / np.max(cam)  # Normalize
    return np.uint8(255 * cam)  # Convert to 8-bit image


################

def preprocess_image(image, image_size=224):
    """Load and preprocess an image."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    image = image.convert("RGB")
    return transform(image).unsqueeze(0)  # Add batch dimension

# Prediction function
def predict(model_name, image):
    print(image)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the saved model checkpoint
    checkpoint = torch.load(f"./models/{model_name}", map_location=device)

    # Reconstruct Config object from the saved dictionary
    loaded_config = Config(**checkpoint["config"])  # Convert dict back to Config

    if "Vit" in model_name:
        model = ViTPneumoniaClassifier(loaded_config)
        vit=True
    else:
        model = CNNPneumoniaClassifier(loaded_config)
        vit=False

        # Load model weights
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    # Preprocess the input image

    # Run inference
    if not vit:
        image_tensor = preprocess_image(image, loaded_config.image_res).to(device)
        with torch.no_grad():
            output = model(image_tensor)
        probabilities = torch.softmax(output, dim=1)
        pred_idx = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0, pred_idx].item() * 100  # Convert to percentage
        # Get prediction
        classes = ["NORMAL", "PNEUMONIA"]
        pred_label = classes[torch.argmax(output, dim=1).item()]
        print(f"Prediction: {pred_label}, Confidence: {confidence:.2f}%")



    # Generate Grad-CAM visualization
    heatmap = None
    if vit:
        print(model)
        target_layer = model.feature_extractor.vit.encoder.layer[-1].output
        inputs = prepare_input(image)
        grad_cam = GradCam(model, target_layer)
        mask = grad_cam(inputs)  # Compute Grad-CAM mask

        # Load original image for overlay
        img = io.imread(image)
        img = np.float32(cv2.resize(img, (loaded_config.image_res, loaded_config.image_res))) / 255

        heatmap = gen_cam(img, mask)  # Generate heatmap

    else:
        # Ensure gradients are enabled
        for param in model.parameters():
            param.requires_grad = True

        # Select the last convolutional layer dynamically
        target_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module  # Keep overwriting to get the last Conv2D layer

        if target_layer is None:
            raise ValueError("No convolutional layers found in the model!")

        # Ensure the input tensor allows gradient computation
        image_tensor.requires_grad = True

        # Initialize Grad-CAM
        cam = GradCAM(model=model, target_layers=[target_layer])

        # Check if gradients are being recorded
        for name, param in model.named_parameters():
            if param.grad is None:
                print(
                    f"WARNING: No gradients found for {name}. Ensure the model is in train mode and doing a backward pass.")

        # Compute Grad-CAM heatmap
        grayscale_cam = cam(input_tensor=image_tensor, eigen_smooth=False)[0]

        # Normalize and overlay on image
        image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
        heatmap = show_cam_on_image(image_np.astype(np.float32), grayscale_cam, use_rgb=True)


    # Convert Grad-CAM image to PIL format for Gradio output
    heatmap_pil = heatmap

    return f"{pred_label} -- Confidence: {confidence:.2f}%", heatmap_pil


# Gradio Interface
gui = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=available_models, label="Select Model"),
        gr.Image(type="pil", label="Upload Chest X-ray")
    ],
    outputs=[
        gr.Label(label="Prediction with Confidence"),
        gr.Image(label="Grad-CAM Heatmap", width=600, height=600, container=True)
    ],
    title="Pneumonia Detection with AI",
    description="Upload a chest X-ray and select a model to classify the image. Grad-CAM will visualize the areas influencing the decision."
)


def main():
    gui.launch()
    #predict("Vit_gradual_unfreeze_upscale_final.pt", Image.open("data/reordered/test/PNEUMONIA/person19_bacteria_61.jpeg"))


if __name__ == "__main__":
    main()
