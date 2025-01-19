import os
import sys
import torch
import gradio as gr
import numpy as np
import cv2
from torchvision import transforms
from PIL import Image
from transformers import ViTImageProcessor

sys.path.append(os.path.abspath("./code"))
from classifier import PneumoniaClassifier, Config
from classifier import CNNPneumoniaClassifier, ViTPneumoniaClassifier
from pytorch_grad_cam import GradCAM

# Load available models from the saved directory
DEFAULT_MODEL="ResNet50_gradual_unfreeze_final.pt"
MODEL_DIR = "models"
AVAILABLE_MODELS = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pt")]
if DEFAULT_MODEL in AVAILABLE_MODELS:
    AVAILABLE_MODELS.remove(DEFAULT_MODEL)
    AVAILABLE_MODELS.insert(0, DEFAULT_MODEL)


class GradCamViT:
    def __init__(self, model, target_layer):
        print("[GradCamViT] Initializing GradCamViT")
        self.model = model.eval()
        self.target_layer = target_layer
        self.feature = None
        self.gradient = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                output = output[0]
            self.feature = output.clone()

        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple):
                grad_output = grad_output[0]
            self.gradient = grad_output.clone()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate_cam(self, image_tensor, class_idx):
        image_tensor.requires_grad = True
        for param in self.model.parameters():
            param.requires_grad = True

        output = self.model(image_tensor)
        output.retain_grad()
        self.model.zero_grad()
        output[:, class_idx].backward(retain_graph=True)

        if self.gradient is None:
            raise RuntimeError("Gradients were not computed. Ensure the model and inputs are correct.")

        batch_size, seq_len, embed_dim = self.feature.shape  # Extract shape
        seq_len -= 1  # Exclude CLS token
        height = width = int(seq_len ** 0.5)  # Compute square spatial shape

        self.feature = self.feature[:, 1:, :].reshape(batch_size, height, width, embed_dim).permute(0, 3, 1, 2)
        self.gradient = self.gradient[:, 1:, :].reshape(batch_size, height, width, embed_dim).permute(0, 3, 1, 2)

        weights = torch.mean(self.gradient, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * self.feature, dim=1).squeeze().detach().cpu().numpy()
        cam = np.maximum(cam, 0)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min == 0:
            cam = np.zeros_like(cam)
        else:
            cam = (cam - cam_min) / (cam_max - cam_min)

        cam = cv2.resize(cam, (image_tensor.shape[-1], image_tensor.shape[-2]))
        return cam


def apply_gradcam(model, image_tensor, is_vit=False):
    #model.eval()  # Ensure model is in evaluation mode

    if is_vit:
        target_layer = model.feature_extractor.vit.encoder.layer[
            -2].layernorm_after
        grad_cam = GradCamViT(model, target_layer)
        class_idx = model(image_tensor).argmax(dim=1).item()
        print(f"[apply_gradcam] Predicted class: {class_idx}")
        grayscale_cam = grad_cam.generate_cam(image_tensor, class_idx)
    else:
        target_layer = None
        for module in model.modules():
            if isinstance(module, torch.nn.Conv2d):
                target_layer = module
        cam = GradCAM(model=model, target_layers=[target_layer])
        grayscale_cam = cam(input_tensor=image_tensor)[0]

    print(f"[apply_gradcam] Grad-CAM min: {grayscale_cam.min()}, max: {grayscale_cam.max()}")

    grayscale_cam = (grayscale_cam - grayscale_cam.min()) / (grayscale_cam.max() - grayscale_cam.min())
    grayscale_cam = np.uint8(255 * grayscale_cam)

    heatmap = cv2.applyColorMap(grayscale_cam, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    image_np = image_tensor.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    image_np = (image_np - image_np.min()) / (image_np.max() - image_np.min())
    image_np = np.uint8(255 * image_np)

    overlay = cv2.addWeighted(image_np, 0.6, heatmap, 0.4, 0)
    return Image.fromarray(overlay)

def preprocess_image(image, config):
    """Preprocess input image based on the model type (ViT or CNN)."""
    image = image.convert("RGB")
    image = transforms.Resize((config.image_res, config.image_res))(image)

    if "vit" in config.backbone_name.lower():
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor / image_tensor.max()
    else:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image).unsqueeze(0)

    return image_tensor




def predict(model_name, image):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(f"./models/{model_name}", map_location=device)
    loaded_config = Config(**checkpoint["config"])

    if "Vit" in model_name:
        model = ViTPneumoniaClassifier(loaded_config)
        is_vit = True
    else:
        model = CNNPneumoniaClassifier(loaded_config)
        is_vit = False

    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()  # Ensure model is in training mode

    image_tensor = preprocess_image(image, loaded_config).to(device)
    image_tensor.requires_grad = True  # Ensure gradients are tracked

    with torch.no_grad():
        output = model(image_tensor)

    probabilities = torch.softmax(output, dim=1)
    pred_idx = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0, pred_idx].item() * 100
    classes = ["NORMAL", "PNEUMONIA"]
    pred_label = classes[pred_idx]

    heatmap = apply_gradcam(model, image_tensor, is_vit=is_vit)
    return f"{pred_label} -- Confidence: {confidence:.2f}%", heatmap


gui = gr.Interface(
    fn=predict,
    inputs=[
        gr.Dropdown(choices=AVAILABLE_MODELS, label="Select Model"),
        gr.Image(type="pil", label="Upload Chest X-ray")
    ],
    outputs=[
        gr.Label(label="Prediction with Confidence"),
        gr.Image(label="Grad-CAM Heatmap")
    ],
    title="Pneumonia Detection with AI",
    description="Upload a chest X-ray and select a model to classify the image. Grad-CAM will visualize the areas influencing the decision."
)


def main():
    gui.launch()


if __name__ == "__main__":
    main()
