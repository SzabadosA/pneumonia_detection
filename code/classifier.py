import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, Specificity
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import ImageFolder
from code.dataloader import PneumoniaDataset
from code.custom_checkpoint import CustomModelCheckpoint
from transformers import ViTForImageClassification, ViTImageProcessor
from code.project_globals import TEST_DIR, TRAIN_DIR, VAL_DIR
from torchvision import transforms
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import Counter
import numpy as np


def make_square(image, fill=0):
    """
    Pads the image to make it square, preserving the content.
    Args:
        image (PIL.Image): The input image.
        fill (int or tuple): The fill color for padding.
    Returns:
        PIL.Image: The padded square image.
    """
    w, h = image.size
    max_side = max(w, h)
    pad_left = (max_side - w) // 2
    pad_top = (max_side - h) // 2
    pad_right = max_side - w - pad_left
    pad_bottom = max_side - h - pad_top
    return transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)

class PneumoniaClassifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['backbone'])

        # Metrics
        self.accuracy = Accuracy(task='binary')
        self.precision = Precision(task='binary')
        self.recall = Recall(task='binary')
        self.f1 = F1Score(task='binary')
        self.specificity = Specificity(task='binary')

        # Add a tracker for unfrozen layers
        self.currently_unfrozen = 0  # Number of layers currently unfrozen

        # Create dataloaders and compute class weights
        self.train_loader, self.val_loader, self.test_loader, self.gradcam_loader = self.create_dataloaders()
        if config.use_class_weights:
            self.class_weights = self.compute_class_weights().to(self.device)
        else:
            self.class_weights = None

        # Callbacks and trainer
        self.checkpoint_callback = CustomModelCheckpoint(
            monitor='val_loss',
            dirpath='../checkpoints',
            filename=self.config.model_name,
            save_top_k=1,
            mode='min'
        )
        self.early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=self.config.patience,
            verbose=True,
            mode='min'
        )
        self.trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu",
            devices=1,
            logger=TensorBoardLogger("tb_logs", name=self.config.model_name),
            log_every_n_steps=10,
            callbacks=[self.checkpoint_callback, self.early_stopping_callback],
            precision='16-mixed',
            accumulate_grad_batches=1
        )

    def unfreeze_next_layers(self, num_layers_to_unfreeze=1):
        pass

    def get_vit_target_layer(self):
        pass



    def compute_class_weights(self):
        class_counts = Counter(self.train_loader.dataset.labels)
        total_samples = sum(class_counts.values())
        class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
        return torch.tensor(class_weights, dtype=torch.float32).to(self.device)

    def save_checkpoint(self, checkpoint_path):
        checkpoint = {
            "state_dict": self.state_dict(),
            "config": self.config,
            "metadata": {
                "epoch": current_epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
        }
        torch.save(checkpoint, checkpoint_path)

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        config = checkpoint['config']
        model = cls(config)
        model.load_state_dict(checkpoint['state_dict'])
        return model

    def identity_transform(self, x):
        return x

    def create_transforms(self):
        pass

    def create_dataloaders(self):
        train_transform, val_transform, test_transform = self.create_transforms()

        train_folder = None
        val_folder = None
        test_folder = None

        if self.config.image_type == 0:
            train_folder = "../data/raw/train"
            val_folder = "../data/raw/val"
            test_folder = "../data/raw/test"
        elif self.config.image_type == 1:
            train_folder = "../data/reordered/train"
            val_folder = "../data/reordered/val"
            test_folder = "../data/reordered/test"
        elif self.config.image_type == 2:
            train_folder = "../data/equalized/train"
            val_folder = "../data/equalized/val"
            test_folder = "../data/equalized/test"
        elif self.config.image_type == 3:
            train_folder = "../data/premultiplied/train"
            val_folder = "../data/premultiplied/val"
            test_folder = "../data/premultiplied/test"

        train_dataset = PneumoniaDataset(
            root_dir=train_folder,
            transform=train_transform,
            resolution=self.config.image_res
        )
        val_dataset = PneumoniaDataset(
            root_dir=val_folder,
            transform=val_transform,
            resolution=self.config.image_res
        )
        test_dataset = PneumoniaDataset(
            root_dir=test_folder,
            transform=test_transform,
            resolution=self.config.image_res
        )
        gradcam_dataset = PneumoniaDataset(
            root_dir=test_folder,
            transform=test_transform,
            resolution=self.config.image_res
        )

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                  num_workers=self.config.num_workers, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False,
                                num_workers=self.config.num_workers, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                 num_workers=self.config.num_workers, persistent_workers=True)
        gradcam_loader = DataLoader(gradcam_dataset, batch_size=self.config.batch_size, shuffle=False,
                                 num_workers=self.config.num_workers, persistent_workers=True)

        return train_loader, val_loader, test_loader, gradcam_loader

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        data, label, _ = batch
        print(
            f"Data device: {data.device}, Label device: {label.device}, Model device: {next(self.parameters()).device}, Class weights device: {self.class_weights.device if self.class_weights is not None else 'N/A'}"
        )
        data, label = data.to(self.device), label.to(self.device)  # Move input to the correct device

        # Forward pass
        output = self.forward(data)

        # Use the precomputed and device-moved class weights
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights) if self.class_weights is not None else nn.CrossEntropyLoss()
        loss = loss_fn(output, label)

        preds = torch.argmax(output, dim=1)

        # Update metrics
        self.accuracy.update(preds, label)
        self.precision.update(preds, label)
        self.recall.update(preds, label)
        self.f1.update(preds, label)
        self.specificity.update(preds, label)

        # Log metrics
        self.log('train_loss_step', loss, prog_bar=True, on_step=True)
        return loss

    def on_fit_start(self):
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)

    def validation_step(self, batch, batch_idx):
        val_data, val_label, _ = batch
        val_output = self.forward(val_data)
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)

        val_preds = torch.argmax(val_output, dim=1)
        self.accuracy.update(val_preds, val_label)
        self.precision.update(val_preds, val_label)
        self.recall.update(val_preds, val_label)
        self.f1.update(val_preds, val_label)
        self.specificity.update(val_preds, val_label)
        self.log('val_loss', val_loss)
        #self.log('val_acc_step', self.accuracy.compute(), prog_bar=True)
        #self.log('val_precision_step', self.precision.compute(), prog_bar=True)
        #self.log('val_recall_step', self.recall.compute(), prog_bar=True)
        #self.log('val_f1_step', self.f1.compute(), prog_bar=True)
        self.accuracy.compute()
        self.precision.compute()
        self.recall.compute()
        self.f1.compute()
        self.specificity.compute()

        return val_loss

    def on_train_epoch_end(self):
        # Unfreeze layers gradually every `unfreeze_interval` epochs
        if self.config.gradually_unfreeze:
            if (self.current_epoch + 1) % self.config.unfreeze_interval == 0:
                self.unfreeze_next_layers(num_layers_to_unfreeze=self.config.num_layers_to_unfreeze)
        else:
            pass


    def test_step(self, batch, batch_idx):
        test_data, test_label, _ = batch
        test_output = self.forward(test_data)
        test_loss = nn.CrossEntropyLoss()(test_output, test_label)

        test_preds = torch.argmax(test_output, dim=1)
        self.accuracy.update(test_preds, test_label)
        self.precision.update(test_preds, test_label)
        self.recall.update(test_preds, test_label)
        self.f1.update(test_preds, test_label)
        self.specificity.update(test_preds, test_label)
        #self.log('test_loss', test_loss)
        #self.log('test_acc_step', self.accuracy.compute(), prog_bar=True)
        #self.log('test_precision_step', self.precision.compute(), prog_bar=True)
        #self.log('test_recall_step', self.recall.compute(), prog_bar=True)
        #self.log('test_f1_step', self.f1.compute(), prog_bar=True)

        self.accuracy.compute()
        self.precision.compute()
        self.recall.compute()
        self.f1.compute()
        self.specificity.compute()
        return test_loss

    def on_test_epoch_end(self):
        acc = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        specificity = self.specificity.compute()

        self.log('test_acc_epoch', acc, prog_bar=True)
        self.log('test_precision_epoch', precision, prog_bar=True)
        self.log('test_recall_epoch', recall, prog_bar=True)
        self.log('test_f1_epoch', f1, prog_bar=True)
        self.log('test_specificity_epoch', specificity, prog_bar=True)

        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.specificity.reset()

    def on_validation_epoch_end(self):
        acc = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        specificity = self.specificity.compute()

        # Log validation epoch metrics
        self.log('val_acc_epoch', acc, prog_bar=True, on_epoch=True)
        self.log('val_precision_epoch', precision, prog_bar=True, on_epoch=True)
        self.log('val_recall_epoch', recall, prog_bar=True, on_epoch=True)
        self.log('val_f1_epoch', f1, prog_bar=True, on_epoch=True)
        self.log('val_specificity_epoch', specificity, prog_bar=True, on_epoch=True)

        # Reset metrics
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.specificity.reset()

    def configure_optimizers(self):
        if self.config.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate,
                                         weight_decay=self.config.weight_decay)
        elif self.config.optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.learning_rate,
                                        weight_decay=self.config.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")

        # Use different learning rates for frozen and unfrozen layers
        if self.config.gradually_unfreeze:
            optimizer = torch.optim.Adam([
                {'params': self.feature_extractor.parameters(), 'lr': self.config.frozen_lr},
                {'params': self.classifier.parameters(), 'lr': self.config.unfrozen_lr}
            ], weight_decay=self.config.weight_decay)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate,
                                         weight_decay=self.config.weight_decay)

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def train_model(self):
        self.trainer.fit(self, self.train_loader, self.val_loader)

    def test_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.trainer.test(self, self.test_loader)
        metadata = checkpoint.get("metadata", {})
        return metadata


######################################################
######################################################
class CNNPneumoniaClassifier(PneumoniaClassifier):
    def __init__(self, config):
        super().__init__(config)

        # Model backbone
        backbone = getattr(models, config.backbone_name)(weights='DEFAULT')
        if 'efficientnet' in config.backbone_name:
            num_filters = backbone.classifier[1].in_features
        elif 'densenet' in config.backbone_name:
            num_filters = backbone.classifier.in_features
        else:
            num_filters = backbone.fc.in_features

        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # Initially freeze all layers
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(num_filters, 2)

        if config.transfer_learning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def create_transforms(self):
        train_transform = transforms.Compose([
            transforms.Lambda(make_square),  # Pass the function itself, not the result of calling it
            transforms.Resize((self.config.image_res, self.config.image_res)),  # Resize to target resolution
            transforms.RandomRotation(3),  # Apply small random rotation
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        val_transform = transforms.Compose([
            transforms.Lambda(make_square),
            transforms.Resize((self.config.image_res, self.config.image_res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        test_transform = transforms.Compose([
            transforms.Lambda(make_square),
            transforms.Resize((self.config.image_res, self.config.image_res)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return train_transform, val_transform, test_transform

    def forward(self, x):
        features = self.feature_extractor(x).flatten(1)
        return self.classifier(features)

    def unfreeze_next_layers(self, num_layers_to_unfreeze=1):
        """
        Unfreeze the next set of layers in the feature extractor.
        Args:
            num_layers_to_unfreeze (int): Number of layers to unfreeze in each step.
        """
        # Handle non-ViT layers
        layers = list(self.feature_extractor.children())
        for i in range(
            self.currently_unfrozen,
            min(self.currently_unfrozen + num_layers_to_unfreeze, len(layers))
        ):
            for param in layers[i].parameters():
                param.requires_grad = True

        self.currently_unfrozen += num_layers_to_unfreeze
        print(f"Unfroze up to layer {self.currently_unfrozen}")

    def visualize_gradcam(self, num_samples, target_layer, class_names, threshold=0.5):
        """
        Visualize Grad-CAM with an adjustable threshold for highlighting hot regions.

        Args:
            num_samples (int): Number of samples to visualize.
            target_layer (int): Target layer index for Grad-CAM.
            class_names (list): List of class names for labels.
            threshold (float): Threshold for heatmap, values below this are set to 0. Default is 0.5.
        """
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import random
        import matplotlib.pyplot as plt
        import numpy as np

        # Ensure model is in evaluation mode
        self.eval()

        # Retrieve the target layer
        if "efficientnet" in self.config.backbone_name:
            target_layer = target_layer

        target_layer = self.feature_extractor[target_layer]

        def vit_reshape_transform(output):
            hidden_states = output[0]  # First element is the hidden states
            return hidden_states[:, 1:, :].mean(dim=1)  # Mean pooling (ignores CLS token)

        # Initialize GradCAM
        gradcam = GradCAM(
            model=self,
            target_layers=[target_layer],
            reshape_transform=None
        )

        # Sample random images from the test set
        dataset = self.gradcam_loader.dataset
        samples = random.sample(range(len(dataset)), num_samples)

        for idx in samples:
            inputs, label, img_name = dataset[idx]  # Adjust based on your dataset structure
            inputs = inputs.unsqueeze(0).to(self.device)  # Add batch dimension
            label = int(label)

            # Generate Grad-CAM heatmap
            grayscale_cam = gradcam(input_tensor=inputs)[0]

            # Apply thresholding
            grayscale_cam[grayscale_cam < threshold] = 0  # Set values below the threshold to 0

            # Normalize and prepare the input image
            input_image = inputs[0].permute(1, 2, 0).cpu().numpy()
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
            input_image = input_image.astype(np.float32)

            # Generate the heatmap
            heatmap = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

            # Display the heatmap
            plt.imshow(heatmap)
            plt.title(f"Class: {class_names[label]} - {img_name} - (Threshold: {threshold})")
            plt.axis("off")
            plt.show()


#######################################################
#######################################################
class ViTPneumoniaClassifier(PneumoniaClassifier):
    def __init__(self, config):
        super().__init__(config)

        self.processor = ViTImageProcessor.from_pretrained(config.backbone_name)
        self.feature_extractor = ViTForImageClassification.from_pretrained(
            config.backbone_name, num_labels=2, ignore_mismatched_sizes=True
        )
        # Replace classifier with custom one
        self.feature_extractor.classifier = nn.Identity()  # Remove the default classifier
        self.classifier = nn.Linear(self.feature_extractor.config.hidden_size, 2)
        self.vit_layers = list(self.feature_extractor.vit.encoder.layer)  # Store ViT encoder layers

        if config.transfer_learning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def create_transforms(self):
        train_transform = transforms.Compose([
            transforms.Lambda(make_square),  # Pass the function itself, not the result of calling it
            transforms.Resize((self.config.image_res, self.config.image_res)),  # Resize to target resolution
            transforms.RandomRotation(3),  # Apply small random rotation
            transforms.ToTensor(),
        ])
        val_transform = transforms.Compose([
            transforms.Lambda(make_square),
            transforms.Resize((self.config.image_res, self.config.image_res)),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.Lambda(make_square),
            transforms.Resize((self.config.image_res, self.config.image_res)),
            transforms.ToTensor(),
        ])
        return train_transform, val_transform, test_transform

    def forward(self, x):
        x = self.processor(images=x, return_tensors="pt", do_rescale=False)["pixel_values"].to(self.device)
        features = self.feature_extractor.vit(pixel_values=x).last_hidden_state[:, 0, :]
        return self.classifier(features)

    def unfreeze_next_layers(self, num_layers_to_unfreeze=1):
        """
        Unfreeze the next set of layers in the feature extractor.
        Args:
            num_layers_to_unfreeze (int): Number of layers to unfreeze in each step.
        """
        # Handle ViT layers specifically
        for i in range(
            self.currently_unfrozen,
            min(self.currently_unfrozen + num_layers_to_unfreeze, len(self.vit_layers))
        ):
            for param in self.vit_layers[i].parameters():
                param.requires_grad = True
        self.currently_unfrozen += num_layers_to_unfreeze
        print(f"Unfroze up to layer {self.currently_unfrozen}")

    def get_vit_target_layer(self, layer_index=None):
        """
        Retrieve the target layer for Grad-CAM in ViT.
        Args:
            layer_index (int): Index of the transformer encoder layer to use.
                               If None, defaults to the last encoder layer.
        Returns:
            nn.Module: The target layer for Grad-CAM.
        """
        if "vit" not in self.config.backbone_name:
            raise ValueError("This method is only applicable to ViT backbones.")

        # Default to the last encoder layer
        if layer_index is None:
            layer_index = len(self.vit_layers) - 1
        return self.vit_layers[layer_index]

    def visualize_gradcam(self, num_samples, target_layer, class_names, threshold=0.5):
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import random
        import matplotlib.pyplot as plt
        import numpy as np

        self.eval()

        # Ensure the target layer is correctly retrieved
        target_layer = self.get_vit_target_layer(target_layer)

        def vit_reshape_transform(output):
            hidden_states = output[0]
            return hidden_states[:, 1:, :].mean(dim=1)  # Mean pooling (ignores CLS token)

        gradcam = GradCAM(
            model=self,
            target_layers=[target_layer],
            reshape_transform=vit_reshape_transform
        )

        # Randomly sample from the Grad-CAM loader
        dataset = self.gradcam_loader.dataset
        samples = random.sample(range(len(dataset)), num_samples)

        for idx in samples:
            inputs, label, img_name = dataset[idx]
            inputs = inputs.unsqueeze(0).to(self.device)  # Add batch dimension

            # Generate Grad-CAM heatmap
            grayscale_cam = gradcam(input_tensor=inputs)[0]
            if grayscale_cam is None:
                print(f"Grad-CAM returned None for {img_name}")
                continue

            # Thresholding
            grayscale_cam[grayscale_cam < threshold] = 0

            # Normalize and prepare the input image
            input_image = inputs[0].permute(1, 2, 0).cpu().numpy()
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
            input_image = input_image.astype(np.float32)

            # Generate and show the heatmap
            heatmap = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)
            plt.imshow(heatmap)
            plt.title(f"Class: {class_names[label]} - {img_name}")
            plt.axis("off")
            plt.show()


class Config:
    def __init__(
            self,
            backbone_name,
            transfer_learning,
            learning_rate,
            batch_size,
            max_epochs,
            weight_decay,
            dropout,
            num_workers,
            model_name,
            version,
            optimizer_name,
            use_class_weights,
            image_res,
            patience,
            image_type,
            frozen_lr=None,
            unfrozen_lr=None,
            unfreeze_interval=None,
            num_layers_to_unfreeze=None,
            gradually_unfreeze=False
        ):
        self.backbone_name = backbone_name
        self.transfer_learning = transfer_learning
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.num_workers = num_workers
        self.model_name = model_name
        self.version = version
        self.optimizer_name = optimizer_name
        self.use_class_weights = use_class_weights
        self.image_res = image_res
        self.patience = patience
        self.image_type = image_type
        self.frozen_lr = frozen_lr
        self.unfrozen_lr = unfrozen_lr
        self.unfreeze_interval = unfreeze_interval
        self.num_layers_to_unfreeze = num_layers_to_unfreeze
        self.gradually_unfreeze = gradually_unfreeze