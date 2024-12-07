import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score, Specificity
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from code.dataloader import PneumoniaDataset
from code.custom_checkpoint import CustomModelCheckpoint
from transformers import ViTForImageClassification, ViTImageProcessor
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from collections import Counter



def make_square(image, fill=0):
    """
    Pads the image to make it square, preserving the original content.
    Args:
        image (PIL.Image): The input image to be processed.
        fill (int or tuple): The fill color for padding (default is black or 0).
    Returns:
        PIL.Image: The padded square image.
    """
    w, h = image.size  # Extract the width and height of the image
    max_side = max(w, h)  # Determine the largest side to calculate padding
    pad_left = (max_side - w) // 2  # Calculate padding for the left side
    pad_top = (max_side - h) // 2  # Calculate padding for the top
    pad_right = max_side - w - pad_left  # Calculate padding for the right side
    pad_bottom = max_side - h - pad_top  # Calculate padding for the bottom
    # Add padding to make the image square
    return transforms.functional.pad(image, (pad_left, pad_top, pad_right, pad_bottom), fill=fill)


class PneumoniaClassifier(pl.LightningModule):

    def __init__(self, config):
        super().__init__()  # Initialize the base LightningModule
        self.config = config  # Configuration object containing hyperparameters
        self.save_hyperparameters(ignore=['backbone'])  # Save hyperparameters for reproducibility

        # Define metrics for model evaluation
        self.accuracy = Accuracy(task='binary')  # Binary accuracy
        self.precision = Precision(task='binary')  # Precision metric
        self.recall = Recall(task='binary')  # Recall metric
        self.f1 = F1Score(task='binary')  # F1 score metric
        self.specificity = Specificity(task='binary')  # Specificity metric for imbalanced datasets

        # Tracker for transfer learning (gradual unfreezing of layers)
        self.currently_unfrozen = 0  # Tracks how many layers are currently unfrozen

        # Create data loaders and compute class weights if necessary
        self.train_loader, self.val_loader, self.test_loader, self.gradcam_loader = self.create_dataloaders()
        if config.use_class_weights:  # Check if class weighting is enabled
            self.class_weights = self.compute_class_weights().to(self.device)  # Compute and move to the appropriate device
        else:
            self.class_weights = None  # No class weights if not specified

        # Define callbacks for checkpointing and early stopping
        self.checkpoint_callback = CustomModelCheckpoint(
            monitor='val_loss',  # Monitor validation loss
            dirpath='../checkpoints',  # Directory to save checkpoints
            filename=self.config.model_name,  # Model filename
            save_top_k=1,  # Save the best model
            mode='min'  # Minimize validation loss
        )
        self.early_stopping_callback = EarlyStopping(
            monitor='val_loss',  # Monitor validation loss
            patience=self.config.patience,  # Number of epochs to wait before stopping
            verbose=True,  # Enable verbose logging
            mode='min'  # Stop when validation loss stops decreasing
        )

        # Initialize the Lightning Trainer
        self.trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,  # Maximum number of epochs
            accelerator="gpu",  # Use GPU if available
            devices=1,  # Number of devices (GPUs) to use
            logger=TensorBoardLogger("tb_logs", name=self.config.model_name),  # Log metrics to TensorBoard
            log_every_n_steps=10,  # Log every 10 steps
            callbacks=[self.checkpoint_callback, self.early_stopping_callback],  # Add callbacks
            precision='16-mixed',  # Use mixed precision for faster training
            accumulate_grad_batches=1  # Accumulate gradients across batches
        )

    def unfreeze_next_layers(self, num_layers_to_unfreeze=1):
        pass

    def get_vit_target_layer(self):
        pass

    def compute_class_weights(self):
        """
        Computes class weights to handle imbalanced datasets.
        Returns:
            torch.Tensor: Tensor of class weights.
        """
        class_counts = Counter(self.train_loader.dataset.labels)  # Count occurrences of each class
        total_samples = sum(class_counts.values())  # Total number of samples
        # Compute weights: total samples divided by the number of samples for each class
        class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]
        return torch.tensor(class_weights, dtype=torch.float32).to(self.device)  # Convert to tensor

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
        """
        Creates data loaders for training, validation, and testing.
        Returns:
            tuple: DataLoader objects for train, validation, test, and Grad-CAM datasets.
        """
        train_transform, val_transform, test_transform = self.create_transforms()  # Get preprocessing transforms

        # Define dataset paths based on image type
        if self.config.image_type == 0:
            train_folder, val_folder, test_folder = "../data/raw/train", "../data/raw/val", "../data/raw/test"
        elif self.config.image_type == 1:
            train_folder, val_folder, test_folder = "../data/reordered/train", "../data/reordered/val", "../data/reordered/test"
        elif self.config.image_type == 2:
            train_folder, val_folder, test_folder = "../data/equalized/train", "../data/equalized/val", "../data/equalized/test"
        elif self.config.image_type == 3:
            train_folder, val_folder, test_folder = "../data/premultiplied/train", "../data/premultiplied/val", "../data/premultiplied/test"

        # Create datasets
        train_dataset = PneumoniaDataset(root_dir=train_folder, transform=train_transform,
                                         resolution=self.config.image_res)
        val_dataset = PneumoniaDataset(root_dir=val_folder, transform=val_transform, resolution=self.config.image_res)
        test_dataset = PneumoniaDataset(root_dir=test_folder, transform=test_transform,
                                        resolution=self.config.image_res)
        gradcam_dataset = PneumoniaDataset(root_dir=test_folder, transform=test_transform,
                                           resolution=self.config.image_res)

        # Wrap datasets with DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True,
                                  num_workers=self.config.num_workers, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False,
                                num_workers=self.config.num_workers, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False,
                                 num_workers=self.config.num_workers, persistent_workers=True)
        gradcam_loader = DataLoader(gradcam_dataset, batch_size=self.config.batch_size, shuffle=False,
                                    num_workers=self.config.num_workers, persistent_workers=True)

        return train_loader, val_loader, test_loader, gradcam_loader  # Return all loaders

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        """
        Executes a single training step for the model.
        Args:
            batch (tuple): A batch of data containing inputs, labels, and optional metadata.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value for the current training step.
        """
        data, label, _ = batch  # Unpack the batch into input data, labels, and metadata.
        print(
            f"Data device: {data.device}, Label device: {label.device}, Model device: {next(self.parameters()).device}, Class weights device: {self.class_weights.device if self.class_weights is not None else 'N/A'}"
        )
        data, label = data.to(self.device), label.to(
            self.device)  # Move data and labels to the correct device (e.g., GPU).

        # Forward pass: Run the input data through the model to get predictions.
        output = self.forward(data)

        # Compute the loss using a weighted or unweighted CrossEntropyLoss, depending on class weights.
        loss_fn = nn.CrossEntropyLoss(
            weight=self.class_weights) if self.class_weights is not None else nn.CrossEntropyLoss()
        loss = loss_fn(output, label)  # Calculate the loss between predictions and true labels.

        preds = torch.argmax(output, dim=1)  # Get predicted class indices by selecting the max logit for each sample.

        # Update metrics based on predictions and true labels.
        self.accuracy.update(preds, label)
        self.precision.update(preds, label)
        self.recall.update(preds, label)
        self.f1.update(preds, label)
        self.specificity.update(preds, label)

        # Log the loss for the current step. `prog_bar=True` shows it in the progress bar.
        self.log('train_loss_step', loss, prog_bar=True, on_step=True)
        return loss  # Return the computed loss for backpropagation.

    def on_fit_start(self):
        """
        Prepares the class weights and moves them to the correct device at the start of training.
        """
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(
                self.device)  # Move class weights to the appropriate device (e.g., GPU).

    def validation_step(self, batch, batch_idx):
        """
        Executes a single validation step for the model.
        Args:
            batch (tuple): A batch of validation data containing inputs, labels, and optional metadata.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Validation loss for the current step.
        """
        val_data, val_label, _ = batch  # Unpack the batch into input data, labels, and metadata.

        # Forward pass: Run the validation data through the model.
        val_output = self.forward(val_data)

        # Compute validation loss using CrossEntropyLoss.
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)

        val_preds = torch.argmax(val_output, dim=1)  # Get predicted class indices for validation data.

        # Update metrics for validation.
        self.accuracy.update(val_preds, val_label)
        self.precision.update(val_preds, val_label)
        self.recall.update(val_preds, val_label)
        self.f1.update(val_preds, val_label)
        self.specificity.update(val_preds, val_label)

        # Log the validation loss for the current step.
        self.log('val_loss', val_loss)

        # Compute intermediate values for metrics (optional).
        self.accuracy.compute()
        self.precision.compute()
        self.recall.compute()
        self.f1.compute()
        self.specificity.compute()

        return val_loss  # Return the validation loss.

    def on_train_epoch_end(self):
        """
        Handles actions at the end of each training epoch.
        Gradually unfreezes layers if `gradually_unfreeze` is enabled in the configuration.
        """
        if self.config.gradually_unfreeze:  # Check if gradual unfreezing is enabled.
            # Unfreeze layers every `unfreeze_interval` epochs.
            if (self.current_epoch + 1) % self.config.unfreeze_interval == 0:
                self.unfreeze_next_layers(num_layers_to_unfreeze=self.config.num_layers_to_unfreeze)
        else:
            pass

    def test_step(self, batch, batch_idx):
        """
        Executes a single test step for the model.
        Args:
            batch (tuple): A batch of test data containing inputs, labels, and optional metadata.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Test loss for the current step.
        """
        test_data, test_label, _ = batch  # Unpack the batch into input data, labels, and metadata.

        # Forward pass: Run the test data through the model.
        test_output = self.forward(test_data)

        # Compute the test loss using CrossEntropyLoss.
        test_loss = nn.CrossEntropyLoss()(test_output, test_label)

        test_preds = torch.argmax(test_output, dim=1)  # Get predicted class indices for the test data.

        # Update metrics for testing.
        self.accuracy.update(test_preds, test_label)
        self.precision.update(test_preds, test_label)
        self.recall.update(test_preds, test_label)
        self.f1.update(test_preds, test_label)
        self.specificity.update(test_preds, test_label)

        # Compute intermediate values for metrics (optional).
        self.accuracy.compute()
        self.precision.compute()
        self.recall.compute()
        self.f1.compute()
        self.specificity.compute()

        return test_loss  # Return the test loss.

    def on_test_epoch_end(self):
        """
        Logs the final metrics at the end of the test epoch and resets all metric states.
        """
        # Compute the final values for each metric.
        acc = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        specificity = self.specificity.compute()

        # Log the metrics for the entire test epoch.
        self.log('test_acc_epoch', acc, prog_bar=True)
        self.log('test_precision_epoch', precision, prog_bar=True)
        self.log('test_recall_epoch', recall, prog_bar=True)
        self.log('test_f1_epoch', f1, prog_bar=True)
        self.log('test_specificity_epoch', specificity, prog_bar=True)

        # Reset all metrics to clear their states for the next epoch.
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.specificity.reset()

    def on_validation_epoch_end(self):
        """
        Logs the final metrics at the end of the validation epoch and resets all metric states.
        """
        # Compute the final values for each validation metric.
        acc = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        specificity = self.specificity.compute()

        # Log the metrics for the entire validation epoch.
        self.log('val_acc_epoch', acc, prog_bar=True, on_epoch=True)
        self.log('val_precision_epoch', precision, prog_bar=True, on_epoch=True)
        self.log('val_recall_epoch', recall, prog_bar=True, on_epoch=True)
        self.log('val_f1_epoch', f1, prog_bar=True, on_epoch=True)
        self.log('val_specificity_epoch', specificity, prog_bar=True, on_epoch=True)

        # Reset all metrics to clear their states for the next epoch.
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()
        self.specificity.reset()

    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler for training.
        Returns:
            dict: A dictionary containing the optimizer and learning rate scheduler.
        """
        # Initialize optimizer based on the configuration.
        if self.config.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate,
                                         weight_decay=self.config.weight_decay)
        elif self.config.optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.learning_rate,
                                        weight_decay=self.config.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")

        # If gradual unfreezing is enabled, assign different learning rates for frozen and unfrozen layers.
        if self.config.gradually_unfreeze:
            optimizer = torch.optim.Adam([
                {'params': self.feature_extractor.parameters(), 'lr': self.config.frozen_lr},
                {'params': self.classifier.parameters(), 'lr': self.config.unfrozen_lr}
            ], weight_decay=self.config.weight_decay)

        # Define a learning rate scheduler to reduce the learning rate based on validation loss.
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
        """
        Initializes the CNN-based PneumoniaClassifier with the specified configuration.
        Args:
            config (object): Configuration object containing model and training parameters.
        """
        super().__init__(config)

        # Initialize the backbone model from torchvision using the specified architecture name in the config.
        backbone = getattr(models, config.backbone_name)(weights='DEFAULT')

        # Determine the number of output filters in the backbone's last layer based on its architecture type.
        if 'efficientnet' in config.backbone_name:
            num_filters = backbone.classifier[1].in_features
        elif 'densenet' in config.backbone_name:
            num_filters = backbone.classifier.in_features
        else:
            num_filters = backbone.fc.in_features

        # Extract all layers from the backbone except the final classification head.
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)  # Define feature extractor as a sequential model.

        # Freeze all layers initially to prevent them from being trained unless explicitly unfrozen.
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        # Define a dropout layer for regularization.
        self.dropout = nn.Dropout(p=config.dropout)

        # Define the final classification layer with two output classes (e.g., Normal and Pneumonia).
        self.classifier = nn.Linear(num_filters, 2)

        # Special case for DenseNet: Adjust the input size of the classifier to match DenseNet's flattened feature map.
        if 'dense' in config.backbone_name:
            self.classifier = nn.Linear(50176, 2)

        # If transfer learning is enabled, ensure that the feature extractor remains frozen.
        if config.transfer_learning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def create_transforms(self):
        """
        Creates data augmentation and preprocessing transforms for training, validation, and testing datasets.
        Returns:
            tuple: A tuple containing transforms for training, validation, and testing.
        """
        train_transform = transforms.Compose([
            transforms.Lambda(make_square),  # Pad the image to make it square.
            transforms.Resize((self.config.image_res, self.config.image_res)),  # Resize to target resolution.
            transforms.RandomRotation(3),  # Apply small random rotations for augmentation.
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize the image using ImageNet stats.
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
        """
        Defines the forward pass of the model.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Logits from the classifier.
        """
        # Extract features from the input image and flatten the feature map.
        features = self.feature_extractor(x).flatten(1)

        # Pass the features through the classifier to get logits.
        return self.classifier(features)

    def unfreeze_next_layers(self, num_layers_to_unfreeze=1):
        """
        Unfreezes the next set of layers in the feature extractor for training.
        Args:
            num_layers_to_unfreeze (int): Number of layers to unfreeze in each step.
        """
        # Get the list of layers in the feature extractor.
        layers = list(self.feature_extractor.children())

        # Unfreeze the next `num_layers_to_unfreeze` layers in the list.
        for i in range(
            self.currently_unfrozen,
            min(self.currently_unfrozen + num_layers_to_unfreeze, len(layers))
        ):
            for param in layers[i].parameters():
                param.requires_grad = True

        # Update the counter to track how many layers have been unfrozen.
        self.currently_unfrozen += num_layers_to_unfreeze
        print(f"Unfroze up to layer {self.currently_unfrozen}")

    def visualize_gradcam(self, num_samples, target_layer, class_names, threshold=0.5):
        """
        Visualizes Grad-CAM for interpreting model predictions.
        Args:
            num_samples (int): Number of random samples to visualize.
            target_layer (int): Target layer index for Grad-CAM computation.
            class_names (list): List of class names corresponding to the output labels.
            threshold (float): Threshold for Grad-CAM heatmap values (default: 0.5).
        """
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import random
        import matplotlib.pyplot as plt
        import numpy as np

        # Ensure the model is in evaluation mode.
        self.eval()

        # Retrieve the target layer from the feature extractor.
        if 'densenet' in self.config.backbone_name:
            target_layer = self.feature_extractor[0].denseblock4.denselayer16 # Specific layer in denseblock4
        elif 'efficientnet' in self.config.backbone_name:
            target_layer = self.feature_extractor[0][8][0]
        else:
            target_layer = self.feature_extractor[target_layer]

        # Function to reshape ViT outputs (if applicable) for Grad-CAM compatibility.
        def vit_reshape_transform(output):
            hidden_states = output[0]  # First element is the hidden states.
            return hidden_states[:, 1:, :].mean(dim=1)  # Mean pooling (ignores CLS token).

        # Initialize Grad-CAM with the selected target layer.
        gradcam = GradCAM(
            model=self,
            target_layers=[target_layer],
            reshape_transform=None  # Set to `vit_reshape_transform` for Vision Transformers.
        )

        # Sample random indices from the dataset for visualization.
        dataset = self.gradcam_loader.dataset
        samples = random.sample(range(len(dataset)), num_samples)

        for idx in samples:
            # Load the sample image, label, and optional metadata.
            inputs, label, img_name = dataset[idx]
            inputs = inputs.unsqueeze(0).to(self.device)  # Add a batch dimension and move to the correct device.
            label = int(label)

            # Generate the Grad-CAM heatmap for the input image.
            grayscale_cam = gradcam(input_tensor=inputs)[0]

            # Apply thresholding to remove low-importance regions.
            grayscale_cam[grayscale_cam < threshold] = 0

            # Normalize the input image for visualization.
            input_image = inputs[0].permute(1, 2, 0).cpu().numpy()
            input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
            input_image = input_image.astype(np.float32)

            # Overlay the heatmap on the input image.
            heatmap = show_cam_on_image(input_image, grayscale_cam, use_rgb=True)

            # Display the heatmap with the corresponding label and metadata.
            plt.imshow(heatmap)
            plt.title(f"Class: {class_names[label]} - {img_name} - (Threshold: {threshold})")
            plt.axis("off")
            plt.show()



#######################################################
#######################################################
class ViTPneumoniaClassifier(PneumoniaClassifier):
    def __init__(self, config):
        """
        Initializes a Vision Transformer (ViT)-based Pneumonia Classifier.
        Args:
            config (object): Configuration object containing model and training parameters.
        """
        super().__init__(config)

        # Load the ViT processor for preprocessing input images.
        self.processor = ViTImageProcessor.from_pretrained(config.backbone_name)

        # Load the pretrained ViT model for image classification with specified number of labels.
        self.feature_extractor = ViTForImageClassification.from_pretrained(
            config.backbone_name, num_labels=2, ignore_mismatched_sizes=True
        )

        # Replace the default classifier with an identity layer (removes the classification head).
        self.feature_extractor.classifier = nn.Identity()

        # Add a custom classifier layer to output logits for 2 classes (Normal and Pneumonia).
        self.classifier = nn.Linear(self.feature_extractor.config.hidden_size, 2)

        # Store the layers of the ViT encoder for potential Grad-CAM or layer-specific operations.
        self.vit_layers = list(self.feature_extractor.vit.encoder.layer)

        # If transfer learning is enabled, freeze the parameters of the pretrained model.
        if config.transfer_learning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

    def forward(self, x):
        """
        Defines the forward pass for the model.
        Args:
            x (torch.Tensor): Input tensor of images.
        Returns:
            torch.Tensor: Logits output by the classifier.
        """
        # Validate the input range to ensure all values are normalized in [0, 1].
        if x.max() > 1 or x.min() < 0:
            raise ValueError(f"Input tensor values must be in [0, 1]. Got range [{x.min()}, {x.max()}]")

        # Process the input images using the ViT processor and prepare them for the model.
        x = self.processor(images=x, return_tensors="pt", do_rescale=False)["pixel_values"].to(self.device)

        # Forward pass through the ViT model to extract features.
        vit_output = self.feature_extractor.vit(pixel_values=x)

        # Use the [CLS] token representation from the last hidden state as the feature vector.
        features = vit_output.last_hidden_state[:, 0, :]

        # Pass the features through the custom classifier to obtain logits.
        logits = self.classifier(features)
        return logits

    def create_transforms(self):
        """
        Creates data preprocessing and augmentation transforms for training, validation, and testing.
        Returns:
            tuple: A tuple containing train, validation, and test transforms.
        """
        train_transform = transforms.Compose([
            transforms.Lambda(make_square),  # Pad the image to make it square.
            transforms.Resize((self.config.image_res, self.config.image_res)),  # Resize to target resolution.
            transforms.RandomRotation(3),  # Apply small random rotations for augmentation.
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor.
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

    def get_vit_target_layer(self, layer_index=None):
        """
        Retrieves the target layer from the ViT encoder for visualization or layer-specific operations.
        Args:
            layer_index (int, optional): Index of the target layer. Defaults to the last layer.
        Returns:
            nn.Module: Target layer from the ViT encoder.
        """
        if layer_index is None:
            layer_index = len(self.vit_layers) - 1  # Default to the last layer.

        return self.vit_layers[layer_index]

    def vit_reshape_transform(self, tensor):
        """
        Reshapes the output tensor of the ViT model for Grad-CAM visualization.
        Args:
            tensor (torch.Tensor): Input tensor from the ViT model.
        Returns:
            torch.Tensor: Reshaped tensor with dimensions suitable for Grad-CAM.
        """
        if isinstance(tensor, tuple):
            tensor = tensor[0]  # Extract the primary output tensor.

        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"[DEBUG] Invalid input to reshape_transform. Expected a tensor, got {type(tensor)}")

        # Assuming tensor shape [batch_size, seq_len, hidden_dim].
        result = tensor[:, 1:, :].reshape(tensor.size(0), 14, 14, tensor.size(2))  # Adjust spatial dimensions.
        result = result.permute(0, 3, 1, 2)  # Convert to [batch_size, hidden_dim, height, width].
        return result

    def forward_hook(self, module, input, output):
        """
        Hook to capture intermediate outputs from a specified module during forward pass.
        Args:
            module (nn.Module): The module being hooked.
            input (tuple): Input to the module.
            output (torch.Tensor): Output from the module.
        Returns:
            torch.Tensor: The output of the module.
        """
        return output

    def visualize_gradcam(self, target_layer_index=None, threshold=0.5):
        """
        Simplified Grad-CAM visualization for a single image.
        Args:
            target_layer_index (int, optional): Index of the target layer for Grad-CAM. Defaults to the last layer.
            threshold (float, optional): Threshold for Grad-CAM heatmap values. Defaults to 0.5.
        """
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        import numpy as np
        import matplotlib.pyplot as plt

        print("Starting simplified Grad-CAM visualization...")

        # Load the first sample from the Grad-CAM loader dataset.
        image, label, _ = self.gradcam_loader.dataset[0]  # Get the first sample.
        class_names = ["Normal", "Pneumonia"]  # Example class names.

        # Ensure the model is in evaluation mode.
        self.eval()

        # Retrieve the target layer for Grad-CAM.
        if target_layer_index is None:
            target_layer_index = len(self.vit_layers) - 1
        target_layer = self.get_vit_target_layer(layer_index=target_layer_index)

        # Register a hook to capture outputs from the target layer.
        target_layer.register_forward_hook(self.forward_hook)

        # Initialize Grad-CAM with the specified target layer.
        gradcam = GradCAM(
            model=self,
            target_layers=[target_layer],
            reshape_transform=self.vit_reshape_transform
        )

        # Preprocess the image and add a batch dimension.
        image = image.unsqueeze(0).to(self.device)

        # Forward pass through the model with the preprocessed image.
        image.requires_grad = True
        outputs = self.forward(image)

        # Compute loss for the predicted class and perform backpropagation.
        loss = outputs[0, label]
        self.zero_grad()
        loss.backward(retain_graph=True)

        # Generate Grad-CAM heatmap for the input image.
        grayscale_cam = gradcam(input_tensor=image)[0]
        grayscale_cam[grayscale_cam < threshold] = 0  # Apply thresholding.

        # Normalize the input image for visualization.
        input_image = image[0].permute(1, 2, 0).cpu().numpy()
        input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())
        heatmap = show_cam_on_image(input_image.astype(np.float32), grayscale_cam, use_rgb=True)

        # Display the Grad-CAM heatmap.
        plt.imshow(heatmap)
        plt.title(f"Class: {class_names[label] if class_names else label}")
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