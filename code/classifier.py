import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.datasets import ImageFolder
from code.dataloader import PneumoniaDataset
from code.custom_checkpoint import CustomModelCheckpoint
from code.project_globals import TEST_DIR, TRAIN_DIR, VAL_DIR
from torchvision import transforms
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from collections import Counter

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

        # Model backbone
        backbone = getattr(models, config.backbone_name)(weights='DEFAULT')
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        if config.transfer_learning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(num_filters, 2)

        # Create dataloaders and compute class weights
        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()
        if config.use_class_weights:
            # Precompute class weights and move to device
            self.class_weights = self.compute_class_weights().to(self.device)
        else:
            self.class_weights = None

        self.checkpoint_callback = CustomModelCheckpoint(
            monitor='val_loss',
            dirpath='../checkpoints',
            filename=self.config.model_name,
            save_top_k=1,
            mode='min'
        )
        self.early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=10,
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
            accumulate_grad_batches=2
        )

    def compute_class_weights(self):
        """
        Compute class weights based on the number of samples in each class.
        """
        class_counts = Counter(self.train_loader.dataset.targets)
        total_samples = sum(class_counts.values())
        class_weights = [total_samples / class_counts[i] for i in range(len(class_counts))]

        # Move class weights to the appropriate device
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



    def create_dataloaders(self):
        train_transform = transforms.Compose([
            transforms.Resize(self.config.image_res),
            transforms.CenterCrop((self.config.image_res*0.4, self.config.image_res*0.4)),
            #transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        val_test_transform = transforms.Compose([
            transforms.Resize(self.config.image_res),
            transforms.CenterCrop(self.config.image_res),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        if self.config.use_class_weights:
            train_folder = "../data/processed/train"
            val_folder = "../data/processed/val"
            test_folder = "../data/processed/test"
        else:
            train_folder = "../data/raw/train"
            val_folder = "../data/raw/val"
            test_folder = "../data/raw/test"

        train_dataset = ImageFolder(root=train_folder, transform=train_transform)
        val_dataset = ImageFolder(root=val_folder, transform=val_test_transform)
        test_dataset = ImageFolder(root=test_folder, transform=val_test_transform)

        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers, persistent_workers=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, persistent_workers=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=self.config.num_workers, persistent_workers=True)

        return train_loader, val_loader, test_loader

    def forward(self, x):
        # Do not set the feature extractor to eval here, as Grad-CAM requires gradients.
        representations = self.feature_extractor(x).flatten(1)
        representations = self.dropout(representations)
        return self.classifier(representations)

    def training_step(self, batch, batch_idx):
        data, label = batch
        print(
            f"Data device: {data.device}, Label device: {label.device}, Model device: {next(self.parameters()).device}, Class weights device: {self.class_weights.device}")
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

        # Log metrics
        self.log('train_loss_step', loss, prog_bar=True, on_step=True)
        return loss

    def on_fit_start(self):
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(self.device)

    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        val_output = self.forward(val_data)
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)

        val_preds = torch.argmax(val_output, dim=1)
        self.accuracy.update(val_preds, val_label)
        self.precision.update(val_preds, val_label)
        self.recall.update(val_preds, val_label)
        self.f1.update(val_preds, val_label)
        self.log('val_loss', val_loss)
        #self.log('val_acc_step', self.accuracy.compute(), prog_bar=True)
        #self.log('val_precision_step', self.precision.compute(), prog_bar=True)
        #self.log('val_recall_step', self.recall.compute(), prog_bar=True)
        #self.log('val_f1_step', self.f1.compute(), prog_bar=True)
        return val_loss

    def on_train_epoch_end(self):
        pass

    def test_step(self, batch, batch_idx):
        test_data, test_label = batch
        test_output = self.forward(test_data)
        test_loss = nn.CrossEntropyLoss()(test_output, test_label)

        test_preds = torch.argmax(test_output, dim=1)
        self.accuracy.update(test_preds, test_label)
        self.precision.update(test_preds, test_label)
        self.recall.update(test_preds, test_label)
        self.f1.update(test_preds, test_label)
        self.log('test_loss', test_loss)
        self.log('test_acc_step', self.accuracy.compute(), prog_bar=True)
        self.log('test_precision_step', self.precision.compute(), prog_bar=True)
        self.log('test_recall_step', self.recall.compute(), prog_bar=True)
        self.log('test_f1_step', self.f1.compute(), prog_bar=True)
        return test_loss

    def on_test_epoch_end(self):
        acc = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        self.log('test_acc_epoch', acc, prog_bar=True)
        self.log('test_precision_epoch', precision, prog_bar=True)
        self.log('test_recall_epoch', recall, prog_bar=True)
        self.log('test_f1_epoch', f1, prog_bar=True)
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def on_validation_epoch_end(self):
        acc = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()

        # Log validation epoch metrics
        self.log('val_acc_epoch', acc, prog_bar=True, on_epoch=True)
        self.log('val_precision_epoch', precision, prog_bar=True, on_epoch=True)
        self.log('val_recall_epoch', recall, prog_bar=True, on_epoch=True)
        self.log('val_f1_epoch', f1, prog_bar=True, on_epoch=True)

        # Reset metrics
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def configure_optimizers(self):
        if self.config.optimizer_name.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate,
                                         weight_decay=self.config.weight_decay)
        elif self.config.optimizer_name.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.config.learning_rate,
                                        weight_decay=self.config.weight_decay, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer_name}")

        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

    def train_model(self):
        self.trainer.fit(self, self.train_loader, self.val_loader)

    def test_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.trainer.test(self, self.test_loader)
        metadata = checkpoint.get("metadata", {})
        return metadata

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
            image_res
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
        self.use_class_weights = use_class_weights,
        self.image_res = image_res