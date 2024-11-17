import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from torchvision import models
from torch.optim.lr_scheduler import ReduceLROnPlateau

class PneumoniaClassifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters(ignore=['backbone'])
        self.accuracy = Accuracy(task='binary')
        self.precision = Precision(task='binary')
        self.recall = Recall(task='binary')
        self.f1 = F1Score(task='binary')

        backbone = getattr(models, config.backbone_name)(weights='DEFAULT')
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        if config.transfer_learning:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(p=config.dropout)
        self.classifier = nn.Linear(num_filters, 2)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        representations = self.dropout(representations)
        return self.classifier(representations)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self.forward(data)
        loss = nn.CrossEntropyLoss()(output, label)

        preds = torch.argmax(output, dim=1)
        self.accuracy.update(preds, label)
        self.precision.update(preds, label)
        self.recall.update(preds, label)
        self.f1.update(preds, label)
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy.compute(), prog_bar=True)
        self.log('train_precision_step', self.precision.compute(), prog_bar=True)
        self.log('train_recall_step', self.recall.compute(), prog_bar=True)
        self.log('train_f1_step', self.f1.compute(), prog_bar=True)
        return loss

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
        self.log('val_acc_step', self.accuracy.compute(), prog_bar=True)
        self.log('val_precision_step', self.precision.compute(), prog_bar=True)
        self.log('val_recall_step', self.recall.compute(), prog_bar=True)
        self.log('val_f1_step', self.f1.compute(), prog_bar=True)
        return val_loss

    def on_train_epoch_end(self):
        acc = self.accuracy.compute()
        precision = self.precision.compute()
        recall = self.recall.compute()
        f1 = self.f1.compute()
        self.log('train_acc_epoch', acc, prog_bar=True)
        self.log('train_precision_epoch', precision, prog_bar=True)
        self.log('train_recall_epoch', recall, prog_bar=True)
        self.log('train_f1_epoch', f1, prog_bar=True)
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

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
        self.log('val_acc_epoch', acc, prog_bar=True)
        self.log('val_precision_epoch', precision, prog_bar=True)
        self.log('val_recall_epoch', recall, prog_bar=True)
        self.log('val_f1_epoch', f1, prog_bar=True)
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
        self.f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.config.learning_rate, weight_decay=self.config.weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'val_loss'}

class Config:
    def __init__(self, backbone_name, transfer_learning, learning_rate, batch_size, max_epochs, weight_decay, dropout):
        self.backbone_name = backbone_name
        self.transfer_learning = transfer_learning
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay
        self.dropout = dropout