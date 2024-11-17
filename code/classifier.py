import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.classification import Accuracy

class PneumoniaClassifier(pl.LightningModule):
    def __init__(
            self,
            backbone,
            transfer_learning
        ):
        super().__init__()

        self.save_hyperparameters(ignore=['backbone'])
        self.accuracy = Accuracy(task='binary')
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        for param in self.feature_extractor.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(num_filters, 2)

    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        return self.classifier(representations)  # Return raw logits

    #def train_dataloader(self):
    #    return torch.utils.data.DataLoader(dataset=train, batch_size=32, shuffle=True)
#
    #def val_dataloader(self):
    #    return torch.utils.data.DataLoader(dataset=val, batch_size=32, shuffle=False)

    def training_step(self, batch, batch_idx):
        data, label = batch
        output = self.forward(data)  # Logits
        loss = nn.CrossEntropyLoss()(output, label)

        preds = torch.argmax(output, dim=1)  # Class indices
        self.log('train_loss', loss)
        self.log('train_acc_step', self.accuracy(preds, label))
        return loss

    def validation_step(self, batch, batch_idx):
        val_data, val_label = batch
        val_output = self.forward(val_data)  # Logits
        val_loss = nn.CrossEntropyLoss()(val_output, val_label)

        val_preds = torch.argmax(val_output, dim=1)  # Class indices
        self.log('val_loss', val_loss)
        self.log('val_acc_step', self.accuracy(val_preds, val_label))
        return val_loss

    def on_train_epoch_end(self):
        acc = self.accuracy.compute()
        self.log('train_acc_epoch', acc, prog_bar=True)
        lr = self.optimizers().param_groups[0]['lr']
        print(f"Learning Rate after epoch {self.current_epoch}: {lr}")
        self.accuracy.reset()

    def on_validation_epoch_end(self):
        acc = self.accuracy.compute()
        self.log('val_acc_epoch', acc, prog_bar=True)
        self.accuracy.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        return [optimizer], [scheduler]