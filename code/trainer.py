import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from code.classifier import PneumoniaClassifier, Config
from code.dataloader import PneumoniaDataset
from code.custom_checkpoint import CustomModelCheckpoint
from code.project_globals import TEST_DIR, TRAIN_DIR, VAL_DIR

import torch
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from code.classifier import PneumoniaClassifier, Config
from code.dataloader import PneumoniaDataset
from code.custom_checkpoint import CustomModelCheckpoint
from code.project_globals import TEST_DIR, TRAIN_DIR, VAL_DIR

class PneumoniaTrainer:
    def __init__(self, version, batch_size, max_epochs, backbone, model_name, num_workers, transform, weight_decay, dropout):
        self.version = version
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.backbone = backbone
        self.model_name = model_name
        self.num_workers = num_workers
        self.transform = transform

        self.train_loader, self.val_loader, self.test_loader = self.create_dataloaders()
        self.logger = TensorBoardLogger("tb_logs", name=self.model_name)
        self.config = Config(
            backbone_name=self.backbone,
            transfer_learning=True,
            learning_rate=1e-3,
            batch_size=self.batch_size,
            max_epochs=self.max_epochs,
            weight_decay=weight_decay,
            dropout=dropout
        )
        self.model = PneumoniaClassifier(self.config)
        self.checkpoint_callback = CustomModelCheckpoint(
            monitor='val_loss',
            dirpath='../checkpoints',
            filename=self.model_name + '-{epoch:02d}-{val_loss:.2f}_v' + self.version,
            save_top_k=1,
            mode='min'
        )
        self.early_stopping_callback = EarlyStopping(
            monitor='val_loss',
            patience=3,
            verbose=True,
            mode='min'
        )
        self.trainer = pl.Trainer(
            max_epochs=self.config.max_epochs,
            accelerator="gpu",
            devices=1,
            logger=self.logger,
            log_every_n_steps=1,
            callbacks=[self.checkpoint_callback, self.early_stopping_callback],
            precision='16-mixed',
            accumulate_grad_batches=2
        )

    def create_dataloaders(self):
        train = PneumoniaDataset(root_dir=TRAIN_DIR.as_posix(), transform=self.transform)
        test = PneumoniaDataset(root_dir=TEST_DIR.as_posix(), transform=self.transform)
        val = PneumoniaDataset(root_dir=VAL_DIR.as_posix(), transform=self.transform)

        train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True)
        test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)
        val_loader = torch.utils.data.DataLoader(dataset=val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True)

        return train_loader, val_loader, test_loader

    def train(self):
        self.trainer.fit(self.model, self.train_loader, self.val_loader)

    def test(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=True)
        self.trainer.test(self.model, self.test_loader)
        metadata = checkpoint.get("metadata", {})
        return metadata