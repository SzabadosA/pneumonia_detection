from pytorch_lightning.callbacks import ModelCheckpoint
import torch


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, metadata=None, **kwargs):
        super().__init__(**kwargs)
        self.metadata = metadata

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        checkpoint['metadata'] = self.metadata
        return checkpoint

    def _save_model(self, trainer, pl_module, filepath):
        # Call the parent method to save the model checkpoint
        super()._save_model(trainer, pl_module, filepath)

        # Add custom metadata
        metadata = {
            "epoch": trainer.current_epoch,
            "train_loss": trainer.callback_metrics.get("train_loss"),
            "val_loss": trainer.callback_metrics.get("val_loss"),
            "val_acc": trainer.callback_metrics.get("val_acc_epoch"),
            "hparams": pl_module.hparams if hasattr(pl_module, "hparams") else None,
        }

        # Load the existing checkpoint and add metadata
        checkpoint = torch.load(filepath)
        checkpoint["metadata"] = metadata

        # Save the checkpoint with metadata
        torch.save(checkpoint, filepath)
        print(f"Checkpoint with metadata saved at {filepath}")
