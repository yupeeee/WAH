import lightning as L
import torch
from torchmetrics import Accuracy, MeanMetric

from ...misc import path as _path
from ...misc import random
from ...misc.typing import Module, Optional, Path, Tensor, Trainer
from .criterion import load_criterion
from .optimizer import load_optimizer
from .scheduler import load_scheduler
from .utils import (
    check_config,
    get_lr,
    load_accelerator_and_devices,
    load_checkpoint_callback,
    load_tensorboard_logger,
)

__all__ = [
    "Wrapper",
    "load_trainer",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        **kwargs,
    ) -> None:
        super().__init__()
        check_config(**kwargs)
        self.model = model
        self.config = kwargs
        self.save_hyperparameters(self.config)
        # Set random seed for reproducibility
        seed = self.config.get("seed", -1)
        random.seed(seed)
        # Check if we need to sync distributed training
        devices = self.config["devices"]
        self.sync_dist = (
            isinstance(devices, str)
            and "gpu:" in devices
            and len(devices.split("gpu:")[-1].split(",")) > 1
        )
        # Load criterion
        self.train_criterion = load_criterion(
            train=True,
            reduction="none",
            **self.config,
        )
        self.val_criterion = load_criterion(
            train=False,
            reduction="none",
            **self.config,
        )
        self.softmax = torch.nn.Softmax(dim=-1)
        # Metrics
        self.lr = MeanMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_acc = Accuracy(
            task="multiclass",
            num_classes=self.config["num_classes"],
        )
        self.val_acc = Accuracy(
            task="multiclass",
            num_classes=self.config["num_classes"],
        )

    def configure_optimizers(self):
        optimizer = load_optimizer(self.model, **self.config)
        scheduler_config = {k: v for k, v in self.config.items() if k != "optimizer"}
        lr_scheduler = load_scheduler(optimizer, **scheduler_config)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }

    def training_step(self, batch, batch_idx):
        data: Tensor
        targets: Tensor
        # Unpack batch
        data, targets = batch
        # Forward pass
        outputs: Tensor = self.model(data)
        # Compute loss
        losses: Tensor = self.train_criterion(outputs, targets)
        # Update metrics
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
        self.train_loss(losses)
        self.train_acc(outputs, targets)
        # Return mean loss
        return losses.mean()

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1
        # epoch as global_step
        self.log("step", current_epoch, sync_dist=self.sync_dist)
        # log: lr
        self.lr(get_lr(self.trainer))
        self.log(f"lr-{self.config['optimizer']}", self.lr, sync_dist=self.sync_dist)
        # log: loss, acc@1
        self.log("train/avg_loss", self.train_loss, sync_dist=self.sync_dist)
        self.log("train/acc@1", self.train_acc, sync_dist=self.sync_dist)
        # Save checkpoint
        if self.trainer.is_global_zero:
            if "save_per_epoch" in self.config.keys():
                if (current_epoch + 1) % self.config["save_per_epoch"] == 0:
                    ckpt_dir = _path.join(self.trainer._log_dir, "checkpoints")
                    _path.mkdir(ckpt_dir)
                    torch.save(
                        self.model.state_dict(),
                        _path.join(ckpt_dir, f"epoch={current_epoch}.ckpt"),
                    )

    def validation_step(self, batch, batch_idx):
        data: Tensor
        targets: Tensor
        # Unpack batch
        data, targets = batch
        # Forward pass
        outputs: Tensor = self.model(data)
        # Compute loss
        losses: Tensor = self.val_criterion(outputs, targets)
        # Update metrics
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
        self.val_loss(losses)
        self.val_acc(outputs, targets)

    def on_validation_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1
        # epoch as global_step
        self.log("step", current_epoch, sync_dist=self.sync_dist)
        # log: loss, acc@1
        self.log("val/avg_loss", self.val_loss, sync_dist=self.sync_dist)
        self.log("val/acc@1", self.val_acc, sync_dist=self.sync_dist)


def load_trainer(
    save_dir: Path,
    name: str,
    version: Optional[str] = None,
    **kwargs,
) -> Trainer:
    config = kwargs
    # Load accelerator and devices
    accelerator, devices = load_accelerator_and_devices(config["devices"])
    # Load tensorboard logger
    tensorboard_logger = load_tensorboard_logger(
        save_dir=save_dir,
        name=name,
        version=version,
    )
    _log_dir = tensorboard_logger.log_dir
    checkpoint_callback = load_checkpoint_callback()
    # Set precision
    if config["amp"] is True:
        torch.set_float32_matmul_precision("medium")
    # Load trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if config["amp"] is True else "32-true",
        logger=[
            tensorboard_logger,
        ],
        callbacks=[
            checkpoint_callback,
        ],
        max_epochs=config["epochs"],
        log_every_n_steps=None,
        gradient_clip_val=(
            config["gradient_clip_val"]
            if "gradient_clip_val" in config.keys()
            else None
        ),
        deterministic="warn" if config["seed"] is not None else False,
    )
    # Set log directory
    setattr(trainer, "_log_dir", _log_dir)
    # Return trainer
    return trainer
