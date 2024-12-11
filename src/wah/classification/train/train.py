import os

import lightning as L
import torch
from torchmetrics import Accuracy, MeanMetric

from ...typing import Module, Optional, Path, Tensor, Trainer
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

        seed = self.config.get("seed", -1)
        if seed is None:
            seed = -1
        if seed >= 0:
            L.seed_everything(seed)

        self.sync_dist: bool = False
        if (
            "gpu" in self.config["devices"]
            and len(self.config["devices"].split("gpu:")[-1].split(",")) > 1
        ):
            self.sync_dist = True

        self.train_criterion = load_criterion(train=True, **self.config)
        self.val_criterion = load_criterion(train=False, **self.config)

        # metrics
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

        data, targets = batch

        outputs: Tensor = self.model(data)
        loss: Tensor = self.train_criterion(outputs, targets)

        self.train_loss(loss / data.size(0))

        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
        self.train_acc(outputs, targets)

        return loss

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

        # save checkpoint
        if "save_per_epoch" in self.config.keys():
            if (current_epoch + 1) % self.config["save_per_epoch"] == 0:
                ckpt_dir = os.path.join(self.trainer._log_dir, "checkpoints")
                os.makedirs(ckpt_dir, exist_ok=True)
                torch.save(
                    self.model.state_dict(),
                    os.path.join(ckpt_dir, f"epoch={current_epoch}.ckpt"),
                )

    def validation_step(self, batch, batch_idx):
        data: Tensor
        targets: Tensor

        data, targets = batch

        outputs: Tensor = self.model(data)
        loss: Tensor = self.val_criterion(outputs, targets)

        self.val_loss(loss / data.size(0))

        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
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

    accelerator, devices = load_accelerator_and_devices(config["devices"])

    tensorboard_logger = load_tensorboard_logger(
        save_dir=save_dir,
        name=name,
        version=version,
    )
    _log_dir = tensorboard_logger.log_dir
    checkpoint_callback = load_checkpoint_callback()

    if config["amp"] is True:
        torch.set_float32_matmul_precision("medium")

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
    setattr(trainer, "_log_dir", _log_dir)

    return trainer
