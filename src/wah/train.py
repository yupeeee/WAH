from .typing import Config, Module

import os

import torch
import torch.nn.functional as F
import torch.optim as _optim
import torch.optim.lr_scheduler as _lr_scheduler

import lightning as L
from torchmetrics import (
    Accuracy,
    CalibrationError,
    MeanMetric,
)

__all__ = [
    "Wrapper",
    "load_trainer",
]


class Wrapper(L.LightningModule):
    def __init__(
            self,
            model: Module,
            config: Config,
    ) -> None:
        super().__init__()

        self.model = model
        self.config = config
        self.save_hyperparameters(self.config)

        L.seed_everything(self.config["seed"])

        self.train_loss = MeanMetric()
        self.train_acc = Accuracy(task="multiclass", num_classes=self.config["num_classes"])
        self.train_ece = CalibrationError(task="multiclass", num_classes=self.config["num_classes"])

        self.val_loss = MeanMetric()
        self.val_acc = Accuracy(task="multiclass", num_classes=self.config["num_classes"])
        self.val_ece = CalibrationError(task="multiclass", num_classes=self.config["num_classes"])

    def configure_optimizers(self):
        optimizer = getattr(_optim, self.config["optimizer"])(
            params=self.parameters(),
            lr=self.config["init_lr"],
            **self.config["optimizer_cfg"],
        )
        lr_scheduler = getattr(_lr_scheduler, self.config["lr_scheduler"])(
            optimizer=optimizer,
            **self.config["lr_scheduler_cfg"],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            }
        }

    def training_step(self, batch, batch_idx):
        data, targets = batch

        outputs = self.model(data)
        loss = F.cross_entropy(outputs, targets)

        self.train_loss(loss / data.size(0))
        self.train_acc(outputs, targets)
        self.train_ece(outputs, targets)

        return loss

    def on_after_backward(self) -> None:
        for name, param in self.model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]  # removing '.'

            self.log(f"grads/{layer}/{attr}", torch.norm(param.grad.flatten()))

    def on_train_epoch_end(self) -> None:
        # self.log("step", self.current_epoch)

        self.log("train/avg_loss", self.train_loss)
        self.log("train/acc@1", self.train_acc)
        self.log("train/ece", self.train_ece)

        tensorboard = self.logger.experiment

        for name, param in self.model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]  # removing '.'

            tensorboard.add_histogram(f"params/{layer}/{attr}", param, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        data, targets = batch

        outputs = self.model(data)
        loss = F.cross_entropy(outputs, targets)

        self.val_loss(loss / data.size(0))
        self.val_acc(outputs, targets)
        self.val_ece(outputs, targets)

        return loss

    def on_validation_epoch_end(self) -> None:
        # self.log("step", self.current_epoch)

        self.log("val/avg_loss", self.val_loss)
        self.log("val/acc@1", self.val_acc)
        self.log("val/ece", self.val_ece)


def load_trainer(
        config: Config,
        logger,
        callbacks,
) -> L.Trainer:
    return L.Trainer(
        logger=logger,
        max_epochs=config["epochs"],
        callbacks=callbacks,
        deterministic="warn" if config["seed"] is not None else False,
    )
