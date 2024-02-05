from .typing import (
    Callable,
    Config,
    List,
    Metric,
    Module,
    Optional,
    Path,
    Tuple,
)

import math
import os

import torch
from torch import nn
import torch.optim as _optim
import torch.optim.lr_scheduler as _lr_scheduler
from torchvision.models.feature_extraction import (
    create_feature_extractor,
    get_graph_node_names,
)

import lightning as L
from torchmetrics import MeanMetric

from .log import (
    load_tensorboard_logger,
    load_lr_monitor,
    load_checkpoint_callback,
)
from .metrics import load_metric
from .utils import clean, seed_everything

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
        super(Wrapper, self).__init__()

        self.model = model
        self.config = config
        self.save_hyperparameters(self.config)

        seed_everything(self.config["seed"])

        self.criterion = self.load_criterion()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.train_metrics = load_metric(config)
        self.val_metrics = load_metric(config)

        self.init_metrics(self.train_metrics, header="train")
        self.init_metrics(self.val_metrics, header="val")

        _, eval_nodes = get_graph_node_names(self.model)
        self.feature_extractor = create_feature_extractor(
            model=self.model,
            return_nodes={eval_nodes[-2]: "features"},
        )
        self.train_features_l2 = []
        self.val_features_l2 = []

    def load_criterion(self) -> Callable:
        criterion_cfg = self.config["criterion"]

        if isinstance(criterion_cfg, str):
            return getattr(nn, criterion_cfg)()

        else:
            return criterion_cfg

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

    def init_metrics(
            self,
            metrics: List[Tuple[str, Metric]],
            header: Optional[str] = "",
    ) -> None:
        for label, _ in metrics:
            label = "_".join([header, label]) if len(header) else label

            setattr(self, clean(label), ...)

    def training_step(self, batch, batch_idx):
        data, targets = batch

        outputs = self.model(data)
        loss = self.criterion(outputs, targets)

        self.train_loss(loss / data.size(0))

        for _, metric in self.train_metrics:
            metric.to(self.device)(outputs, targets)

        with torch.no_grad():
            features = self.feature_extractor(data)["features"].reshape(len(batch), -1)
            features_l2 = torch.norm(features, p=2, dim=-1).cpu() / math.sqrt(features.size(-1))
            self.train_features_l2.append(features_l2)

        return loss

    def on_after_backward(self) -> None:
        for name, param in self.model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]  # removing '.'

            self.log(f"grads/l2_norm/{self.logger.name}/{layer}/{attr}", torch.norm(param.grad.flatten(), p=2))

    def on_train_epoch_end(self) -> None:
        # self.log("step", self.current_epoch)

        self.log("train/avg_loss", self.train_loss)

        for label, metric in self.train_metrics:
            self.log(f"train/{label}", metric, metric_attribute=f"train_{clean(label)}")

        tensorboard = self.logger.experiment

        self.train_features_l2 = torch.cat(self.train_features_l2)
        tensorboard.add_histogram(f"train/features/l2_norm", self.train_features_l2, self.current_epoch)
        self.train_features_l2 = []

        for name, param in self.model.named_parameters():
            layer, attr = os.path.splitext(name)
            attr = attr[1:]  # removing '.'

            tensorboard.add_histogram(f"params/{self.logger.name}/{layer}/{attr}", param, self.current_epoch)

    def validation_step(self, batch, batch_idx):
        data, targets = batch

        outputs = self.model(data)
        loss = self.criterion(outputs, targets)

        self.val_loss(loss / data.size(0))

        for _, metric in self.val_metrics:
            metric.to(self.device)(outputs, targets)

        with torch.no_grad():
            features = self.feature_extractor(data)["features"].reshape(len(batch), -1)
            features_l2 = torch.norm(features, p=2, dim=-1).cpu() / math.sqrt(features.size(-1))
            self.val_features_l2.append(features_l2)

        return loss

    def on_validation_epoch_end(self) -> None:
        # self.log("step", self.current_epoch)

        self.log("val/avg_loss", self.val_loss)

        for label, metric in self.val_metrics:
            self.log(f"val/{label}", metric, metric_attribute=f"val_{clean(label)}")

        tensorboard = self.logger.experiment

        self.val_features_l2 = torch.cat(self.val_features_l2)
        tensorboard.add_histogram(f"val/features/l2_norm", self.val_features_l2, self.current_epoch)
        self.val_features_l2 = []


def load_trainer(
        config: Config,
        save_dir: Path,
        name: str,
        every_n_epochs: int,
) -> L.Trainer:
    tensorboard_logger = load_tensorboard_logger(
        config=config,
        save_dir=save_dir,
        name=name,
    )
    lr_monitor = load_lr_monitor()
    checkpoint_callback = load_checkpoint_callback(every_n_epochs)

    return L.Trainer(
        logger=[tensorboard_logger, ],
        callbacks=[lr_monitor, checkpoint_callback, ],
        max_epochs=config["epochs"],
        log_every_n_steps=1,
        deterministic="warn" if config["seed"] is not None else False,
    )
