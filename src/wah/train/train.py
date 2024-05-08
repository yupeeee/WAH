# import math

import lightning as L
import torch
import torch.optim as _optim
from torch import nn
from torchmetrics import MeanMetric

# from ..models.feature_extractor import FeatureExtractor
from ..typing import (
    Callable,
    Config,
    Dict,
    List,
    Metric,
    Module,
    Optional,
    Path,
    Tensor,
)
from ..utils.random import seed_everything
from .log import (
    load_checkpoint_callback,
    load_lr_monitor,
    load_tensorboard_logger,
)
from .scheduler import load_scheduler
from .utils import clean, load_metrics

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

        seed_everything(self.config["seed"])

        self.criterion = self.load_criterion()

        ###########
        # Metrics #
        ###########
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # config: train/val metrics
        train_metrics = load_metrics(config, train=True)
        val_metrics = load_metrics(config, train=False)

        self.train_metrics: Dict[str, Metric] = self.init_metrics(
            train_metrics, header="train",
        )
        self.val_metrics: Dict[str, Metric] = self.init_metrics(
            val_metrics, header="val",
        )

        # grad_l2
        self.grad_layers: Dict[str, str] = {}
        self.grad_l2: Dict[str, List[Tensor]] = {}

        self.init_grad_l2()

        # self.track_grad_l2: bool = False
        # self.track_feature_rms: bool = False

        # if "track" in self.config.keys():
        #     # grad_l2
        #     self.track_grad_l2 = (
        #         True if "grad_l2" in self.config["track"] else False
        #     )
        #     self.grad_layers: Dict[str, str] = {}
        #     self.grad_l2: Dict[str, List[Tensor]] = {}

        #     if self.track_grad_l2:
        #         self.init_grad_l2()

        #     # feature_rms
        #     self.track_feature_rms = (
        #         True if "feature_rms" in self.config["track"] else False
        #     )
        #     self.feature_extractor: Module = None
        #     self.train_rms: Dict[str, List[Tensor]] = {}
        #     self.val_rms: Dict[str, List[Tensor]] = {}

        #     if self.track_feature_rms:
        #         self.init_feature_rms()

    def load_criterion(self) -> Callable:
        criterion = getattr(nn, self.config["criterion"])

        if "criterion_cfg" in self.config.keys():
            criterion_cfg = self.config["criterion_cfg"]
            criterion = criterion(**criterion_cfg)

        else:
            criterion = criterion()

        return criterion

    def configure_optimizers(self):
        optimizer = getattr(_optim, self.config["optimizer"])(
            params=self.parameters(),
            lr=self.config["init_lr"],
            **self.config["optimizer_cfg"],
        )
        lr_scheduler = load_scheduler(self.config, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }

    def init_metrics(
        self,
        metrics: List[Metric],
        header: Optional[str] = "",
    ) -> Dict[str, Metric]:
        metrics_dict = {}

        for metric in metrics:
            name = "_".join([header, metric.label]) if len(header) else metric.label
            name = clean(name)

            setattr(self, name, metric)
            metrics_dict[f"{header}/{metric.label}"] = getattr(self, name)

        return metrics_dict

    def init_grad_l2(
        self,
    ) -> None:
        for i, (name, _) in enumerate(self.model.named_parameters()):
            self.grad_layers[name] = f"{i}_{name}"
            self.grad_l2[f"{i}_{name}"] = []

    # def init_feature_rms(
    #     self,
    # ) -> None:
    #     self.feature_extractor = FeatureExtractor(self.model)

    #     self.train_rms = dict(
    #         (layer, []) for layer in self.feature_extractor.feature_layers.values()
    #     )
    #     self.val_rms = dict(
    #         (layer, []) for layer in self.feature_extractor.feature_layers.values()
    #     )

    # @staticmethod
    # def flatten_feature(feature, batch) -> Tensor:
    #     # vit: self_attention
    #     if isinstance(feature, tuple):
    #         feature = [f for f in feature if f is not None]
    #         feature = torch.cat(feature, dim=0)

    #     feature = feature.reshape(len(batch), -1)

    #     return feature

    def training_step(self, batch, batch_idx):
        rank = self.local_rank

        data, targets = batch
        # data: Tensor = data.to(rank)
        # targets: Tensor = targets.to(rank)

        outputs: Tensor = self.model(data)
        loss: Tensor = self.criterion(outputs, targets)

        self.train_loss(loss / data.size(0))

        for _, metric in self.train_metrics.items():
            # mixup/cutmix target reshape
            if len(targets.shape) == 2:
                targets = targets.argmax(dim=1)

            metric(outputs, targets)

        # if self.track_feature_rms:
        #     with torch.no_grad():
        #         features: Dict[str, Tensor] = self.feature_extractor(data)

        #         for i_layer, feature in features.items():
        #             feature = self.flatten_feature(feature, batch)

        #             f_rms = torch.norm(feature, p=2, dim=-1) / math.sqrt(
        #                 feature.size(-1)
        #             )
        #             self.train_rms[i_layer].append(f_rms)

        #             del feature
        #             torch.cuda.empty_cache()

        return loss

    def on_after_backward(self) -> None:
        # if self.track_grad_l2:
        for i, (name, param) in enumerate(self.model.named_parameters()):
            grad_l2 = torch.norm(param.grad.flatten(), p=2)
            self.grad_l2[f"{i}_{name}"].append(grad_l2.view(1))

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1

        # global step as epoch
        self.log("step", current_epoch)

        self.log("train/avg_loss", self.train_loss)

        for label, metric in self.train_metrics.items():
            self.log(label, metric)

        tensorboard = self.logger.experiment

        # grad_l2
        # if self.track_grad_l2:
        for layer, l2 in self.grad_l2.items():
            l2 = torch.cat(l2)
            tensorboard.add_histogram(
                f"grad_l2/{layer}",
                l2,
                current_epoch,
            )

            # reset
            self.grad_l2[layer].clear()

        # # feature_rms
        # if self.track_feature_rms:
        #     for layer, rms in self.train_rms.items():
        #         rms = torch.cat(rms)
        #         tensorboard.add_histogram(
        #             f"feature_rms/train/{layer}",
        #             rms,
        #             current_epoch,
        #         )

        #         # reset
        #         self.train_rms[layer].clear()

    def validation_step(self, batch, batch_idx):
        data, targets = batch

        outputs = self.model(data)
        loss = self.criterion(outputs, targets)

        self.val_loss(loss / data.size(0))

        for _, metric in self.val_metrics.items():
            # mixup/cutmix target reshape
            if len(targets.shape) == 2:
                targets = targets.argmax(dim=1)

            metric(outputs, targets)

        # if self.track_feature_rms:
        #     with torch.no_grad():
        #         if self.feature_extractor.checked_layers is False:
        #             _ = self.feature_extractor(data)
        #             self.train_rms = dict(
        #                 (layer, [])
        #                 for layer in self.feature_extractor.feature_layers.values()
        #             )
        #             self.val_rms = dict(
        #                 (layer, [])
        #                 for layer in self.feature_extractor.feature_layers.values()
        #             )

        #         features = self.feature_extractor(data)

        #         for i_layer, feature in features.items():
        #             feature = self.flatten_feature(feature, batch)

        #             f_rms = torch.norm(feature, p=2, dim=-1) / math.sqrt(
        #                 feature.size(-1)
        #             )
        #             self.val_rms[i_layer].append(f_rms)

        #             del feature
        #             torch.cuda.empty_cache()

        return loss

    def on_validation_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1

        # global step as epoch
        self.log("step", current_epoch)

        # loss
        self.log("val/avg_loss", self.val_loss)

        for label, metric in self.val_metrics.items():
            self.log(label, metric)

        # tensorboard = self.logger.experiment

        # # feature_rms
        # if self.track_feature_rms:
        #     for layer, rms in self.val_rms.items():
        #         rms = torch.cat(rms)
        #         tensorboard.add_histogram(
        #             f"feature_rms/val/{layer}",
        #             rms,
        #             current_epoch,
        #         )

        #         # reset
        #         self.val_rms[layer].clear()


def load_trainer(
    config: Config,
    save_dir: Path,
    name: str,
    every_n_epochs: int,
) -> L.Trainer:
    accelerator = "auto"
    devices = "auto"

    if "gpu" in config.keys():
        accelerator = "gpu"
        devices = config["gpu"]

    tensorboard_logger = load_tensorboard_logger(
        config=config,
        save_dir=save_dir,
        name=name,
    )
    lr_monitor = load_lr_monitor()
    checkpoint_callback = load_checkpoint_callback(every_n_epochs)

    return L.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=[
            tensorboard_logger,
        ],
        callbacks=[
            lr_monitor,
            checkpoint_callback,
        ],
        max_epochs=config["epochs"],
        log_every_n_steps=1,
        deterministic="warn" if config["seed"] is not None else False,
    )
