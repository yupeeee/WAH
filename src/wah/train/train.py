import math

import lightning as L
import torch
from torch import nn
import torch.optim as _optim
from torchmetrics import MeanMetric
from torchvision.models.feature_extraction import (
    get_graph_node_names,
    create_feature_extractor,
)

from ..typing import (
    Callable,
    Config,
    Dict,
    List,
    Metric,
    Module,
    Optional,
    Path,
)
from .log import (
    load_checkpoint_callback,
    load_lr_monitor,
    load_tensorboard_logger,
)
from .scheduler import load_scheduler
from .utils import (
    clean,
    load_metrics,
    seed_everything,
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

        self.train_metrics = \
            self.init_metrics(train_metrics, header="train")
        self.val_metrics = \
            self.init_metrics(val_metrics, header="val")

        # grad_l2
        self.track_grad_l2 = \
            True if "grad_l2" in self.config["track"] else False
        self.grad_l2 = {}

        if self.track_grad_l2:
            self.init_grad_l2()

        # feature_rms
        self.track_feature_rms = \
            True if "feature_rms" in self.config["track"] else False
        self.feature_extractor = None
        self.layers = []
        self.train_rms = {}
        self.val_rms = {}

        if self.track_feature_rms:
            self.init_feature_rms()

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
            }
        }

    def init_metrics(
        self,
        metrics: List[Metric],
        header: Optional[str] = "",
    ) -> Dict[str, Metric]:
        metrics_dict = {}

        for metric in metrics:
            name = "_".join([header, metric.label]) \
                if len(header) else metric.label
            name = clean(name)

            setattr(self, name, metric)
            metrics_dict[f"{header}/{metric.label}"] = getattr(self, name)

        return metrics_dict

    def init_grad_l2(self, ) -> None:
        for name, _ in self.model.named_parameters():
            # layer, attr = os.path.splitext(name)
            # attr = attr[1:]  # removing '.'
            # label = f"{layer}/{attr}"

            self.grad_l2[name] = []

    def init_feature_rms(self, ) -> None:
        # features/l2_norm
        _, self.layers = get_graph_node_names(self.model)
        self.feature_extractor = create_feature_extractor(
            model=self.model,
            return_nodes=self.layers,
        )

        for layer in self.layers:
            self.train_rms[layer] = []
            self.val_rms[layer] = []

    def training_step(self, batch, batch_idx):
        data, targets = batch

        outputs = self.model(data)
        loss = self.criterion(outputs, targets)

        self.train_loss(loss / data.size(0))

        for _, metric in self.train_metrics.items():
            metric.to(self.device)(outputs, targets)

        if self.track_feature_rms:
            with torch.no_grad():
                features = self.feature_extractor(data)

                for layer in self.layers:
                    f = features[layer].reshape(len(batch), -1)
                    f_rms = torch.norm(f, p=2, dim=-1) / math.sqrt(f.size(-1))
                    self.train_rms[layer].append(f_rms)

                    del f
                    torch.cuda.empty_cache()

        return loss

    def on_after_backward(self) -> None:
        if self.track_grad_l2:
            for name, param in self.model.named_parameters():
                grad_l2 = torch.norm(param.grad.flatten(), p=2)
                self.grad_l2[name].append(grad_l2.view(1))

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1

        # global step as epoch
        self.log("step", current_epoch)

        self.log("train/avg_loss", self.train_loss)

        for label, metric in self.train_metrics.items():
            self.log(label, metric)

        tensorboard = self.logger.experiment

        # grad_l2
        if self.track_grad_l2:
            for layer, l2 in self.grad_l2.items():
                l2 = torch.cat(l2)
                tensorboard.add_histogram(
                    f"grad_l2/{layer}",
                    l2,
                    current_epoch,
                )

                # reset
                self.grad_l2[layer] = []

        # feature_rms
        if self.track_feature_rms:
            for layer, rms in self.train_rms.items():
                rms = torch.cat(rms)
                tensorboard.add_histogram(
                    f"feature_rms/train/{layer}",
                    rms,
                    current_epoch,
                )

                # reset
                self.train_rms[layer] = []

    def validation_step(self, batch, batch_idx):
        data, targets = batch

        outputs = self.model(data)
        loss = self.criterion(outputs, targets)

        self.val_loss(loss / data.size(0))

        for _, metric in self.val_metrics.items():
            metric.to(self.device)(outputs, targets)

        if self.track_feature_rms:
            with torch.no_grad():
                features = self.feature_extractor(data)

                for layer in self.layers:
                    f = features[layer].reshape(len(batch), -1)
                    f_rms = torch.norm(f, p=2, dim=-1) / math.sqrt(f.size(-1))
                    self.val_rms[layer].append(f_rms)

                    del f
                    torch.cuda.empty_cache()

        return loss

    def on_validation_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1

        # global step as epoch
        self.log("step", current_epoch)

        # loss
        self.log("val/avg_loss", self.val_loss)

        for label, metric in self.val_metrics.items():
            self.log(label, metric)

        tensorboard = self.logger.experiment

        # feature_rms
        if self.track_feature_rms:
            for layer, rms in self.val_rms.items():
                rms = torch.cat(rms)
                tensorboard.add_histogram(
                    f"feature_rms/val/{layer}",
                    rms,
                    current_epoch,
                )

                # reset
                self.val_rms[layer] = []


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
        logger=[tensorboard_logger, ],
        callbacks=[lr_monitor, checkpoint_callback, ],
        max_epochs=config["epochs"],
        log_every_n_steps=1,
        deterministic="warn" if config["seed"] is not None else False,
    )
