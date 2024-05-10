from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint  # , LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import torch
from torchmetrics import MeanMetric

from ... import utils
from ...typing import (
    Config,
    Dict,
    List,
    Metric,
    Module,
    Optional,
    Path,
    SummaryWriter,
    Tensor,
    Tuple,
)
from . import track
from .criterion import load_criterion
from .metric import as_attr, load_metrics
from .optimizer import load_optimizer
from .scheduler import load_scheduler
from .utils import check_config, get_lr, init_seed

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
        """
        Wrapper that employs PyTorch Lightning for model training.

        Parameters:
        - model (Module): Model to train.
        - config (Config): YAML configuration for training.
        """
        super().__init__()

        check_config(config)

        self.model = model
        self.config = config
        self.save_hyperparameters(config)

        utils.random.seed_everything(init_seed(config))

        self.criterion = load_criterion(config)

        # metrics (default)
        self.lr = MeanMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        # metrics (config)
        train_metrics = load_metrics(config)
        val_metrics = load_metrics(config)
        self.train_metrics_dict = self.init_metrics(train_metrics, header="train")
        self.val_metrics_dict = self.init_metrics(val_metrics, header="val")

        # track
        self._track = track.load_track(config)
        if self._track.FEATURE_RMS:
            self.feature_extractor, self.train_rms, self.val_rms = (
                track.feature_rms.init(model)
            )
        if self._track.GRAD_L2:
            self.grad_l2_dict = track.grad_l2.init(model)
        if self._track.PARAM_SVDVAL_MAX:
            self.param_svdval_max_dict = track.param_svdval_max.init(model)

    def configure_optimizers(self):
        optimizer = load_optimizer(self.config, self)
        lr_scheduler = load_scheduler(self.config, optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }

    def init_metrics(
        self,
        metrics: List[Tuple[str, Metric]],
        header: Optional[str] = "",
    ) -> Dict[str, Metric]:
        metrics_dict: Dict[str, Metric] = {}

        for metric_name, metric in metrics:
            metric_attr = as_attr(
                "_".join([header, metric_name]) if len(header) else metric_name
            )

            setattr(self, metric_attr, metric)
            metrics_dict[f"{header}/{metric_name}"] = getattr(self, metric_attr)

        return metrics_dict

    def training_step(self, batch, batch_idx):
        data: Tensor
        targets: Tensor

        data, targets = batch

        outputs: Tensor = self.model(data)
        loss: Tensor = self.criterion(outputs, targets)

        self.train_loss(loss / data.size(0))

        # metrics
        for _, metric in self.train_metrics_dict.items():
            # targets reshape for mixup/cutmix
            if len(targets.shape) == 2:
                targets = targets.argmax(dim=1)

            metric(outputs, targets)

        # track
        if self._track.FEATURE_RMS:
            track.feature_rms.compute(
                data=data,
                feature_extractor=self.feature_extractor,
                feature_rms_dict=self.train_rms,
            )

        return loss

    def on_after_backward(self) -> None:
        # track: grad_l2
        if self._track.GRAD_L2:
            track.grad_l2.compute(self.model, self.grad_l2_dict)

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch  # + 1

        # epoch as global_step
        self.log("step", current_epoch)

        # log: lr
        self.lr(get_lr(self.trainer))
        self.log(f"lr-{self.config['optimizer']}", self.lr)

        # log: loss
        self.log("train/avg_loss", self.train_loss)

        # log: metrics
        for label, metric in self.train_metrics_dict.items():
            self.log(label, metric)

        # log: track
        tensorboard: SummaryWriter = self.logger.experiment
        if self._track.FEATURE_RMS:
            track.feature_rms.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                feature_rms_dict=self.train_rms,
                header="feature_rms/train",
            )
        if self._track.GRAD_L2:
            track.grad_l2.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                grad_l2_dict=self.grad_l2_dict,
                header="grad_l2",
            )
        if self._track.PARAM_SVDVAL_MAX:
            track.param_svdval_max.compute(
                model=self.model,
                param_svdval_max_dict=self.param_svdval_max_dict,
            )
            track.param_svdval_max.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                param_svdval_max_dict=self.param_svdval_max_dict,
                header="param_svdval_max",
            )

    def validation_step(self, batch, batch_idx):
        data: Tensor
        targets: Tensor

        data, targets = batch

        outputs: Tensor = self.model(data)
        loss: Tensor = self.criterion(outputs, targets)

        # metrics
        self.val_loss(loss / data.size(0))

        for _, metric in self.val_metrics_dict.items():
            # targets reshape for mixup/cutmix
            if len(targets.shape) == 2:
                targets = targets.argmax(dim=1)

            metric(outputs, targets)

        # track
        if self._track.FEATURE_RMS:
            with torch.no_grad():
                # if not checked_layers, check
                if self.feature_extractor.checked_layers is False:
                    _ = self.feature_extractor(data)
                    self.train_rms = dict(
                        (i_layer, [])
                        for i_layer in self.feature_extractor.feature_layers.values()
                    )
                    self.val_rms = dict(
                        (i_layer, [])
                        for i_layer in self.feature_extractor.feature_layers.values()
                    )
                track.feature_rms.compute(
                    data=data,
                    feature_extractor=self.feature_extractor,
                    feature_rms_dict=self.val_rms,
                )

        return loss

    def on_validation_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1

        # epoch as global_step
        self.log("step", current_epoch)

        # log: loss
        self.log("val/avg_loss", self.val_loss)

        # log: metrics
        for label, metric in self.val_metrics_dict.items():
            self.log(label, metric)

        # log: track
        tensorboard: SummaryWriter = self.logger.experiment

        if self._track.FEATURE_RMS:
            track.feature_rms.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                feature_rms_dict=self.val_rms,
                header="feature_rms/val",
            )


def load_tensorboard_logger(
    config: Config,
    save_dir: Path,
    name: str,
) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=f"{config['seed']}-{datetime.now().strftime('%y%m%d%H%M%S')}",
    )


# def load_lr_monitor() -> LearningRateMonitor:
#     return LearningRateMonitor(logging_interval="epoch")


def load_checkpoint_callback(
    every_n_epochs: int,
) -> ModelCheckpoint:
    return ModelCheckpoint(
        filename="epoch={epoch:03d}-val_acc={val/acc@1:.4f}",
        # monitor="val/acc@1",
        save_top_k=-1,
        # mode="max",
        auto_insert_metric_name=False,
        every_n_epochs=every_n_epochs,
        save_on_train_epoch_end=False,
    )


def load_accelerator_and_devices(
    config: Config,
) -> Tuple[str, List[int]]:
    devices_cfg: str = config["devices"]
    devices_cfg: List[str] = devices_cfg.split(":")

    accelerator = "auto"
    devices = "auto"

    if len(devices) == 1:
        accelerator = devices_cfg[0]
    else:
        accelerator = devices_cfg[0]
        devices = [int(d) for d in devices_cfg[1].split(",")]

    return accelerator, devices


def load_trainer(
    config: Config,
    save_dir: Path,
    name: str,
    every_n_epochs: int,
) -> L.Trainer:
    accelerator, devices = load_accelerator_and_devices(config)

    tensorboard_logger = load_tensorboard_logger(
        config=config,
        save_dir=save_dir,
        name=name,
    )
    # lr_monitor = load_lr_monitor()
    checkpoint_callback = load_checkpoint_callback(every_n_epochs)

    return L.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=[
            tensorboard_logger,
        ],
        callbacks=[
            # lr_monitor,
            checkpoint_callback,
        ],
        max_epochs=config["epochs"],
        log_every_n_steps=None,
        deterministic="warn" if config["seed"] is not None else False,
    )
