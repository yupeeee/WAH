from datetime import datetime

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
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
from .utils import (
    check_config,
    get_lr,
    get_tag,
    init_seed,
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
        """
        Wrapper that employs PyTorch Lightning for model training.

        ### Parameters
        - `model (Module)`: Model to train.
        - `config (Config)`: YAML configuration for training.
        """
        super().__init__()

        check_config(config)

        self.model = model
        self.config = config
        self.save_hyperparameters(config)

        utils.random.seed_everything(init_seed(config))

        self.sync_dist: bool = False
        if (
            "gpu" in config["devices"]
            and len(config["devices"].split("gpu:")[-1].split(",")) > 1
        ):
            self.sync_dist = True

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
        if self._track.FEATURE_SIGN:
            self.feature_extractor, self.train_sign, self.val_sign = (
                track.feature_sign.init(model)
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
        if self._track.FEATURE_SIGN:
            track.feature_sign.compute(
                data=data,
                feature_extractor=self.feature_extractor,
                feature_sign_dict=self.train_sign,
            )

        return loss

    def on_after_backward(self) -> None:
        # track: grad_l2
        if self._track.GRAD_L2:
            track.grad_l2.compute(self.model, self.grad_l2_dict)

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch  # + 1

        # epoch as global_step
        self.log("step", current_epoch, sync_dist=self.sync_dist)

        # log: lr
        self.lr(get_lr(self.trainer))
        self.log(f"lr-{self.config['optimizer']}", self.lr, sync_dist=self.sync_dist)

        # log: loss
        self.log("train/avg_loss", self.train_loss, sync_dist=self.sync_dist)

        # log: metrics
        for label, metric in self.train_metrics_dict.items():
            self.log(label, metric, sync_dist=self.sync_dist)

        # log: track
        tensorboard: SummaryWriter = self.logger.experiment
        tag = get_tag(self.trainer)

        if self._track.FEATURE_RMS:
            track.feature_rms.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                feature_rms_dict=self.train_rms,
                header=f"feature_rms/{tag}/train",
            )
            track.feature_rms.reset(self.train_rms)
        if self._track.FEATURE_SIGN:
            track.feature_sign.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                feature_sign_dict=self.train_sign,
                header=f"feature_sign/{tag}/train",
            )
            track.feature_sign.reset(self.train_sign)
        if self._track.GRAD_L2:
            track.grad_l2.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                grad_l2_dict=self.grad_l2_dict,
                header=f"grad_l2/{tag}",
            )
            track.grad_l2.reset(self.grad_l2_dict)
        if self._track.PARAM_SVDVAL_MAX:
            track.param_svdval_max.compute(
                model=self.model,
                param_svdval_max_dict=self.param_svdval_max_dict,
            )
            track.param_svdval_max.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                param_svdval_max_dict=self.param_svdval_max_dict,
                header=f"param_svdval_max/{tag}",
            )
            track.param_svdval_max.reset(self.param_svdval_max_dict)
        if self._track.STATE_DICT:
            tensorboard_log_dir = self.trainer.tensorboard_log_dir

            track.state_dict.save(
                model=self.model,
                epoch=current_epoch,
                every_n_epochs=1,
                tensorboard_log_dir=tensorboard_log_dir,
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

        # track; if not checked_layers, check
        if self._track.FEATURE_RMS or self._track.FEATURE_SIGN:
            with torch.no_grad():
                if self.feature_extractor.checked_layers is False:
                    _ = self.feature_extractor(data)

                    if self._track.FEATURE_RMS:
                        self.train_rms = dict(
                            (i_layer, [])
                            for i_layer in self.feature_extractor.feature_layers.values()
                        )
                        self.val_rms = dict(
                            (i_layer, [])
                            for i_layer in self.feature_extractor.feature_layers.values()
                        )
                    if self._track.FEATURE_SIGN:
                        self.train_sign = dict(
                            (i_layer, [])
                            for i_layer in self.feature_extractor.feature_layers.values()
                        )
                        self.val_sign = dict(
                            (i_layer, [])
                            for i_layer in self.feature_extractor.feature_layers.values()
                        )

        # track
        if self._track.FEATURE_RMS:
            track.feature_rms.compute(
                data=data,
                feature_extractor=self.feature_extractor,
                feature_rms_dict=self.val_rms,
            )
        if self._track.FEATURE_SIGN:
            track.feature_sign.compute(
                data=data,
                feature_extractor=self.feature_extractor,
                feature_sign_dict=self.val_sign,
            )

        return loss

    def on_validation_epoch_end(self) -> None:
        current_epoch = self.current_epoch  # + 1

        # epoch as global_step
        self.log("step", current_epoch, sync_dist=self.sync_dist)

        # log: loss
        self.log("val/avg_loss", self.val_loss, sync_dist=self.sync_dist)

        # log: metrics
        for label, metric in self.val_metrics_dict.items():
            self.log(label, metric, sync_dist=self.sync_dist)

        # log: track
        tensorboard: SummaryWriter = self.logger.experiment
        tag = get_tag(self.trainer)

        if self._track.FEATURE_RMS:
            track.feature_rms.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                feature_rms_dict=self.val_rms,
                header=f"feature_rms/{tag}/val",
            )
            track.feature_rms.reset(self.val_rms)
        if self._track.FEATURE_SIGN:
            track.feature_sign.track(
                epoch=current_epoch,
                tensorboard=tensorboard,
                feature_sign_dict=self.val_sign,
                header=f"feature_sign/{tag}/val",
            )
            track.feature_sign.reset(self.val_sign)


def load_tensorboard_logger(
    config: Config,
    save_dir: Path,
    name: str,
    version: Optional[str] = None,
) -> TensorBoardLogger:
    """
    Loads the TensorBoard logger for training.

    ### Parameters
    - `config (Config)`: YAML configuration for training.
    - `save_dir (Path)`: Directory to save the logs.
    - `name (str)`: Name of the logging experiment.
    - `version (Optional[str])`: Version of the logging experiment. If `None`, uses the current timestamp.

    ### Returns
    - `TensorBoardLogger`: The TensorBoard logger.
    """
    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=(
            f"{config['seed']}-{datetime.now().strftime('%y%m%d%H%M%S')}"
            if version is None
            else version
        ),
    )


def load_checkpoint_callback() -> ModelCheckpoint:
    """
    Loads the model checkpoint callback.

    ### Returns
    - `ModelCheckpoint`: The model checkpoint callback.
    """
    return ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        save_on_train_epoch_end=True,
    )


def load_accelerator_and_devices(
    config: Config,
) -> Tuple[str, List[int]]:
    """
    Loads the accelerator and devices based on the configuration.

    ### Parameters
    - `config (Config)`: YAML configuration for training.

    ### Returns
    - `Tuple[str, List[int]]`: The accelerator type and list of devices.
    """
    devices_cfg: str = config["devices"]
    devices_cfg: List[str] = devices_cfg.split(":")

    accelerator = "auto"
    devices = "auto"

    if len(devices_cfg) == 1:
        accelerator = devices_cfg[0]
    else:
        accelerator = devices_cfg[0]
        devices = [int(d) for d in devices_cfg[1].split(",")]

    return accelerator, devices


def load_trainer(
    config: Config,
    save_dir: Path,
    name: str,
    version: Optional[str] = None,
) -> L.Trainer:
    """
    Loads the PyTorch Lightning trainer based on the configuration.

    ### Parameters
    - `config (Config)`: YAML configuration for training.
    - `save_dir (Path)`: Directory to save the logs and checkpoints.
    - `name (str)`: Name of the training experiment.
    - `version (Optional[str])`: Version of the training experiment. If `None`, uses the current timestamp.

    ### Returns
    - `L.Trainer`: The PyTorch Lightning trainer.
    """
    accelerator, devices = load_accelerator_and_devices(config)

    tensorboard_logger = load_tensorboard_logger(
        config=config,
        save_dir=save_dir,
        name=name,
        version=version,
    )
    tensorboard_log_dir = tensorboard_logger.log_dir
    checkpoint_callback = load_checkpoint_callback()

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=[
            tensorboard_logger,
        ],
        callbacks=[
            checkpoint_callback,
        ],
        max_epochs=config["epochs"],
        log_every_n_steps=None,
        deterministic="warn" if config["seed"] is not None else False,
    )
    setattr(trainer, "tensorboard_log_dir", tensorboard_log_dir)

    return trainer
