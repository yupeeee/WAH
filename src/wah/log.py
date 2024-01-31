from .typing import (
    Config,
)

from datetime import datetime

from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

__all__ = [
    "load_tensorboard_logger",
    "load_lr_monitor",
    "load_checkpoint_callback",
]


def load_tensorboard_logger(
        config: Config,
        save_dir: str,
        name: str,
) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=f"{config['seed']}-{datetime.now().strftime('%y%m%d%H%M%S')}",
    )


def load_lr_monitor() -> LearningRateMonitor:
    return LearningRateMonitor()


def load_checkpoint_callback(
        every_n_epochs: int,
) -> ModelCheckpoint:
    return ModelCheckpoint(
        # monitor="val/acc@1",
        # mode="max",
        save_top_k=-1,
        filename="epoch={epoch:03d}-val_acc={val/acc@1:.4f}",
        auto_insert_metric_name=False,
        every_n_epochs=every_n_epochs,
        save_on_train_epoch_end=False,
    )
