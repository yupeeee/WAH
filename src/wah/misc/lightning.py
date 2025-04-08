from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from .typing import Devices, List, LRScheduler, Optional, Path, Trainer, Tuple

__all__ = [
    "get_lr",
    "load_accelerator_and_devices",
    "load_checkpoint_callback",
    "load_tensorboard_logger",
    "ProgressBar",
]


def get_lr(
    trainer: Trainer,
) -> float:
    scheduler: LRScheduler = trainer.lr_scheduler_configs[0].scheduler
    lr = scheduler.get_last_lr()[0]
    return lr


def load_accelerator_and_devices(
    devices: Devices = "auto",
) -> Tuple[str, List[int]]:
    if isinstance(devices, str) and "cuda" in devices:
        devices = devices.replace("cuda", "gpu")
    devices_cfg: List[str] = devices.split(":")
    accelerator = "auto"
    devices = "auto"
    if len(devices_cfg) == 1:
        accelerator = devices_cfg[0]
    else:
        accelerator = devices_cfg[0]
        devices = [int(d) for d in devices_cfg[1].split(",")]
    return accelerator, devices


def load_checkpoint_callback() -> ModelCheckpoint:
    return ModelCheckpoint(
        save_last=True,
        save_top_k=0,
        save_on_train_epoch_end=True,
    )


def load_tensorboard_logger(
    save_dir: Path,
    name: str,
    version: Optional[str] = None,
) -> TensorBoardLogger:
    return TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
    )


class ProgressBar(TQDMProgressBar):
    def __init__(self, desc: str):
        super().__init__()
        self.desc = desc

    def init_validation_tqdm(self):
        bar = super().init_validation_tqdm()
        bar.set_description(self.desc)
        return bar
