import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from ...typing import (
    Devices,
    List,
    LRScheduler,
    Optional,
    Path,
    Sequence,
    Tensor,
    Trainer,
    Tuple,
)

__all__ = [
    "check_config",
    "get_lr",
    "load_accelerator_and_devices",
    "load_checkpoint_callback",
    "load_tensorboard_logger",
    "process_gathered_data",
]

config_requirements: List[str] = [
    "devices",
    "num_classes",
    "batch_size",
    "num_workers",
    "epochs",
    "init_lr",
    "optimizer",
    "criterion",
]


def check_config(
    **config,
) -> None:
    required_args = config_requirements
    missing_args = [arg for arg in required_args if arg not in config.keys()]

    assert (
        len(missing_args) == 0
    ), f"{len(missing_args)} missing args in config: {', '.join(missing_args)}"


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


def process_gathered_data(
    data: List[Tensor],
    unsqueeze_until: int = 1,
    cat_along: int = -1,
    permute_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    assert (
        data[0].dim() <= unsqueeze_until
    ), f"x.dim() in data must be smaller than or equal to unsqueeze_until, got {data[0].dim()}"

    while data[0].dim() != unsqueeze_until:
        data: List[Tensor] = [x.unsqueeze(0) for x in data]

    data: Tensor = torch.cat(data, dim=cat_along)

    if permute_dims is not None:
        data = data.permute(*permute_dims)

    data = data.flatten()

    return data.cpu()
