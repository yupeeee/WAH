from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from ...typing import Config, Devices, List, LRScheduler, Optional, Path, Trainer, Tuple
from ...utils.time import time
from .config_requirements import config_requirements

__all__ = [
    "check_config",
    "get_lr",
    "get_tag",
    "init_seed",
    "load_accelerator_and_devices",
    "load_checkpoint_callback",
    "load_tensorboard_logger",
]


def check_config(
    config: Config,
) -> None:
    """
    Checks whether the provided YAML configuration file includes the necessary arguments for training.

    ### Parameters
    - `config (Config)`: YAML configuration for training.

    ### Returns
    - `None`

    ### Raises
    - `AttributeError`: If the necessary arguments are missing in the YAML configuration.
    """
    assert (
        "task" in config.keys()
    ), f"'task' must be specified in config, but is missing"

    required_args = config_requirements
    missing_args = [arg for arg in required_args if arg not in config.keys()]

    assert (
        len(missing_args) == 0
    ), f"{len(missing_args)} missing args in config: {', '.join(missing_args)}"


def get_lr(
    trainer: Trainer,
) -> float:
    """
    Returns the last computed learning rate.

    ### Parameters
    - `trainer (Trainer)`: Trainer for training.

    ### Returns
    - `float`: Last computed learning rate.
    """
    scheduler: LRScheduler = trainer.lr_scheduler_configs[0].scheduler
    lr = scheduler.get_last_lr()[0]

    return lr


def get_tag(
    trainer: Trainer,
) -> Path:
    """
    Returns the tag name used for logging.

    ### Parameters
    - `trainer (Trainer)`: Trainer for training.

    ### Returns
    - `Path`: Tag name used for logging.
    """
    tag = trainer.logger.name

    return tag


def init_seed(
    config: Config,
) -> int:
    """
    Returns the seed for random initialization based on the given YAML configuration.

    ### Parameters
    - `config (Config)`: YAML configuration for training.

    ### Returns
    - `int`: Seed for random initialization. Returns -1 (no seeding) if seed is not provided in config.
    """
    seed = -1

    if "seed" in config.keys():
        seed = config["seed"]

    return seed


def load_accelerator_and_devices(
    devices: Devices = "auto",
) -> Tuple[str, List[int]]:
    """
    Parses the device configuration string and returns the appropriate accelerator type and list of device IDs.

    ### Parameters
    - `devices (Devices)`: A string representing the devices to use, formatted as "accelerator:device_ids" (e.g., "cuda:0,1" or "gpu:0,1"). Defaults to "auto".

    ### Returns
    - `Tuple[str, List[int]]`: A tuple containing:
        - `accelerator (str)`: The type of accelerator (e.g., "gpu", "cpu", "tpu").
        - `devices (List[int])`: A list of device IDs to use.

    ### Notes
    - If the devices string contains "cuda", it is automatically converted to "gpu" for compatibility with Lightning.
    - If only the accelerator is specified (e.g., "gpu"), the devices list is set to "auto".
    - If specific devices are provided (e.g., "gpu:0,1"), they are parsed into a list of integers.
    """
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
        version=(f"{config['seed']}-{time()}" if version is None else version),
    )
