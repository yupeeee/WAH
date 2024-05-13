from ...typing import (
    Config,
    LRScheduler,
    Path,
    Trainer,
)
from . import config_requirements

__all__ = [
    "check_config",
    "init_seed",
    "get_lr",
    "get_tag",
]


def check_config(
    config: Config,
) -> None:
    """
    Checks whether the provided YAML configuration file includes the necessary arguments for training.

    Parameters:
    - config (Config): YAML configuration for training.

    Returns:
    - None

    Raises:
    - AttributeError: If the necessary arguments is missing in the YAML configuration.
    """
    assert (
        "task" in config.keys()
    ), f"'task' must be specified in config, but is missing"

    required_args = config_requirements.all
    required_args += getattr(config_requirements, config["task"])

    missing_args = [arg for arg in required_args if arg not in config.keys()]

    assert (
        len(missing_args) == 0
    ), f"{len(missing_args)} missing args in config: {', '.join(missing_args)}"


def init_seed(
    config: Config,
) -> int:
    """
    Returns the seed for random initialization based on the given YAML configuration.

    Parameters:
    - config (Config): YAML configuration for training.

    Returns:
    - int: Seed for random initialization.
    Returns -1 (no seeding) if seed is not provided in config.
    """
    seed = -1

    if "seed" in config.keys():
        seed = config["seed"]

    return seed


def get_lr(
    trainer: Trainer,
) -> float:
    """
    Returns the last computed learning rate.

    Parameters:
    - trainer (Trainer): Trainer for training.

    Returns:
    - float: Last computed learning rate.
    """
    scheduler: LRScheduler = trainer.lr_scheduler_configs[0].scheduler
    lr = scheduler.get_last_lr()[0]

    return lr


def get_tag(
    trainer: Trainer,
) -> Path:
    tag = trainer.logger.name

    return tag
