from torch.optim import lr_scheduler as _lr_scheduler

from ...typing import (
    Config,
    LRScheduler,
    Optimizer,
)

__all__ = [
    "load_scheduler",
]


def load_scheduler(
    config: Config,
    optimizer: Optimizer,
) -> LRScheduler:
    """
    Loads a learning rate scheduler based on the given YAML configuration.
    See https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate for supported schedulers.

    ### Parameters
    - `config (Config)`: YAML configuration for training.
    - `optimizer (Optimizer)`: Optimizer for training.

    ### Returns
    - `LRScheduler`: Learning rate scheduler for training.
    """
    if "warmup_lr_scheduler" in config.keys():
        warmup_epochs = config["warmup_lr_scheduler_cfg"]["total_iters"]
        warmup_lr_scheduler = getattr(_lr_scheduler, config["warmup_lr_scheduler"])(
            optimizer=optimizer,
            **config["warmup_lr_scheduler_cfg"],
        )
    else:
        warmup_epochs = 0
        warmup_lr_scheduler = None

    if config["lr_scheduler"] == "CosineAnnealingLR":
        config["lr_scheduler_cfg"]["T_max"] = config["epochs"] - warmup_epochs

    main_lr_scheduler: LRScheduler = getattr(_lr_scheduler, config["lr_scheduler"])(
        optimizer=optimizer,
        **config["lr_scheduler_cfg"],
    )

    if warmup_lr_scheduler is not None:
        lr_scheduler = _lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_lr_scheduler, main_lr_scheduler],
            milestones=[config["warmup_lr_scheduler_cfg"]["total_iters"]],
        )
    else:
        lr_scheduler = main_lr_scheduler

    return lr_scheduler
