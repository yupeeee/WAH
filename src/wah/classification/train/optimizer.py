from torch import optim

from ...misc.typing import Module, Optimizer

__all__ = [
    "load_optimizer",
]


def load_optimizer(
    model: Module,
    **kwargs,
) -> Optimizer:
    if "optimizer" not in kwargs:
        raise ValueError("'optimizer' must be specified.")
    if "init_lr" not in kwargs:
        raise ValueError("'init_lr' must be specified.")
    optimizer_name = kwargs["optimizer"]
    if not hasattr(optim, optimizer_name):
        raise ValueError(
            f"Invalid optimizer '{optimizer_name}'. Must be a valid torch.optim optimizer."
        )
    optimizer_cfg = {
        "params": model.parameters(),
        "lr": kwargs["init_lr"],
    }
    if "optimizer_cfg" in kwargs:
        optimizer_cfg.update(kwargs["optimizer_cfg"])
    return getattr(optim, optimizer_name)(**optimizer_cfg)
