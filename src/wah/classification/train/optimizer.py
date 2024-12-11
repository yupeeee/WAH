from torch import optim

from ...typing import Module, Optimizer

__all__ = [
    "load_optimizer",
]


def load_optimizer(
    model: Module,
    **kwargs,
) -> Optimizer:
    assert "optimizer" in kwargs.keys()
    assert "init_lr" in kwargs.keys()

    optimizer: Optimizer = getattr(optim, kwargs.get("optimizer"))

    optimizer_cfg = {
        "params": model.parameters(),
        "lr": kwargs.get("init_lr"),
    }

    if "optimizer_cfg" in kwargs.keys():
        optimizer_cfg = {**optimizer_cfg, **kwargs.get("optimizer_cfg")}

    optimizer = optimizer(**optimizer_cfg)

    return optimizer
