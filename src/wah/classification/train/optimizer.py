from torch import optim

from ...typing import Config, Module, Optimizer

__all__ = [
    "load_optimizer",
]


def load_optimizer(
    config: Config,
    model: Module,
) -> Optimizer:
    """
    Loads an optimizer based on the given YAML configuration.
    See https://pytorch.org/docs/stable/optim.html#algorithms for supported optimizers.

    ### Parameters
    - `config (Config)`: YAML configuration for training.
    - `model (Module)`: Model to train.

    ### Returns
    - `Optimizer`: Optimizer for training.
    """
    optimizer: Optimizer = getattr(optim, config["optimizer"])

    optimizer_cfg = {
        "params": model.parameters(),
        "lr": config["init_lr"],
    }

    if "optimizer_cfg" in config.keys():
        optimizer_cfg = {**optimizer_cfg, **config["optimizer_cfg"]}

    optimizer = optimizer(**optimizer_cfg)

    return optimizer
