from torch import nn

from ...typing import Config, Module, Optional

__all__ = [
    "load_criterion",
]


def load_criterion(
    config: Config,
    train: Optional[bool] = True,
) -> Module:
    """
    Loads a criterion (loss function) based on the given YAML configuration.
    See https://pytorch.org/docs/stable/nn.html#loss-functions for supported criterions.

    ### Parameters
    - `config (Config)`: YAML configuration for training.

    ### Returns
    - `Module`: Criterion (loss function) for training.
    """
    criterion: Module = getattr(nn, config["criterion"])

    criterion_cfg = {}
    if "criterion_cfg" in config.keys():
        criterion_cfg = config["criterion_cfg"]

        if train is False and "label_smoothing" in criterion_cfg.keys():
            criterion_cfg["label_smoothing"] = 0.0

    criterion = criterion(**criterion_cfg)

    return criterion
