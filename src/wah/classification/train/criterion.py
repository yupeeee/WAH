from torch import nn

from ...misc.typing import Literal, Module

__all__ = [
    "load_criterion",
]


def load_criterion(
    train: bool,
    reduction: Literal["none", "mean", "sum"] = "mean",
    **kwargs,
) -> Module:
    if "criterion" not in kwargs:
        raise ValueError("'criterion' must be specified.")
    criterion_name = kwargs["criterion"]
    if not hasattr(nn, criterion_name):
        raise ValueError(
            f"Invalid criterion '{criterion_name}'. Must be a valid torch.nn criterion."
        )
    criterion_cfg = {"reduction": reduction}
    if "criterion_cfg" in kwargs:
        criterion_cfg.update(kwargs["criterion_cfg"])
    if train and "label_smoothing" in kwargs:
        criterion_cfg["label_smoothing"] = kwargs["label_smoothing"]
    return getattr(nn, criterion_name)(**criterion_cfg)
