from torch import nn

from ...typing import Literal, Module

__all__ = [
    "load_criterion",
]


def load_criterion(
    train: bool,
    reduction: Literal["none", "mean", "sum"] = "mean",
    **kwargs,
) -> Module:
    assert "criterion" in kwargs.keys()

    criterion: Module = getattr(nn, kwargs["criterion"])

    criterion_cfg = {
        "reduction": reduction,
    }
    if train and "label_smoothing" in kwargs.keys():
        criterion_cfg["label_smoothing"] = kwargs["label_smoothing"]

    criterion = criterion(**criterion_cfg)

    return criterion
