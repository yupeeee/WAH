from ..typing import (
    Config,
    List,
    Metric,
)
from . import metrics as lib

__all__ = [
    "clean",
    "load_metrics",
]


def clean(
    s: str,
) -> str:
    s = "".join(filter(lambda c: str.isidentifier(c) or str.isdecimal(c), s))

    return s


requires_num_classes = [
    "Acc1",
    "Acc5",
    "ECE",
    "sECE",
]

train_only_metrics = [
    # train_only_metrics
]


def load_metrics(
    config: Config,
    train: bool,
) -> List[Metric]:
    metrics = []

    for m in config["metrics"]:
        if not train and m in train_only_metrics:
            continue

        metric_cfg = {}
        if m in requires_num_classes:
            metric_cfg["num_classes"] = config["num_classes"]

        metric = getattr(lib, m)(**metric_cfg)

        metrics.append(metric)

    return metrics
