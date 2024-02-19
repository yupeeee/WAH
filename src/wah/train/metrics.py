from ..typing import (
    Config,
    List,
    Metric,
    Tuple,
)

import torchmetrics

__all__ = [
    "load_metric",
]

requires_num_classes = [
    "Accuracy",
    "CalibrationError",
]


def load_metric(
        config: Config,
) -> List[Tuple[str, Metric]]:
    metric_labels = list(config["metrics"].keys())

    metrics = []

    for label in metric_labels:
        metric = list(config["metrics"][label].keys())[0]
        metric_cfg = config["metrics"][label][metric]

        if metric in requires_num_classes:
            metric_cfg["num_classes"] = config["num_classes"]

        metric = getattr(torchmetrics, metric)(**metric_cfg)

        metrics.append((label, metric))

    return metrics
