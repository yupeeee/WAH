from torchmetrics import Accuracy

from ...typing import Config, List, Metric, Tuple
from .torchmetrics_custom import MulticlassCalibrationError

__all__ = [
    "as_attr",
    "MetricLoader",
    "load_metrics",
]


def as_attr(
    name: str,
) -> str:
    """
    Converts a given string into a valid Python attribute name by filtering out characters
    that are not suitable for an identifier or numeric, and prefixes it with an underscore
    (ensuring it can safely be used as a private attribute name).

    ### Parameters
    - `name (str)`: Input string to be converted.

    ### Returns
    - `str`: A string suitable to be used as a private attribute name, prefixed with an underscore.
    """
    attr: str = "".join(filter(lambda c: str.isidentifier(c) or str.isdecimal(c), name))

    attr = "_" + attr

    return attr


class MetricLoader:
    def __init__(
        self,
        config: Config,
    ) -> None:
        """
        Initializes the MetricLoader with the given configuration.

        ### Parameters
        - `config (Config)`: The configuration object containing settings for metrics.
        """
        self.config = config

    def __call__(
        self,
        name: str,
    ) -> Metric:
        """
        Loads a metric based on the given name.

        ### Parameters
        - `name (str)`: The name of the metric to load.
          Supported metrics are:
            - "acc@`int`"
            - "ce@`l1 | sign | l2 | max`"

        ### Returns
        - `Metric`: The metric instance to monitor in training.

        ### Raises
        - `ValueError`: If the metric name is unsupported.
        """
        if "acc@" in name:
            assert self.config["task"] == "classification"

            return Accuracy(
                task="multiclass",
                num_classes=self.config["num_classes"],
                top_k=int(name.split("@")[-1]),
            )

        elif "ce@" in name:
            assert self.config["task"] == "classification"

            norm = name.split("@")[-1]

            return MulticlassCalibrationError(
                num_classes=self.config["num_classes"],
                n_bins=10,
                norm=norm,
            )

        else:
            raise ValueError(f"Unsupported metric: {name}")


def load_metrics(
    config: Config,
) -> List[Tuple[str, Metric]]:
    """
    Loads a list of evaluation metrics based on the given YAML configuration.
    The function uses a `MetricLoader` instance to dynamically fetch each specified metric
    and stores them in a list along with their names.

    ### Parameters
    - `config (Config)`: YAML configuration for training.

    ### Returns
    - `List[Tuple[str, Metric]]`: A list of tuples, where each tuple contains the metric name (as a string) and an initialized metric instance.
    """
    metrics: List[Tuple[str, Metric]] = []

    if (
        "metrics" not in config.keys()
        or config["metrics"] == "None"
        or config["metrics"][0] == "None"
    ):
        return metrics

    metric_loader = MetricLoader(config)

    for metric_name in config["metrics"]:
        metric = metric_loader(metric_name)
        metrics.append((metric_name, metric))

    return metrics
