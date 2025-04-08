from ...misc.typing import List

__all__ = [
    "check_config",
]

config_requirements: List[str] = [
    "devices",
    "num_classes",
    "batch_size",
    "num_workers",
    "epochs",
    "init_lr",
    "optimizer",
    "criterion",
]


def check_config(
    **config,
) -> None:
    missing_args = [arg for arg in config_requirements if arg not in config]
    if missing_args:
        raise ValueError(
            f"Missing required config arguments: {', '.join(missing_args)}"
        )
