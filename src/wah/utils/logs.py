import logging

__all__ = [
    "disable_lightning_logging",
]


def disable_lightning_logging() -> None:
    """
    Disables the default logging of certain Lightning components by setting their log level to WARNING.

    This function specifically adjusts the logging level for:
    - `lightning.pytorch.utilities.rank_zero`: Disables info and debug messages related to rank-zero utilities in PyTorch Lightning.
    - `lightning.pytorch.accelerators.cuda`: Suppresses info and debug logs from the CUDA accelerators in PyTorch Lightning.

    This can be useful when you want to reduce verbosity during model training or testing.
    """
    logging.getLogger("lightning.pytorch.utilities.rank_zero").setLevel(logging.WARNING)
    logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
