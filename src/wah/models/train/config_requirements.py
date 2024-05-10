from ...typing import (
    List,
)

__all__ = [
    "all",
    "classification",
]

all: List[str] = [
    "task",
    "devices",
    "batch_size",
    "num_workers",
    "epochs",
    "init_lr",
    "optimizer",
    "criterion",
]

classification: List[str] = [
    "num_classes",
]
