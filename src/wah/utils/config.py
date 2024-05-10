import yaml

from ..typing import (
    Config,
    Path,
)

__all__ = [
    "load",
]


def load(
    path: Path,
) -> Config:
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
