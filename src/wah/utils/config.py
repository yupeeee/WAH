import yaml

from ..typing import (
    Config,
    Path,
)

__all__ = [
    "load_config",
]


def load_config(
    path: Path,
) -> Config:
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
