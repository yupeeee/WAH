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
    """
    Loads a YAML configuration file from the specified path.

    ### Parameters
    - `path (Path)`: The path to the YAML configuration file.

    ### Returns
    - `Config`: The configuration data loaded from the YAML file.

    ### Notes
    - This function reads the YAML file from the given path and loads its content using `yaml.load`.
    - The `Loader` used is `yaml.FullLoader` to ensure safe loading of the YAML content.
    """
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    return config
