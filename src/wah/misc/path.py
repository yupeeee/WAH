import os
from typing import Union

__all__ = [
    "clean",
]


def clean(
    path: Union[str, os.PathLike],
) -> str:
    path = os.path.expanduser(path)
    path = os.path.normpath(path)
    return path
