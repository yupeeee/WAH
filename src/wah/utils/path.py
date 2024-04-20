import os
import shutil

from ..typing import (
    List,
    Optional,
    Path,
)
from .sort import sort_str_list

__all__ = [
    "mkdir",
    "rmdir",
    "join_path",
    "ext",
    "ls",
]


def mkdir(
    path: Path,
) -> None:
    os.makedirs(
        name=path,
        exist_ok=True,
    )


def rmdir(
    path: Path,
) -> None:
    """WARNING: PERMANENT DELETION"""

    shutil.rmtree(path)


def clean(
    path: Path,
) -> Path:
    return os.path.normpath(path)


def join_path(
    *path_list,
) -> Path:
    return clean(os.path.join(*path_list))


def ext(
    path: Path,
) -> str:
    return os.path.splitext(path)[-1]


def ls(
    path: Path,
    fext: Optional[str] = None,
    sort: bool = True,
) -> List[str]:
    file_list = os.listdir(path)

    if sort:
        file_list = sort_str_list(
            str_list=file_list,
            return_indices=False,
        )

    if fext in [None, "", ]:
        return file_list

    # return directories (folders) only
    elif fext == "dir":
        return [
            f for f in file_list
            if os.path.isdir(os.path.join(path, f))
        ]

    # return files w/ specified extensions only
    else:
        if "." not in fext:
            fext = "." + fext

        return [
            f for f in file_list
            if ext(f) == fext
        ]
