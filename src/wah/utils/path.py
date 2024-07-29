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
    "join",
    "ext",
    "ls",
]


def mkdir(
    path: Path,
) -> None:
    """
    Creates a directory at the specified path if it does not already exist.

    ### Parameters
    - `path (Path)`: The path to the directory to create.

    ### Returns
    - `None`

    ### Notes
    - This function uses `os.makedirs` with `exist_ok=True` to ensure the directory is created if it does not exist.
    """
    os.makedirs(
        name=path,
        exist_ok=True,
    )


def rmdir(
    path: Path,
) -> None:
    """
    ## WARNING
    This function permanently deletes the directory and all its contents.
    Use with caution.

    Recursively deletes a directory and all its contents.

    ### Parameters
    - `path (Path)`: The path to the directory to delete.

    ### Returns
    - `None`
    """
    shutil.rmtree(path)


def clean(
    path: Path,
) -> Path:
    """
    Normalizes a filesystem path.

    ### Parameters
    - `path (Path)`: The path to normalize.

    ### Returns
    - `Path`: The normalized path.

    ### Notes
    - This function uses `os.path.normpath` to normalize the given path.
    """
    return os.path.normpath(path)


def join(
    *path_list,
) -> Path:
    """
    Joins multiple path components and normalizes the resulting path.

    ### Parameters
    - `*path_list`: The path components to join.

    ### Returns
    - `Path`: The joined and normalized path.

    ### Notes
    - This function uses `os.path.join` to join the given path components and then normalizes the result using `clean`.
    """
    return clean(os.path.join(*path_list))


def splitext(
    path: Path,
) -> str:
    """
    Returns the file extension of the given path.

    ### Parameters
    - `path (Path)`: The path to extract the file extension from.

    ### Returns
    - `str`: The file extension of the path.

    ### Notes
    - This function uses `os.path.splitext` to get the file extension.
    """
    return os.path.splitext(path)[-1]


def ls(
    path: Path,
    fext: Optional[str] = None,
    sort: bool = True,
) -> List[str]:
    """
    Lists files and directories in the specified path, optionally filtering by extension and sorting.

    ### Parameters
    - `path (Path)`: The path to list the files and directories from.
    - `fext (Optional[str])`: The file extension to filter by. Defaults to `None`.
    - `sort (bool)`: Whether to sort the list. Defaults to `True`.

    ### Returns
    - `List[str]`: A list of file and directory names in the specified path.

    ### Notes
    - This function uses `os.listdir` to get the list of files and directories.
    - It can filter the results by file extension and sort the list if specified.
    - If `fext` is `"dir"`, it returns only directories.
    """
    file_list = os.listdir(path)

    if sort:
        file_list = sort_str_list(
            str_list=file_list,
            return_indices=False,
        )

    if fext in [
        None,
        "",
    ]:
        return file_list

    # return directories (folders) only
    elif fext == "dir":
        return [f for f in file_list if os.path.isdir(os.path.join(path, f))]

    # return files w/ specified extensions only
    else:
        if "." not in fext:
            fext = "." + fext

        return [f for f in file_list if splitext(f) == fext]
