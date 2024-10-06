import os
import shutil

from .typing import List, Optional, Path, Tuple
from .utils.lst import sort_str_list

__all__ = [
    "basename",
    "clean",
    "dirname",
    "exists",
    "isdir",
    "join",
    "ls",
    "mkdir",
    "rmdir",
    "rmfile",
    "split",
    "splitext",
]


def basename(
    path: Path,
) -> str:
    """
    Returns the base name of the file or directory from a given path.

    ### Parameters
    - `path` (Path): The path to extract the base name from.

    ### Returns
    - `str`: The base name of the file or directory.
    """
    return os.path.basename(clean(path))


def clean(
    path: Path,
) -> Path:
    """
    Cleans a file path by normalizing it, ensuring consistency in path separators.

    ### Parameters
    - `path` (Path): The file path to clean.

    ### Returns
    - `Path`: The cleaned file path.
    """
    return os.path.normpath(path)


def dirname(
    path: Path,
) -> Path:
    """
    Returns the directory name from a given file path.

    ### Parameters
    - `path` (Path): The file path to extract the directory name from.

    ### Returns
    - `Path`: The directory name.
    """
    return os.path.dirname(clean(path))


def exists(
    path: Path,
) -> bool:
    """
    Checks if a given file or directory path exists.

    ### Parameters
    - `path` (Path): The path to check.

    ### Returns
    - `bool`: `True` if the path exists, `False` otherwise.
    """
    return os.path.exists(clean(path))


def isdir(
    path: Path,
) -> bool:
    """
    Checks if the given path is a directory.

    ### Parameters
    - `path` (Path): The path to check.

    ### Returns
    - `bool`: `True` if the path is a directory, `False` otherwise.
    """
    return os.path.isdir(clean(path))


def join(
    *path_list,
) -> Path:
    """
    Joins multiple paths into a single path and normalizes it.

    ### Parameters
    - `*path_list`: Paths to join.

    ### Returns
    - `Path`: The joined and cleaned path.
    """
    return clean(os.path.join(*path_list))


def ls(
    path: Path,
    fext: Optional[str] = None,
    sort: bool = True,
    absolute: Optional[bool] = False,
) -> List[str]:
    """
    Lists the contents of a directory with optional filtering and sorting.

    ### Parameters
    - `path` (Path): The directory to list contents from.
    - `fext` (Optional[str], optional): Filter files by extension or directories by `"dir"`. Defaults to `None`.
    - `sort` (bool, optional): Sorts the list if `True`. Defaults to `True`.
    - `absolute` (Optional[bool], optional): Returns absolute paths if `True`. Defaults to `False`.

    ### Returns
    - `List[str]`: A list of files or directories matching the criteria.
    """
    file_list = os.listdir(clean(path))

    if sort:
        file_list = sort_str_list(
            str_list=file_list,
            return_indices=False,
        )

    # no extension specifications
    if fext in [
        None,
        "",
    ]:
        file_list = file_list

    # return directories (folders) only (fext="dir")
    elif fext == "dir":
        file_list = [f for f in file_list if isdir(join(path, f))]

    # return files w/ specified extensions only
    else:
        if "." not in fext:
            fext = "." + fext

        file_list = [f for f in file_list if splitext(f) == fext]

    # make as absolute path if necessary
    if absolute:
        file_list = [join(path, f) for f in file_list]

    return file_list


def mkdir(
    path: Path,
) -> None:
    """
    Creates a directory at the specified path, including intermediate directories.

    ### Parameters
    - `path` (Path): The directory path to create.
    """
    os.makedirs(
        name=clean(path),
        exist_ok=True,
    )


def rmdir(
    path: Path,
) -> None:
    """
    Removes a directory and all its contents.

    ### Parameters
    - `path` (Path): The directory path to remove.
    """
    shutil.rmtree(clean(path))


def rmfile(
    fpath: Path,
) -> None:
    """
    Removes a file at the specified path.

    ### Parameters
    - `fpath` (Path): The file path to remove.
    """
    os.remove(clean(fpath))


def split(
    path: Path,
) -> Tuple[Path, str]:
    """
    Splits a path into its directory and base name.

    ### Parameters
    - `path` (Path): The file path to split.

    ### Returns
    - `Tuple[Path, str]`: A tuple where the first element is the directory and the second is the base name.
    """
    return os.path.split(clean(path))


def splitext(
    path: Path,
) -> str:
    """
    Splits a path into its root and extension.

    ### Parameters
    - `path` (Path): The file path to split.

    ### Returns
    - `str`: The file extension.
    """
    return os.path.splitext(clean(path))[-1]
