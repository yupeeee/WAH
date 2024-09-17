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
    return os.path.basename(clean(path))


def clean(
    path: Path,
) -> Path:
    return os.path.normpath(path)


def dirname(
    path: Path,
) -> Path:
    return os.path.dirname(clean(path))


def exists(
    path: Path,
) -> bool:
    return os.path.exists(clean(path))


def isdir(
    path: Path,
) -> bool:
    return os.path.isdir(clean(path))


def join(
    *path_list,
) -> Path:
    return clean(os.path.join(*path_list))


def ls(
    path: Path,
    fext: Optional[str] = None,
    sort: bool = True,
    absolute: Optional[bool] = False,
) -> List[str]:
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
    os.makedirs(
        name=clean(path),
        exist_ok=True,
    )


def rmdir(
    path: Path,
) -> None:
    shutil.rmtree(clean(path))


def rmfile(
    fpath: Path,
) -> None:
    os.remove(clean(fpath))


def split(
    path: Path,
) -> Tuple[Path, str]:
    return os.path.split(clean(path))


def splitext(
    path: Path,
) -> str:
    return os.path.splitext(clean(path))[-1]
