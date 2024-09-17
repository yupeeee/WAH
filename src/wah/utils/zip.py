import tarfile
import zipfile

from .. import path as _path
from ..typing import Optional, Path

__all__ = [
    "extract",
]


def set_mode(
    ext: str,
) -> str:
    if ext == ".zip":
        return "r"
    elif ext == ".tar":
        return "r"
    elif ext == ".gz":
        return "r:gz"
    elif ext == ".xz":
        return "r:xz"
    else:
        return "r"


def extract(
    fpath: Path,
    save_dir: Optional[Path] = None,
) -> None:
    # set extraction mode
    ext = _path.splitext(fpath)[-1]
    mode = set_mode(ext)

    # set/make save directory
    if save_dir is None:
        save_dir = _path.dirname(fpath)
    else:
        _path.mkdir(save_dir)

    # extract
    if ext == ".zip":
        with zipfile.ZipFile(fpath, mode) as f:
            f.extractall(save_dir)
    else:
        with tarfile.open(fpath, mode) as f:
            f.extractall(save_dir)
