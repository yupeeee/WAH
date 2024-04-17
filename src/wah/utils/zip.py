import os
import tarfile
import zipfile

from ..typing import (
    Optional,
    Path,
)

__all__ = [
    "extract",
]


def extract(
    fpath: Path,
    save_dir: Optional[Path] = None,
    mode: Optional[str] = "r",
) -> None:
    ext = os.path.splitext(fpath)[-1]

    if save_dir is None:
        save_dir = os.path.dirname(fpath)

    else:
        os.makedirs(save_dir, exist_ok=True)

    if ext == ".zip":
        with zipfile.ZipFile(fpath, mode) as f:
            f.extractall(save_dir)

    else:
        with tarfile.open(fpath, mode) as f:
            f.extractall(save_dir)
