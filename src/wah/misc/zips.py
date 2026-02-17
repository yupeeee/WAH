import os
import tarfile
import zipfile
from typing import Optional

__all__ = [
    "extract",
]


def _set_mode(
    ext: str,
) -> str:
    modes = {
        ".zip": "r",
        ".tar": "r",
        ".gz": "r:gz",
        ".xz": "r:xz",
    }
    return modes.get(ext, "r")


def extract(
    path: os.PathLike,
    save_dir: Optional[os.PathLike] = None,
) -> None:
    assert os.path.exists(path), f"File {path} does not exist"

    # set extraction mode and target directory
    ext = os.path.splitext(path)
    save_dir = save_dir or os.path.dirname(path)
    if save_dir != os.path.dirname(path):
        os.makedirs(save_dir, exist_ok=True)

    # extract
    try:
        if ext == ".zip":
            with zipfile.ZipFile(path, "r") as f:
                f.extractall(save_dir)
        else:
            mode = _set_mode(ext)
            with tarfile.open(path, mode) as f:
                f.extractall(save_dir)
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        raise ValueError(f"Failed to extract {path}: {str(e)}")
