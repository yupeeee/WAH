import tarfile
import zipfile

from ..misc import path as _path
from ..misc.typing import Optional, Path

__all__ = [
    "extract",
]


def set_mode(
    ext: str,
) -> str:
    """Set the mode for opening compressed files.

    ### Args
        - `ext` (str): File extension of the compressed file

    ### Returns
        - `str`: Mode string for opening the compressed file

    ### Example
    ```python
    >>> ext = ".gz"
    >>> mode = set_mode(ext)
    >>> print(mode)
    'r:gz'
    ```
    """
    modes = {
        ".zip": "r",
        ".tar": "r",
        ".gz": "r:gz",
        ".xz": "r:xz",
    }
    return modes.get(ext, "r")


def extract(
    path: Path,
    save_dir: Optional[Path] = None,
) -> None:
    """Extract a compressed file to a directory.

    ### Args
        - `path` (Path): Path to the compressed file
        - `save_dir` (Optional[Path]): Directory to extract to. Defaults to the directory containing the compressed file.

    ### Returns
        - `None`

    ### Example
    ```python
    >>> path = "data.zip"
    >>> extract(path)
    # Extracts data.zip to the same directory

    >>> extract(path, "extracted/")
    # Extracts data.zip to extracted/
    ```
    """
    assert _path.exists(path), f"File {path} does not exist"
    # set extraction mode and save directory
    ext = _path.splitext(path)
    save_dir = save_dir or _path.dirname(path)
    if save_dir != _path.dirname(path):
        _path.mkdir(save_dir)
    # extract
    try:
        if ext == ".zip":
            with zipfile.ZipFile(path, "r") as f:
                f.extractall(save_dir)
        else:
            mode = set_mode(ext)
            with tarfile.open(path, mode) as f:
                f.extractall(save_dir)
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        raise ValueError(f"Failed to extract {path}: {str(e)}")
