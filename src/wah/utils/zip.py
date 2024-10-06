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
    """
    Sets the extraction mode based on the file extension.

    ### Parameters
    - `ext` (str): The file extension to determine the mode.

    ### Returns
    - `str`: The extraction mode.
    """
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
    """
    Extracts a compressed file (zip, tar, gz, xz) to a specified directory.

    ### Parameters
    - `fpath` (Path): The path to the compressed file.
    - `save_dir` (Optional[Path], optional): The directory to extract files to.
    If not provided, files are extracted to the directory where the compressed file is located.
    Defaults to `None`.

    ### Notes
    - The extraction mode is set based on the file extension. Supported formats include `.zip`, `.tar`, `.gz`, and `.xz`.
    """
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
