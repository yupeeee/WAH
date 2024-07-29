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


def unzip_mode(
    ext: str,
) -> str:
    """
    Determines the mode to use for extracting archives based on their file extension.

    ### Parameters
    - `ext` (str): The file extension of the archive.

    ### Returns
    - `str`: The mode to use for extracting the archive.

    ### Notes
    - Supports `.zip`, `.tar`, `.gz`, and `.xz` extensions.
    - Returns `"r"` for unknown extensions.
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
    Extracts the contents of an archive to the specified directory.

    ### Parameters
    - `fpath` (Path): The path to the archive file.
    - `save_dir` (Path, optional): The directory where the contents will be extracted. Defaults to the directory of the archive file.

    ### Returns
    - `None`

    ### Notes
    - The function determines the extraction mode based on the file extension.
    - Supports `.zip`, `.tar`, `.gz`, and `.xz` archives.
    - Creates the save directory if it does not exist.
    """
    ext = os.path.splitext(fpath)[-1]
    mode = unzip_mode(ext)

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
