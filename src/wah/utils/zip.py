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
    """
    Extracts the contents of a compressed file to a specified directory.

    ### Parameters
    - `fpath` (Path):
      The path to the compressed file to extract.
    - `save_dir` (Optional[Path]):
      The directory to extract the contents to. Defaults to the directory of `fpath`.
    - `mode` (Optional[str]):
      The mode to open the file. Defaults to `"r"`.

    ### Returns
    - `None`

    ### Notes
    - This function handles both `.zip` and tar-like files (e.g., `.tar`, `.tar.gz`, `.tgz`).
    - If `save_dir` is not provided, the contents are extracted to the directory containing `fpath`.
    - The function creates `save_dir` if it does not exist.
    - The function uses `zipfile.ZipFile` for `.zip` files and `tarfile.open` for tar-like files.
    """
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
