import hashlib
import ssl
from urllib.request import Request, urlopen

from tqdm import tqdm

from .. import path as _path
from ..typing import Path

__all__ = [
    "disable_ssl_verification",
    "download_url",
    "md5_check",
]


def disable_ssl_verification() -> None:
    """
    Disables SSL certificate verification for HTTPS requests.

    ### Notes
    - Modifies the default SSL context to ignore certificate verification.
    """
    ssl._create_default_https_context = ssl._create_unverified_context


def download_url(
    url: str,
    root: Path,
) -> Path:
    """
    Downloads a file from a given URL to a specified directory.

    ### Parameters
    - `url` (str): URL of the file to download.
    - `root` (Path): Directory to save the downloaded file.

    ### Returns
    - `Path`: Path to the downloaded file.

    ### Notes
    - If the file already exists at the destination, the download is skipped.
    """
    fname = _path.basename(url)
    fpath = _path.join(root, fname)

    _path.mkdir(root)

    # skip download if downloaded
    if _path.exists(fpath):
        print(f"{fname} already downloaded to {root}.")

    # download if undownloaded
    else:
        urlretrieve(
            url,
            fpath,
            desc=f"Downloading {fname} to {root}",
        )

    return fpath


def md5_check(
    fpath: Path,
    checksum: str,
    chunk_size: int = 1024 * 32,
) -> bool:
    """
    Verifies the MD5 checksum of a file.

    ### Parameters
    - `fpath` (Path): Path to the file to check.
    - `checksum` (str): Expected MD5 checksum to verify against.
    - `chunk_size` (int, optional): Size of the chunks to read the file in bytes. Defaults to 32KB.

    ### Returns
    - `bool`: `True` if the checksum matches, `False` otherwise.
    """
    with open(fpath, "rb") as f:
        h = hashlib.md5()

        while chunk := f.read(chunk_size):
            h.update(chunk)

        md5 = h.hexdigest()
        # print(fpath, md5)

        if md5 == checksum:
            return True

        else:
            return False


def urlretrieve(
    url: str,
    fpath: Path,
    chunk_size: int = 1024 * 32,
    **kwargs,
) -> None:
    """
    Retrieves a file from a URL and writes it to the specified file path.

    ### Parameters
    - `url` (str): URL of the file to retrieve.
    - `fpath` (Path): Path to save the downloaded file.
    - `chunk_size` (int, optional): Size of the chunks to read the file in bytes. Defaults to 32KB.
    - `**kwargs`: Additional keyword arguments for `tqdm` progress bar.

    ### Notes
    - Uses the tqdm package to display a progress bar during the download.
    """
    headers = {"user-agent": "yupeeee/wah"}
    request = Request(url, headers=headers)

    with urlopen(request) as response:
        with open(fpath, "wb") as fh, tqdm(total=response.length, **kwargs) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))
