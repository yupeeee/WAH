import hashlib
import os
import ssl
from urllib.request import (
    Request,
    urlopen,
)

from tqdm import tqdm

from ..typing import (
    Path,
)

__all__ = [
    "disable_verification",
    "urlretrieve",
    "check",
    "download_url",
]


def disable_verification() -> None:
    """
    Disables SSL certificate verification.

    ### Parameters
    - `None`

    ### Returns
    - `None`

    ### Notes
    - This function sets the default SSL context to an unverified context, disabling SSL certificate verification.
    - This is useful for downloading files from servers with self-signed certificates or other SSL issues.
    """
    ssl._create_default_https_context = ssl._create_unverified_context


def urlretrieve(
    url: str,
    fpath: Path,
    chunk_size: int = 1024 * 32,
    **kwargs,
) -> None:
    """
    Downloads a file from a URL and saves it to the specified path with a progress bar.

    ### Parameters
    - `url` (str):
      The URL of the file to download.
    - `fpath` (Path):
      The path where the downloaded file will be saved.
    - `chunk_size` (int):
      The size of each chunk to read during download.
      Defaults to 32 KB.
    - `**kwargs`:
      Additional keyword arguments to pass to `tqdm`.

    ### Returns
    - `None`

    ### Notes
    - This function uses `urlopen` to download the file in chunks, displaying a progress bar with `tqdm`.
    - A custom user-agent is used for the request headers.
    """
    headers = {"user-agent": "yupeeee/wah"}
    request = Request(url, headers=headers)

    with urlopen(request) as response:
        with open(fpath, "wb") as fh, tqdm(total=response.length, **kwargs) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))


def check(
    fpath: Path,
    checksum: str,
) -> bool:
    """
    Verifies the checksum of a file.

    ### Parameters
    - `fpath` (Path):
      The path to the file to check.
    - `checksum` (str):
      The expected MD5 checksum of the file.

    ### Returns
    - `bool`:
      `True` if the file's checksum matches the expected checksum, otherwise `False`.

    ### Notes
    - This function reads the file in binary mode and computes its MD5 checksum.
    - It compares the computed checksum with the expected checksum.
    """
    with open(fpath, "rb") as f:
        md5 = hashlib.md5(f.read()).hexdigest()
        # print(fpath, md5)

        if md5 == checksum:
            return True

        else:
            return False


def download_url(
    url: str,
    root: Path,
) -> Path:
    """
    Downloads a file from the given URL to the specified root directory.

    ### Parameters
    - `url` (str):
      The URL of the file to download.
    - `root` (Path):
      The root directory where the file will be saved.

    ### Returns
    - `Path`:
      The file path where the downloaded file is saved.

    ### Notes
    - This function checks if the file has already been downloaded to avoid redundant downloads.
    - It creates the root directory if it does not exist.
    """
    fname = os.path.basename(url)
    fpath = os.fspath(os.path.join(root, fname))

    os.makedirs(root, exist_ok=True)

    # skip download if downloaded
    if os.path.exists(fpath):
        print(f"{fname} already downloaded to {root}.")

    # download if undownloaded
    else:
        urlretrieve(
            url,
            fpath,
            desc=f"Downloading {fname} to {root}",
        )

    return fpath
