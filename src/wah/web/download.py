import hashlib
import os
import ssl

import requests
from tqdm import tqdm

__all__ = [
    "disable_ssl_verification",
    "from_url",
    "md5_check",
]


def disable_ssl_verification() -> None:
    """Disable SSL certificate verification.

    This function disables SSL certificate verification for HTTPS requests by setting the default HTTPS context to an unverified one.
    Use with caution as this reduces security.

    ### Example
    ```python
    >>> disable_ssl_verification()
    # SSL verification is now disabled for subsequent HTTPS requests
    ```
    """
    ssl._create_default_https_context = ssl._create_unverified_context


def from_url(
    url: str,
    root: os.PathLike,
) -> os.PathLike:
    """Download a file from a URL to a local directory.

    ### Args
        - `url` (str): URL to download from
        - `root` (os.PathLike): Local directory to download to

    ### Returns
        - `os.PathLike`: Path to the downloaded file

    ### Example
    ```python
    >>> url = "https://example.com/data.txt"
    >>> root = "data/"
    >>> path = download_url(url, root)
    >>> print(path)
    'data/data.txt'
    ```
    """
    fname = os.path.basename(url)
    path = os.path.join(root, fname)
    # skip download if downloaded
    if os.path.exists(path):
        print(f"{fname} already downloaded to {root}")
    # download if undownloaded
    else:
        os.makedirs(root, exist_ok=True)
        urlretrieve(
            url=url,
            path=path,
            desc=f"Downloading {fname} to {root}",
            chunk_size=1024 * 32,  # 32KB chunks
        )
    return path


def md5_check(
    path: os.PathLike,
    checksum: str,
    chunk_size: int = 1024 * 32,
) -> bool:
    """Check if a file matches an MD5 checksum.

    ### Args
        - `path` (os.PathLike): Path to the file to check
        - `checksum` (str): Expected MD5 checksum to compare against
        - `chunk_size` (int): Size of chunks to read when computing hash. Defaults to 32KB.

    ### Returns
        - `bool`: True if the file's MD5 matches the checksum, False otherwise

    ### Example
    ```python
    >>> path = "data.txt"
    >>> checksum = "d41d8cd98f00b204e9800998ecf8427e"
    >>> md5_check(path, checksum)
    True
    ```
    """
    with open(path, "rb") as f:
        h = hashlib.md5()
        while chunk := f.read(chunk_size):
            h.update(chunk)
        # print(h.hexdigest())
        return h.hexdigest() == checksum


def urlretrieve(
    url: str,
    path: os.PathLike,
    chunk_size: int = 1024 * 32,
    **kwargs,
) -> None:
    """Download a file from a URL and save it to a local path with a progress bar.

    ### Args
        - `url` (str): URL to download from
        - `path` (os.PathLike): Local path to save the downloaded file to
        - `chunk_size` (int): Size of chunks to download at a time. Defaults to 32KB.
        - `**kwargs`: Additional arguments passed to tqdm progress bar

    ### Returns
        - `None`

    ### Example
    ```python
    >>> url = "https://example.com/file.txt"
    >>> path = "downloaded_file.txt"
    >>> urlretrieve(url, path)
    ```
    """
    response = requests.get(url, stream=True, headers={"user-agent": "yupeeee/wah"})
    response.raise_for_status()
    total_size = int(response.headers.get("content-length", 0))
    with open(path, "wb") as fh, tqdm(total=total_size, **kwargs) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                pbar.update(len(chunk))
