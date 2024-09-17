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
    ssl._create_default_https_context = ssl._create_unverified_context


def download_url(
    url: str,
    root: Path,
) -> Path:
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
    headers = {"user-agent": "yupeeee/wah"}
    request = Request(url, headers=headers)

    with urlopen(request) as response:
        with open(fpath, "wb") as fh, tqdm(total=response.length, **kwargs) as pbar:
            while chunk := response.read(chunk_size):
                fh.write(chunk)
                pbar.update(len(chunk))
