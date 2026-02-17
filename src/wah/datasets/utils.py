import hashlib
import os
import ssl

import requests
from tqdm import tqdm

__all__ = [
    "_disable_ssl_verification",
    "_download_from_url",
    "_md5_check",
]


def _disable_ssl_verification() -> None:
    ssl._create_default_https_context = ssl._create_unverified_context


def _urlretrieve(
    url: str,
    path: os.PathLike,
    chunk_size: int = 1024 * 32,
    **kwargs,
) -> None:
    response = requests.get(url, stream=True, headers={"user-agent": "yupeeee/wah"})
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(path, "wb") as fh, tqdm(total=total_size, **kwargs) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                fh.write(chunk)
                pbar.update(len(chunk))


def _download_from_url(
    url: str,
    root: os.PathLike,
) -> os.PathLike:
    fname = os.path.basename(url)
    path = os.path.join(root, fname)

    # skip download if downloaded
    if os.path.exists(path):
        print(f"{fname} already downloaded to {root}")

    # download if undownloaded
    else:
        os.makedirs(root, exist_ok=True)
        _urlretrieve(
            url=url,
            path=path,
            desc=f"Downloading {fname} to {root}",
            chunk_size=1024 * 32,  # 32KB chunks
        )

    return path


def _md5_check(
    path: os.PathLike,
    checksum: str,
    chunk_size: int = 1024 * 32,
) -> bool:
    with open(path, "rb") as f:
        h = hashlib.md5()
        while chunk := f.read(chunk_size):
            h.update(chunk)
        # print(h.hexdigest())
        return h.hexdigest() == checksum
