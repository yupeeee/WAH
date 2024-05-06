import hashlib
import os
import ssl
from urllib.request import (
    Request,
    urlopen,
)

from tqdm import tqdm

from ..typing import (
    List,
    Path,
    Tuple,
)

__all__ = [
    "urlretrieve",
    "check",
    "download_url",
    "disable_verification",
]


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


def check(
    fpath: Path,
    checksum: str,
) -> bool:
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
    checklist: List[Tuple[Path, str]],
) -> str:
    def check_extracted_files(
        root: Path,
        checklist: List[Tuple[Path, str]],
    ) -> str:
        for ext_fname, ext_checksum in checklist[1:]:
            ext_fpath = os.fspath(os.path.join(root, ext_fname))

            if not check(ext_fpath, ext_checksum):
                e = f"Wrong file {ext_fpath}. Redownloading {fname} to {root}."
                print(e)

                raise FileNotFoundError(e)

        return "*extracted"

    fname = os.path.basename(url)
    fpath = os.fspath(os.path.join(root, fname))

    os.makedirs(root, exist_ok=True)

    # no zipped file
    if not os.path.exists(fpath):
        # zipped file removed; extracted files persist
        try:
            return check_extracted_files(root, checklist)

        # dataset not yet downloaded
        except FileNotFoundError:
            urlretrieve(
                url,
                fpath,
                desc=f"Downloading {fname} to {root}",
            )

    else:
        if not check(fpath, checklist[0][1]):
            # zipped file corrupted; extracted files persist
            try:
                return check_extracted_files(root, checklist)

            # zipped/extracted file both corrupted; redownload dataset
            except FileNotFoundError:
                urlretrieve(
                    url,
                    fpath,
                    desc=f"Redownloading {fname} to {root}",
                )

        else:
            # already extracted zipped file
            try:
                return check_extracted_files(root, checklist)

            # zipped file not extracted yet
            except FileNotFoundError:
                print("Files already downloaded and verified.")

    return fpath


def disable_verification() -> None:
    ssl._create_default_https_context = ssl._create_unverified_context
