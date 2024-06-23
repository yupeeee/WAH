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
    checklist: List[Tuple[Path, str]],
) -> str:
    """
    Downloads and verifies a file from a URL, checking its integrity against provided checksums.

    ### Parameters
    - `url` (str):
      The URL of the file to download.
    - `root` (Path):
      The directory where the file will be saved.
    - `checklist` (List[Tuple[Path, str]]):
      A list of tuples containing file paths and their corresponding checksums to verify.

    ### Returns
    - `str`:
      The path to the downloaded file or "*extracted" if the files were already verified.

    ### Notes
    - This function checks if the file already exists and verifies its integrity.
    - If the file or its extracted contents are corrupted or missing, it redownloads the file.
    - It calls `urlretrieve` to handle the file download and `check` to verify file integrity.
    """

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
