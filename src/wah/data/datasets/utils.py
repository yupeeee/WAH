import hashlib
import os
import tarfile
from urllib.request import Request, urlopen
from torch.utils.data import Dataset

from tqdm import tqdm

from ...typing import (
    Any,
    Callable,
    List,
    Optional,
    Path,
    Tuple,
)

__all__ = [
    "DNTDataset",
]


def _urlretrieve(
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
            root: Path, checklist: List[Tuple[Path, str]], ) -> str:
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
            _urlretrieve(
                url, fpath,
                desc=f"Downloading {fname} to {root}",
            )

    else:
        if not check(fpath, checklist[0][1]):
            # zipped file corrupted; extracted files persist
            try:
                return check_extracted_files(root, checklist)

            # zipped/extracted file both corrupted; redownload dataset
            except FileNotFoundError:
                _urlretrieve(
                    url, fpath,
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


def extract(fpath: Path, mode: str) -> None:
    with tarfile.open(fpath, mode) as f:
        f.extractall(os.path.dirname(fpath))


class DNTDataset(Dataset):
    URL: str = ...
    ROOT: Path = ...
    MODE: str = ...

    def __init__(
            self,
            root: Path = ROOT,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # must be initialized through self.initialize()
        self.data = ...
        self.targets = ...
        self.labels = ...

        # self.initialize()

    def __getitem__(self, index: int, ) -> None:
        data, target = self.data[index], self.targets[index]

        data = self.preprocess_data(data)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(self, ) -> None:
        return len(self.data)

    def download(self, checklist: List[Tuple[Path, str]], ) -> None:
        fpath = download_url(self.URL, self.root, checklist)

        if fpath != "*extracted":
            extract(fpath, self.MODE)

    def initialize(self, ) -> None:
        raise NotImplementedError

    def preprocess_data(self, data: Any, ) -> Any:
        raise NotImplementedError
