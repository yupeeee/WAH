import os
from typing import List, Tuple

from torch.utils.data import Dataset

from ....misc import parquet
from ....web import download

__all__ = [
    "PromptDataset",
]


class PromptDataset(Dataset):
    def __init__(
        self,
        root: os.PathLike,
        url: str,
        checksum: str,
    ):
        self._root = root
        self._url = url
        self._filename = os.path.basename(url)
        self._checksum = checksum

        # Download dataset
        if not os.path.exists(os.path.join(self._root, self._filename)):
            print(f"Downloading {self._filename} to {self._root}...")
            download.from_url(self._url, self._root)

        # Check checksum
        while not download.md5_check(
            path=os.path.join(self._root, self._filename),
            checksum=self._checksum,
        ):
            print("Dataset corrupted. Redownloading dataset.")
            download.from_url(self._url, self._root)

        # Load dataset
        self.data = parquet.load(
            path=os.path.join(self._root, self._filename),
        )
        self.indices = list(self.data.index)
        self.urls = list(self.data["url"])
        self.captions = list(self.data["caption"])
        self.types = list(self.data["overfit_type"])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[int, str, str, str]:
        return self.indices[idx], self.urls[idx], self.captions[idx], self.types[idx]

    def get_index(self, idx: int) -> int:
        return self.indices[idx]

    def get_url(self, index: int) -> str:
        return self.urls[self.indices.index(index)]

    def get_caption(self, index: int) -> str:
        return self.captions[self.indices.index(index)]

    def get_type(self, index: int) -> str:
        return self.types[self.indices.index(index)]

    def split_indices(self) -> Tuple[List[int], List[int], List[int], List[int]]:
        types = [self.get_type(index) for index in self.indices]

        # MATCHING VERBATIM
        MV_indices = [
            self.indices[i] for i in range(len(self.indices)) if types[i] == "MV"
        ]
        # TEMPLATE DUPLICATE
        TV_indices = [
            self.indices[i] for i in range(len(self.indices)) if types[i] == "TV"
        ]
        # RETRIEVED VERBATIM
        RV_indices = [
            self.indices[i] for i in range(len(self.indices)) if types[i] == "RV"
        ]
        # NONE
        N_indices = [
            self.indices[i] for i in range(len(self.indices)) if types[i] == "N"
        ]

        return MV_indices, TV_indices, RV_indices, N_indices
