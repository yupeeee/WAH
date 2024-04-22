from torch.utils.data import Dataset

from ...typing import (
    Any,
    Callable,
    List,
    Optional,
    Path,
    Tuple,
)
from ...utils.download_from_url import download_url
from ...utils.zip import extract

__all__ = [
    "DNTDataset",
]


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

    def __getitem__(
        self,
        index: int,
    ) -> None:
        data, target = self.data[index], self.targets[index]

        data = self.preprocess_data(data)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(
        self,
    ) -> None:
        return len(self.data)

    def download(
        self,
        checklist: List[Tuple[Path, str]],
    ) -> None:
        fpath = download_url(self.URL, self.root, checklist)

        if fpath != "*extracted":
            extract(fpath, self.MODE)

    def initialize(
        self,
    ) -> None:
        raise NotImplementedError

    def preprocess_data(
        self,
        data: Any,
    ) -> Any:
        raise NotImplementedError
