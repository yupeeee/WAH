from torch.utils.data import Dataset

from ..typing import (
    Any,
    Callable,
    List,
    Optional,
    Path,
    Tuple,
)
from ..utils.download_from_url import download_url
from ..utils.zip import extract

__all__ = [
    "ClassificationDataset",
]


class ClassificationDataset(Dataset):
    """
    Dataset for classification tasks.

    ### Attributes
    - `root` (path):
      Root directory where the dataset exists or will be saved to.
    - `transform` (callable, optional):
      A function/transform that takes in the data (PIL image, numpy.ndarray, etc.) and transforms it.
      If None, no transformation is performed.
      Defaults to None.
    - `target_transform` (callable, optional):
      A function/transform that takes in the target (int, etc.) and transforms it.
      If None, no transformation is performed.
      Defaults to None.
    - `data`:
      Data of the dataset (numpy.ndarray, torch.Tensor, list of paths to data, etc.).
      Must be initialized through `self._initialize()`.
    - `targets`:
      Targets of the dataset (numpy.ndarray, torch.Tensor, list of ints, etc.).
      Must be initialized through `self._initialize()`.
      Note that targets must be in the range [0, num_classes - 1].
    - `labels`:
      Labels of the dataset (list of str labels, dict[class_idx -> label], etc.).
      Must be initialized through `self._initialize()`.

    ### Methods
    - `__getitem__`:
      Returns (data, target) of dataset using the specified index.

      Example:
      ```python
      dataset = ClassificationDataset(root="path/to/dataset")
      data, target = dataset[0]
      ```
    - `__len__`:
      Returns the size of the dataset.

      Example:
      ```python
      dataset = ClassificationDataset(root="path/to/dataset")
      num_data = len(dataset)
      ```
    """

    URL: str = ...
    ROOT: Path = ...
    MODE: str = ...

    def __init__(
        self,
        root: Path = ROOT,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        - `root` (path):
          Root directory where the dataset exists or will be saved to.
        - `transform` (callable, optional):
          A function/transform that takes in the data (PIL image, numpy.ndarray, etc.) and transforms it.
          If None, no transformation is performed.
          Defaults to None.
        - `target_transform` (callable, optional):
          A function/transform that takes in the target (int, etc.) and transforms it.
          If None, no transformation is performed.
          Defaults to None.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        # must be initialized through self._initialize()
        self.data = ...
        self.targets = ...
        self.labels = ...

        # self._initialize()

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Any, Any]:
        data, target = self.data[index], self.targets[index]

        data = self._preprocess_data(data)

        if self.transform is not None:
            data = self.transform(data)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return data, target

    def __len__(
        self,
    ) -> None:
        return len(self.data)

    def _download(
        self,
        checklist: List[Tuple[Path, str]],
    ) -> None:
        fpath = download_url(self.URL, self.root, checklist)

        if fpath != "*extracted":
            extract(fpath, mode=self.MODE)

    def _initialize(
        self,
    ) -> None:
        raise NotImplementedError

    def _preprocess_data(
        self,
        data: Any,
    ) -> Any:
        raise NotImplementedError
