import os

from torch.utils.data import Dataset

from ..typing import (
    Any,
    Callable,
    List,
    Optional,
    Path,
    Tuple,
)
from ..utils.download import check, download_url
from ..utils.zip import extract

__all__ = [
    "ClassificationDataset",
]


class ClassificationDataset(Dataset):
    """
    Dataset for classification tasks.

    ### Attributes
    - `root` (Path):
      Root directory where the dataset exists or will be saved to.
    - `transform` (Callable, optional):
      A function/transform that takes in the data (PIL image, numpy.ndarray, etc.) and transforms it.
      If None, no transformation is performed. Defaults to None.
    - `target_transform` (Callable, optional):
      A function/transform that takes in the target (int, etc.) and transforms it.
      If None, no transformation is performed. Defaults to None.
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
    - `__len__`:
      Returns the size of the dataset.
    - `_check`:
      Checks if the dataset exists and is valid.
    - `_download`:
      Downloads and extracts the dataset.
    - `_initialize`:
      Initializes the dataset. Must be implemented by subclasses.
    - `_preprocess_data`:
      Preprocesses the data. Must be implemented by subclasses.
    - `set_return_data_only`:
      Sets the flag to return only data without targets.
    - `unset_return_data_only`:
      Unsets the flag to return only data without targets.

    ### Example
    ```python
    dataset = ClassificationDataset(root="path/to/dataset")
    data, target = dataset[0]
    num_data = len(dataset)
    ```
    """

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_data_only: Optional[bool] = False,
    ) -> None:
        """
        - `root` (Path):
          Root directory where the dataset exists or will be saved to.
        - `transform` (Callable, optional):
          A function/transform that takes in the data (PIL image, numpy.ndarray, etc.) and transforms it.
          If None, no transformation is performed.
          Defaults to None.
        - `target_transform` (Callable, optional):
          A function/transform that takes in the target (int, etc.) and transforms it.
          If None, no transformation is performed.
          Defaults to None.
        - `return_data_only` (bool, optional):
          Whether to return only data without targets.
          Defaults to False.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.return_data_only = return_data_only

        # must be initialized through self._initialize()
        self.data = ...
        self.targets = ...
        self.labels = ...

        # self._initialize()

    def __getitem__(
        self,
        index: int,
    ) -> Tuple[Any, Any]:
        """
        Returns (data, target) of dataset using the specified index.

        ### Parameters
        - `index` (int):
          Index of the data item to retrieve.

        ### Returns
        - `Tuple[Any, Any]`:
          The data and target at the specified index.

        ### Notes
        - If `return_data_only` is set to True, only the data is returned.
        """
        data, target = self.data[index], self.targets[index]

        data = self._preprocess_data(data)

        if self.transform is not None:
            data = self.transform(data)

        if not self.return_data_only:
            if self.target_transform is not None:
                target = self.target_transform(target)

            return data, target

        else:
            return data

    def __len__(
        self,
    ) -> int:
        """
        Returns the size of the dataset.

        ### Returns
        - `int`:
          The number of items in the dataset.
        """
        return len(self.data)

    def _check(
        self,
        checklist: List[Tuple[Path, str]],
        dataset_root: Path,
    ) -> bool:
        """
        Checks if the dataset exists and is valid.

        ### Parameters
        - `checklist` (List[Tuple[Path, str]]):
          A list of (file path, checksum) tuples to verify the dataset files.
        - `dataset_root` (Path):
          The root directory of the dataset.

        ### Returns
        - `bool`:
          True if the dataset is valid, False otherwise.

        ### Notes
        - This method verifies the existence and integrity of the dataset files.
        """
        # skip check
        if checklist is None:
            return True

        else:
            for fpath, checksum in checklist:
                if not check(
                    fpath=os.path.join(dataset_root, fpath),
                    checksum=checksum,
                ):
                    return False

            return True

    def _download(
        self,
        urls: List[str],
        checklist: List[Tuple[Path, str]],
        ext_dir_name: Optional[Path] = ".",
    ) -> None:
        """
        Downloads and extracts the dataset.

        ### Parameters
        - `urls` (List[str]):
          A list of URLs to download the dataset files from.
        - `checklist` (List[Tuple[Path, str]]):
          A list of (file path, checksum) tuples to verify the dataset files.
        - `ext_dir_name` (Path, optional):
          The directory name for extracted files. Defaults to ".".

        ### Returns
        - `None`

        ### Notes
        - This method downloads the dataset files, verifies their integrity, and extracts them to the specified directory.
        """
        dataset_root = os.path.normpath(os.path.join(self.root, ext_dir_name))

        # if dataset folder exists,
        if os.path.exists(dataset_root):
            # check if dataset exists inside the folder
            exist = True
            for fname, _ in checklist[len(urls) :]:
                if not os.path.exists(os.path.join(dataset_root, fname)):
                    exist = False
                    break
        # otherwise dataset does not exist
        else:
            exist = False

        # return if dataset exists and is verified
        if exist and self._check(checklist[len(urls) :], dataset_root):
            return

        # else, download dataset
        fpaths = []
        for url in urls:
            fpath = download_url(url, self.root)
            fpaths.append(fpath)

        # download again if download was unsuccessful
        while not self._check(checklist[: len(urls)], self.root):
            print("Dataset corrupted. Redownloading dataset.")
            # first, delete files
            for fpath in fpaths:
                os.remove(fpath)
            # then, download again
            fpaths = []
            for url in urls:
                fpath = download_url(url, self.root)
                fpaths.append(fpath)

        # unzip dataset
        for fpath in fpaths:
            extract(fpath, dataset_root)

        # check downloaded file
        assert self._check(checklist[len(urls) :], dataset_root)

    def _initialize(
        self,
    ) -> None:
        """
        Initializes the dataset.

        ### Returns
        - `None`

        ### Notes
        - This method must be implemented by subclasses to initialize the dataset.
        """
        raise NotImplementedError

    def _preprocess_data(
        self,
        data: Any,
    ) -> Any:
        """
        Preprocesses the data.

        ### Parameters
        - `data` (Any):
          The data to preprocess.

        ### Returns
        - `Any`:
          The preprocessed data.

        ### Notes
        - This method must be implemented by subclasses to preprocess the data.
        """
        raise NotImplementedError

    def set_return_data_only(
        self,
    ) -> None:
        """
        Sets the flag to return only data without targets.

        ### Returns
        - `None`
        """
        self.return_data_only = True

    def unset_return_data_only(
        self,
    ) -> None:
        """
        Unsets the flag to return only data without targets.

        ### Returns
        - `None`
        """
        self.return_data_only = False
