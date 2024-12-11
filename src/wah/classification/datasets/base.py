from torch.utils.data import Dataset

from ... import path as _path
from ...typing import Any, Callable, List, Optional, Path, Tuple
from ...utils.download import download_url, md5_check
from ...utils.zip import extract

__all__ = [
    "ClassificationDataset",
]


class ClassificationDataset(Dataset):
    """
    Dataset for classification tasks.

    ### Attributes
    - `root` (Path): Root directory where the dataset exists or will be saved to.
    - `transform` (Callable, optional): A function/transform that takes in the data (PIL image, numpy.ndarray, etc.) and transforms it. Defaults to None.
    - `target_transform` (Callable, optional): A function/transform that takes in the target (int, etc.) and transforms it. Defaults to None.
    - `data`: Data of the dataset (numpy.ndarray, torch.Tensor, list of paths to data, etc.). Must be initialized through `self._initialize()`.
    - `targets`: Targets of the dataset (numpy.ndarray, torch.Tensor, list of ints, etc.). Must be initialized through `self._initialize()`.
    - `labels`: Labels of the dataset (list of str labels, dict[class_idx -> label], etc.). Must be initialized through `self._initialize()`.

    ### Methods
    - `__getitem__(index) -> Tuple[Any, Any]`: Returns (data, target) of dataset using the specified index.
    - `__len__() -> int`: Returns the size of the dataset.
    - `_check(checklist, dataset_root) -> bool`: Checks if the dataset exists and is valid.
    - `_download(urls, checklist, ext_dir_name) -> None`: Downloads and extracts the dataset.
    - `_initialize() -> None`: Initializes the dataset. Must be implemented by subclasses.
    - `_preprocess_data(data) -> Any`: Preprocesses the data. Must be implemented by subclasses.
    - `set_return_data_only() -> None`: Sets the flag to return only data without targets.
    - `unset_return_data_only() -> None`: Unsets the flag to return only data without targets.
    - `set_return_w_index() -> None`: Sets the flag to return data with index.
    - `unset_return_w_index() -> None`: Unsets the flag to return data with index.
    """

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        return_data_only: Optional[bool] = False,
        return_w_index: Optional[bool] = False,
    ) -> None:
        """
        Initialize the dataset.

        ### Parameters
        - `root` (Path): Root directory where the dataset exists or will be saved to.
        - `transform` (Callable, optional): A function/transform that takes in the data (PIL image, numpy.ndarray, etc.) and transforms it. Defaults to None.
        - `target_transform` (Callable, optional): A function/transform that takes in the target (int, etc.) and transforms it. Defaults to None.
        - `return_data_only` (bool, optional): Whether to return only data without targets. Defaults to False.
        - `return_w_index` (bool, optional): Whether to return data with index. Defaults to False.
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.return_data_only = return_data_only
        self.return_w_index = return_w_index

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
        - `index` (int): Index of the data item to retrieve.

        ### Returns
        - `Tuple[Any, Any]`: The data and target at the specified index.

        ### Notes
        - If `return_data_only` is set to True, only the data is returned.
        - If `return_w_index` is set to True, returns index along with (data, target) (or data if `return_data_only` is set to True).
        """
        data, target = self.data[index], self.targets[index]
        data = self._preprocess_data(data)

        if self.transform is not None:
            data = self.transform(data)

        if not self.return_data_only:
            if self.target_transform is not None:
                target = self.target_transform(target)

            if self.return_w_index:
                return index, (data, target)
            else:
                return data, target

        else:
            if self.return_w_index:
                return index, data
            else:
                return data

    def __len__(
        self,
    ) -> int:
        """
        Returns the size of the dataset.

        ### Returns
        - `int`: The number of items in the dataset.
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
        - `checklist` (List[Tuple[Path, str]]): A list of (file path, checksum) tuples to verify the dataset files.
        - `dataset_root` (Path): The root directory of the dataset.

        ### Returns
        - `bool`: True if the dataset is valid, False otherwise.
        """
        # skip check
        if checklist is None:
            return True

        else:
            for fpath, checksum in checklist:
                if not md5_check(
                    fpath=_path.join(dataset_root, fpath),
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
        - `urls` (List[str]): A list of URLs to download the dataset files from.
        - `checklist` (List[Tuple[Path, str]]): A list of (file path, checksum) tuples to verify the dataset files.
        - `ext_dir_name` (Path, optional): The directory name for extracted files. Defaults to ".".

        ### Returns
        - `None`
        """
        dataset_root = _path.join(self.root, ext_dir_name)

        # if dataset folder exists,
        if _path.exists(dataset_root):
            # check if dataset exists inside the folder
            exist = True
            for fname, _ in checklist[len(urls) :]:
                if not _path.exists(_path.join(dataset_root, fname)):
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
                _path.rmfile(fpath)
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
        - `data` (Any): The data to preprocess.

        ### Returns
        - `Any`: The preprocessed data.

        ### Notes
        - This method must be implemented by subclasses to preprocess the data.
        """
        raise NotImplementedError

    def set_return_data_only(
        self,
    ) -> None:
        """
        Sets the flag to return only data without targets.
        """
        self.return_data_only = True

    def unset_return_data_only(
        self,
    ) -> None:
        """
        Unsets the flag to return only data without targets.
        """
        self.return_data_only = False

    def set_return_w_index(
        self,
    ) -> None:
        """
        Sets the flag to return data with index.
        """
        self.return_w_index = True

    def unset_return_w_index(
        self,
    ) -> None:
        """
        Unsets the flag to return data with index.
        """
        self.return_w_index = False
