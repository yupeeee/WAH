import os
from typing import Any, Callable, List, Optional, Tuple

from torch.utils.data import Dataset

from ...misc import path as _path
from ...misc import zips as _zips
from ..utils import _download_from_url, _md5_check

__all__ = [
    "DetectionDataset",
]


class DetectionDataset(Dataset):
    """Base class for detection datasets.

    ### Args
        - `root` (os.PathLike): Root directory containing the dataset
        - `transform` (Optional[Callable]): Optional transform to be applied to the data
        - `target_transform` (Optional[Callable]): Optional transform to be applied to the targets

    ### Attributes
        - `root` (os.PathLike): Root directory containing the dataset
        - `transform` (Optional[Callable]): Transform applied to the data
        - `target_transform` (Optional[Callable]): Transform applied to the targets
        - `name` (str): Name of the dataset
        - `data` (Any): Dataset samples, initialized in _initialize()
        - `targets` (Any): Dataset targets, initialized in _initialize()
        - `labels` (List[str]): List of class labels, initialized in _initialize()

    ### Example
    ```python
    >>> dataset = ClassificationDataset("path/to/data")
    >>> len(dataset)  # Get dataset size
    1000
    >>> data, target = dataset[0]  # Get first sample and target
    ```
    """

    def __init__(
        self,
        root: os.PathLike,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        - `root` (os.PathLike): Root directory containing the dataset
        - `transform` (Optional[Callable]): Optional transform to be applied to the data
        - `target_transform` (Optional[Callable]): Optional transform to be applied to the targets
        """
        self.root = _path.clean(root)
        self.transform = transform
        self.target_transform = target_transform

        self._name: str

        # Must be initialized through self._initialize()
        self.images = ...
        self.annotations = ...
        self.categories = ...

    @property
    def name(self) -> str:
        return self._name

    def __str__(self) -> str:
        """Return the name of the dataset.

        ### Returns
            - `str`: Name of the dataset.
        """
        return self.name

    def __getitem__(self, index: int) -> Any:
        """Get a sample from the dataset.

        ### Args
            - `index` (int): Index of the sample to get.

        ### Returns
            - `Tuple[Any, Any]`: Tuple containing (data, target) where:
                - data: The processed input data.
                - target: The corresponding target label.

        ### Example
        ```python
        >>> dataset = ClassificationDataset("path/to/data")
        >>> data, target = dataset[0]  # Get first sample
        ```
        """
        image = self.images[index]
        image = self._preprocess_data(image)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self) -> int:
        """Return the length of the dataset.

        ### Returns
            - `int`: Number of samples in the dataset.
        """
        return len(self.images)

    def _initialize(self) -> None:
        """Initialize dataset.

        This method must be implemented to initialize the dataset attributes:
            - `self.images`: List/array of images or image paths.
            - `self.annotations`: List/array of annotations.
            - `self.categories`: List/array of category names.

        ### Returns
            - `None`
        """
        raise NotImplementedError

    def _preprocess_data(self, data: Any) -> Any:
        """Preprocess data before applying transforms.

        This method must be implemented to preprocess the data.

        ### Args
            - `data` (Any): Data to preprocess.

        ### Returns
            - `Any`: Preprocessed data.
        """
        raise NotImplementedError

    def _check(
        self,
        checklist: List[Tuple[os.PathLike, str]],
        root: os.PathLike,
    ) -> bool:
        """Check if all files in checklist exist and have correct MD5 checksums.

        ### Args
            - `checklist` (List[Tuple[os.PathLike, str]]): List of (filepath, checksum) tuples to verify.
            - `root` (os.PathLike): Root directory containing the files.

        ### Returns
            - `bool`: True if all files exist and have correct checksums, False otherwise.

        ### Example
        ```python
        >>> checklist = [('data.txt', 'abc123'), ('labels.txt', 'def456')]
        >>> _check(checklist, './data')
        True
        ```
        """
        if not checklist:
            return True
        return all(
            _md5_check(
                path=os.path.join(root, path),
                checksum=checksum,
            )
            for path, checksum in checklist
        )

    def _download(
        self,
        urls: List[str],
        checklist: List[Tuple[os.PathLike, str]],
        extract_dir: Optional[os.PathLike] = "",
    ) -> None:
        """Download dataset files from URLs and verify their integrity.

        ### Args
            - `urls` (List[str]): List of URLs to download files from.
            - `checklist` (List[Tuple[os.PathLike, str]]): List of (filepath, checksum) tuples to verify.
                Should include both downloaded and extracted files.
            - `extract_dir` (Optional[os.PathLike]): Directory to extract downloaded files to, relative to self.root.
                Defaults to "". The extracted data will be saved to {self.root}/{extract_dir}/.

        ### Returns
            - `None`

        ### Example
        ```python
        >>> urls = ['http://example.com/data.zip']
        >>> checklist = [
        ...     ('data.zip', 'abc123'),          # Downloaded file
        ... ]
        >>> _download(urls, checklist, 'extracted')  # Files will be extracted to ./extracted/
        ```
        """
        dataset_root = os.path.join(self.root, extract_dir)

        # Download and verify files
        while True:
            # Download files
            paths = [_download_from_url(url, self.root) for url in urls]
            # Verify downloads
            if self._check(checklist, self.root):
                break
            # If verification failed, clean up and retry
            print("Dataset corrupted. Redownloading dataset.")
            for path in paths:
                os.remove(path)

        # Extract downloads
        for path in paths:
            fname = os.path.basename(path)
            try:
                print(f"Extracting {fname} to {dataset_root}")
                _zips.extract(path, dataset_root)
            except:
                print(f"Error extracting {fname} to {dataset_root}")
                pass
