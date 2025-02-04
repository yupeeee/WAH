from torch.utils.data import Dataset

from ... import utils
from ...misc import path as _path
from ...misc.typing import Any, Callable, List, Optional, Path, Tuple

__all__ = [
    "ClassificationDataset",
]


class ClassificationDataset(Dataset):
    """Base class for classification datasets.

    ### Args
        - `root` (Path): Root directory containing the dataset
        - `transform` (Optional[Callable]): Optional transform to be applied to the data
        - `target_transform` (Optional[Callable]): Optional transform to be applied to the targets

    ### Attributes
        - `root` (Path): Root directory containing the dataset
        - `transform` (Optional[Callable]): Transform applied to the data
        - `target_transform` (Optional[Callable]): Transform applied to the targets
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
        root: Path,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        - `root` (Path): Root directory containing the dataset
        - `transform` (Optional[Callable]): Optional transform to be applied to the data
        - `target_transform` (Optional[Callable]): Optional transform to be applied to the targets
        """
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        # Must be initialized through self._initialize()
        self.data = ...
        self.targets = ...
        self.labels = ...

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """Get a sample from the dataset.

        ### Args
            - `index` (int): Index of the sample to get

        ### Returns
            - `Tuple[Any, Any]`: Tuple containing (data, target) where:
                - data: The processed input data
                - target: The corresponding target label

        ### Example
        ```python
        >>> dataset = ClassificationDataset("path/to/data")
        >>> data, target = dataset[0]  # Get first sample
        ```
        """
        data, target = self.data[index], self.targets[index]
        data = self._preprocess_data(data)
        if self.transform is not None:
            data = self.transform(data)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return data, target

    def __len__(self) -> int:
        """Return the length of the dataset.

        ### Returns
            - `int`: Number of samples in the dataset
        """
        return len(self.data)

    def _initialize(self) -> None:
        """Initialize dataset.

        This method must be implemented by subclasses to initialize the dataset attributes:
            - `self.data`: List/array of data samples
            - `self.targets`: List/array of target labels (integers)
            - `self.labels`: List of label names (strings)

        ### Returns
            - `None`
        """
        raise NotImplementedError

    def _preprocess_data(self, data: Any) -> Any:
        """Preprocess data before applying transforms.

        This method must be implemented by subclasses to preprocess the data.

        ### Args
            - `data` (Any): Data to preprocess

        ### Returns
            - `Any`: Preprocessed data
        """
        raise NotImplementedError

    def _download(
        self,
        urls: List[str],
        checklist: List[Tuple[Path, str]],
        extract_dir: Optional[Path] = ".",
    ) -> None:
        """Download dataset files from URLs and verify their integrity.

        ### Args
            - `urls` (List[str]): List of URLs to download files from
            - `checklist` (List[Tuple[Path, str]]): List of (filepath, checksum) tuples to verify. Should include both downloaded and extracted files.
            - `extract_dir` (Optional[Path]): Directory to extract downloaded files to, relative to self.root. Defaults to "."

        ### Returns
            - `None`

        ### Example
        ```python
        >>> urls = ['http://example.com/data.zip']
        >>> checklist = [
        ...     ('data.zip', 'abc123'),          # Downloaded file
        ...     ('extracted/data.txt', 'def456')  # Extracted file
        ... ]
        >>> _download(urls, checklist, 'extracted')
        ```
        """
        dataset_root = _path.join(self.root, extract_dir)
        # Check if dataset already exists and is valid
        dataset_exists = _path.exists(dataset_root) and all(
            _path.exists(_path.join(dataset_root, path))
            for path, _ in checklist[len(urls) :]
        )
        # If so, return
        if dataset_exists and self._check(checklist[len(urls) :], dataset_root):
            return
        # Else, download and verify files
        while True:
            # Download files
            paths = [utils.download.from_url(url, self.root) for url in urls]
            # Verify downloads
            if self._check(checklist[: len(urls)], dataset_root):
                break
            # If verification failed, clean up and retry
            print("Dataset corrupted. Redownloading dataset.")
            for path in paths:
                _path.rmfile(path)
        # Extract downloads
        for path in paths:
            utils.zips.extract(path, dataset_root)
        # Verify extracted files
        assert self._check(checklist[len(urls) :], dataset_root)

    def _check(self, checklist: List[Tuple[Path, str]], root: Path) -> bool:
        """Check if all files in checklist exist and have correct MD5 checksums.

        ### Args
            - `checklist` (List[Tuple[Path, str]]): List of (filepath, checksum) tuples to verify
            - `root` (Path): Root directory containing the files

        ### Returns
            - `bool`: True if all files exist and have correct checksums, False otherwise

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
            utils.download.md5_check(
                path=_path.join(root, path),
                checksum=checksum,
            )
            for path, checksum in checklist
        )
