import pickle

import numpy as np
from torchvision.transforms import Normalize

from ... import path as _path
from ...typing import Callable, Literal, Optional, Path, Union
from .cifar10 import CIFAR10
from .transforms import DeNormalize

__all__ = [
    "CIFAR100",
]


class CIFAR100(CIFAR10):
    """
    [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

    ### Attributes
    - `root` (Path): Root directory where the dataset exists or will be saved to.
    - `transform` (Callable, optional): A function/transform that takes in the data and transforms it. Defaults to None.
    - `target_transform` (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
    - `data`: Data of the dataset.
    - `targets`: Targets of the dataset.
    - `labels`: Labels of the dataset.
    - `MEAN` (list): Mean of dataset; [0.5071, 0.4866, 0.4409].
    - `STD` (list): Standard deviation of dataset; [0.2673, 0.2564, 0.2762].
    - `NORMALIZE` (callable): Transform for dataset normalization.
    - `DENORMALIZE` (callable): Transform for dataset denormalization.

    ### Methods
    - `__getitem__(index) -> Tuple[Any, Any]`: Returns (data, target) of dataset using the specified index.
    - `__len__() -> int`: Returns the size of the dataset.
    - `set_return_data_only() -> None`: Sets the flag to return only data without targets.
    - `unset_return_data_only() -> None`: Unsets the flag to return only data without targets.
    - `set_return_w_index() -> None`: Sets the flag to return data with index.
    - `unset_return_w_index() -> None`: Unsets the flag to return data with index.

    ### Example
    ```python
    import wah

    dataset = wah.datasets.CIFAR100(root="path/to/dataset")
    data, target = dataset[0]
    num_data = len(dataset)
    ```
    """

    URLS = [
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    ]
    ROOT = _path.clean("./datasets/cifar100")

    ZIP_LIST = [
        ("cifar-100-python.tar.gz", "eb9058c3a382ffc7106e4002c42a8d85"),
    ]
    TRAIN_LIST = [
        ("cifar-100-python/train", "16019d7e3df5f24257cddd939b257f8d"),
    ]
    TEST_LIST = [
        ("cifar-100-python/test", "f0ef6b0ae62326f3e7ffdfab6717acfc"),
    ]
    META_LIST = [
        ("cifar-100-python/meta", "7973b15100ade9c7d40fb424638fde48"),
    ]

    MEAN = [0.5071, 0.4866, 0.4409]
    STD = [0.2673, 0.2564, 0.2762]
    NORMALIZE = Normalize(MEAN, STD)
    DENORMALIZE = DeNormalize(MEAN, STD)

    def __init__(
        self,
        root: Path = ROOT,
        split: Literal[
            "train",
            "test",
        ] = "train",
        transform: Union[
            Optional[Callable],
            Literal[
                "auto",
                "tt",
                "train",
                "test",
            ],
        ] = None,
        target_transform: Union[Optional[Callable], Literal["auto",]] = None,
        return_data_only: Optional[bool] = False,
        return_w_index: Optional[bool] = False,
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the CIFAR-100 dataset.

        ### Parameters
        - `root` (Path): Root directory where the dataset exists or will be saved to.
        - `split` (Literal["train", "test"]): The dataset split; supports "train" (default), and "test".
        - `transform` (Union[Optional[Callable], Literal["auto", "tt", "train", "test"]]): A function/transform that takes in the data and transforms it.
          Supports "auto", "tt", "train", "test", and None (default).
          - "auto": Automatically initializes the transform based on the dataset type and `split`.
          - "tt": Converts data into a tensor image.
          - "train": Transform to use in the train stage.
          - "test": Transform to use in the test stage.
          - None (default): No transformation is applied.
        - `target_transform` (Union[Optional[Callable], Literal["auto"]]): A function/transform that takes in the target and transforms it.
          Supports "auto", and None (default).
          - "auto": Automatically initializes the transform based on the dataset type and `split`.
          - None (default): No transformation is applied.
        - `return_data_only` (Optional[bool]): Whether to return only data without targets. Defaults to False.
        - `return_w_index` (Optional[bool]): Whether to return data with index. Defaults to False.
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
          If the dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root,
            split,
            transform,
            target_transform,
            return_data_only,
            return_w_index,
            download,
            **kwargs,
        )

    def _initialize(
        self,
    ) -> None:
        """
        Initializes the CIFAR-100 dataset.

        ### Returns
        - `None`
        """
        self.data = []
        self.targets = []

        # load data/targets
        for fname, _ in self.checklist[1:]:
            fpath = _path.join(self.root, fname)

            with open(fpath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # BCHW
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC

        # load labels
        labels_fname, _ = self.META_LIST[0]
        labels_fpath = _path.join(self.root, labels_fname)

        with open(labels_fpath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.labels = data["fine_label_names"]
