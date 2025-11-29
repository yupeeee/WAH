import os
import pickle
from typing import Callable, Literal, Optional, Union

import numpy as np

from .cifar10 import CIFAR10
from .transform import DeNormalize, Normalize

__all__ = [
    "CIFAR100",
]


class CIFAR100(CIFAR10):
    """[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset.

    ### Args
        - `root` (os.PathLike): Root directory where the dataset exists or will be saved to.
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
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
          If the dataset is already downloaded, it is not downloaded again.

    ### Attributes
        - `root` (os.PathLike): Root directory where the dataset exists or will be saved to.
        - `transform` (Callable, optional): A function/transform that takes in the data and transforms it. Defaults to None.
        - `target_transform` (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
        - `data`: Data of the dataset.
        - `targets`: Targets of the dataset.
        - `labels`: Labels of the dataset.
        - `MEAN` (List[float]): Channel-wise mean for normalization
        - `STD` (List[float]): Channel-wise std for normalization
        - `NORMALIZE` (Normalize): Normalization transform
        - `DENORMALIZE` (DeNormalize): De-normalization transform

    ### Example
    ```python
    >>> dataset = CIFAR100("path/to/dataset", split="train", transform="auto")
    >>> len(dataset)  # Get dataset size
    50000
    >>> data, target = dataset[0]  # Get first sample and target
    ```
    """

    _urls = [
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    ]
    _zip_list = [
        ("cifar-100-python.tar.gz", "eb9058c3a382ffc7106e4002c42a8d85"),
    ]
    _train_list = [
        ("cifar-100-python/train", "16019d7e3df5f24257cddd939b257f8d"),
    ]
    _test_list = [
        ("cifar-100-python/test", "f0ef6b0ae62326f3e7ffdfab6717acfc"),
    ]
    _meta_list = [
        ("cifar-100-python/meta", "7973b15100ade9c7d40fb424638fde48"),
    ]
    MEAN = [0.5071, 0.4866, 0.4409]
    STD = [0.2673, 0.2564, 0.2762]
    NORMALIZE = Normalize(MEAN, STD)
    DENORMALIZE = DeNormalize(MEAN, STD)

    def __init__(
        self,
        root: os.PathLike = "cifar100",
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
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        - `root` (os.PathLike): Root directory where the dataset exists or will be saved to.
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
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
          If the dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root,
            split,
            transform,
            target_transform,
            download,
            **kwargs,
        )

    def _initialize(
        self,
    ) -> None:
        self.data = []
        self.targets = []

        # load data/targets
        for fname, _ in self._checklist[1:]:
            fpath = os.path.join(self.root, fname)

            with open(fpath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # BCHW
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC

        # load labels
        labels_fname, _ = self._meta_list[0]
        labels_fpath = os.path.join(self.root, labels_fname)

        with open(labels_fpath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.labels = data["fine_label_names"]
