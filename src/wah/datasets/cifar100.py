import os
import pickle

import numpy as np
import torchvision.transforms as T

from ..typing import (
    Callable,
    Literal,
    Optional,
    Path,
    Union,
)
from .cifar10 import CIFAR10

__all__ = [
    "CIFAR100",
]


class CIFAR100(CIFAR10):
    """
    [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

    ### Attributes
    - `root` (path):
      Root directory where the dataset exists or will be saved to.
    - `transform` (callable, optional):
      A function/transform that takes in the data (PIL image, numpy.ndarray, etc.) and transforms it.
      If None, no transformation is applied.
      Defaults to None.
    - `target_transform` (callable, optional):
      A function/transform that takes in the target (int, etc.) and transforms it.
      If None, no transformation is applied.
      Defaults to None.
    - `data`:
      Data of the dataset.
    - `targets`:
      Targets of the dataset.
    - `labels`:
      Labels of the dataset.
    - `MEAN` (list):
      mean of dataset; [0.5071, 0.4866, 0.4409].
    - `STD` (list):
      std of dataset; [0.2673, 0.2564, 0.2762].
    - `NORMALIZE` (callable):
      transform for dataset normalization.

    ### Methods
    - `__getitem__`:
      Returns (data, target) of dataset using the specified index.

      Example:
      ```python
      dataset = CIFAR100(root="path/to/dataset")
      data, target = dataset[0]
      ```
    - `__len__`:
      Returns the size of the dataset.

      Example:
      ```python
      dataset = CIFAR100(root="path/to/dataset")
      num_data = len(dataset)
      ```
    """

    URLS = [
        "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz",
    ]
    ROOT = os.path.normpath("./datasets/cifar100")

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
    NORMALIZE = T.Normalize(MEAN, STD)

    TRANSFORM = {
        "train": T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomCrop(32, 4),
                T.ToTensor(),
                NORMALIZE,
            ]
        ),
        "test": T.Compose(
            [
                T.ToTensor(),
                NORMALIZE,
            ]
        ),
    }
    TARGET_TRANSFORM = {
        "train": None,
        "test": None,
    }

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
        download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)

    def _initialize(
        self,
    ) -> None:
        self.data = []
        self.targets = []

        # load data/targets
        for fname, _ in self.checklist[1:]:
            fpath = os.fspath(os.path.join(self.root, fname))

            with open(fpath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # BCHW
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC

        # load labels
        labels_fname, _ = self.META_LIST[0]
        labels_fpath = os.fspath(os.path.join(self.root, labels_fname))

        with open(labels_fpath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.labels = data["fine_label_names"]
