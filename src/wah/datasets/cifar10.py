import os
import pickle

import numpy as np
import torchvision.transforms as T
from PIL import Image

from ..typing import (
    Callable,
    Literal,
    Optional,
    Path,
    Union,
)
from .base import ClassificationDataset

__all__ = [
    "CIFAR10",
]


class CIFAR10(ClassificationDataset):
    """
    [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

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
      mean of dataset; [0.4914, 0.4822, 0.4465].
    - `STD` (list):
      std of dataset; [0.2470, 0.2435, 0.2616].
    - `NORMALIZE` (callable):
      transform for dataset normalization.

    ### Methods
    - `__getitem__`:
      Returns (data, target) of dataset using the specified index.

      Example:
      ```python
      dataset = CIFAR10(root="path/to/dataset")
      data, target = dataset[0]
      ```
    - `__len__`:
      Returns the size of the dataset.

      Example:
      ```python
      dataset = CIFAR10(root="path/to/dataset")
      num_data = len(dataset)
      ```
    """

    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    ROOT = os.path.normpath("./datasets/cifar10")
    MODE = "r:gz"

    ZIP_LIST = [
        ("cifar-10-batches-py.tar.gz", "c58f30108f718f92721af3b95e74349a"),
    ]
    TRAIN_LIST = [
        ("cifar-10-batches-py/data_batch_1", "c99cafc152244af753f735de768cd75f"),
        ("cifar-10-batches-py/data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"),
        ("cifar-10-batches-py/data_batch_3", "54ebc095f3ab1f0389bbae665268c751"),
        ("cifar-10-batches-py/data_batch_4", "634d18415352ddfa80567beed471001a"),
        ("cifar-10-batches-py/data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"),
    ]
    TEST_LIST = [
        ("cifar-10-batches-py/test_batch", "40351d587109b95175f43aff81a1287e"),
    ]
    META_LIST = [
        ("cifar-10-batches-py/batches.meta", "5ff9c542aee3614f3951f8cda6e48888"),
    ]

    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616]
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
        """
        - `root` (path):
          Root directory where the dataset exists or will be saved to.
        - `split` (str):
          The dataset split; supports "train" (default), and "test".
        - `transform` (str):
          A function/transform that takes in the data (PIL image, numpy.ndarray, etc.) and transforms it;
          supports "auto", "tt", "train", "test", and None (default).
          - "auto": Automatically initializes the transform based on the dataset type and `split`.
          - "tt": Converts data into a tensor image.
          - "train": Transform to use in train stage.
          - "test": Transform to use in test stage.
          - None (default): No transformation is applied.
        - `target_transform` (str):
          A function/transform that takes in the target (int, etc.) and transforms it;
          supports "auto", and None (default).
          - "auto": Automatically initializes the transform based on the dataset type and `split`.
          - None (default): No transformation is applied.
        - `download` (bool):
          If True, downloads the dataset from the internet and puts it into the `root` directory.
          If dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(root, transform, target_transform)

        self.checklist = []

        self.checklist += self.ZIP_LIST

        if split == "train":
            self.checklist += self.TRAIN_LIST

        elif split == "test":
            self.checklist += self.TEST_LIST

        else:
            raise ValueError(f"split must be one of ['train', 'test', ], got {split}")

        if self.transform == "auto":
            self.transform = self.TRANSFORM[split]
        elif self.transform == "tt":
            self.transform = T.ToTensor()
        elif self.transform == "train":
            self.transform = self.TRANSFORM["train"]
        elif self.transform == "test":
            self.transform = self.TRANSFORM["test"]
        else:
            pass

        if self.target_transform == "auto":
            self.target_transform = self.TARGET_TRANSFORM[split]

        if download:
            self._download(self.checklist + self.META_LIST)

        self._initialize()

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
                self.targets.extend(entry["labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # BCHW
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC

        # load labels
        labels_fname, _ = self.META_LIST[0]
        labels_fpath = os.fspath(os.path.join(self.root, labels_fname))

        with open(labels_fpath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.labels = data["label_names"]

    def _preprocess_data(
        self,
        data: np.ndarray,
    ) -> Image.Image:
        return Image.fromarray(data)
