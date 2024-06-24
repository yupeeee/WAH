import os

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
    "STL10",
]


class STL10(ClassificationDataset):
    """
    [STL-10](https://ai.stanford.edu/~acoates/stl10/) dataset.

    ### Attributes
    - `root` (path):
      Root directory where the dataset exists or will be saved to.
    - `transform` (callable, optional):
      A function/transform that takes in the data (PIL image, numpy.ndarray, etc.) and transforms it.
      If None, no transformation is applied.
    - `target_transform` (callable, optional):
      A function/transform that takes in the target (int, etc.) and transforms it.
      If None, no transformation is applied.
    - `data`:
      Data of the dataset.
    - `targets`:
      Targets of the dataset.
    - `labels`:
      Labels of the dataset.
    - `MEAN` (list):
      mean of dataset; [0.4467, 0.4398, 0.4066].
    - `STD` (list):
      std of dataset; [0.2603, 0.2566, 0.2713].
    - `NORMALIZE` (callable):
      transform for dataset normalization.

    ### Methods
    - `__getitem__`:
      Returns (data, target) of dataset using the specified index.

      Example:
      ```python
      dataset = STL10(root="path/to/dataset")
      data, target = dataset[0]
      ```
    - `__len__`:
      Returns the size of the dataset.

      Example:
      ```python
      dataset = STL10(root="path/to/dataset")
      num_data = len(dataset)
      ```
    """

    URLS = [
        "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
    ]
    ROOT = os.path.normpath("./datasets/stl10")

    ZIP_LIST = [
        ("stl10_binary.tar.gz", "91f7769df0f17e558f3565bffb0c7dfb"),
    ]
    TRAIN_LIST = [
        ("stl10_binary/train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"),
        ("stl10_binary/train_y.bin", "5a34089d4802c674881badbb80307741"),
    ]
    TEST_LIST = [
        ("stl10_binary/test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"),
        ("stl10_binary/test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"),
    ]
    META_LIST = [
        ("stl10_binary/class_names.txt", "6de44d022411c4d5cda4673b7f147c3f"),
        # ("stl10_binary/fold_indices.txt", "4bbf8cd098ab9f87fe03fbcb37c06b28"),
        # ("stl10_binary/unlabeled_X.bin", "MEMORYERROR"),
    ]

    MEAN = [0.4467, 0.4398, 0.4066]
    STD = [0.2603, 0.2566, 0.2713]
    NORMALIZE = T.Normalize(MEAN, STD)

    TRANSFORM = {
        "train": T.Compose(
            [
                T.RandomResizedCrop(96),
                T.RandomHorizontalFlip(),
                # T.AutoAugment(T.AutoAugmentPolicy(None)),
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
        else:
            pass

        if download:
            self._download(
                urls=self.URLS,
                checklist=self.checklist + self.META_LIST,
                ext_dir_name=".",
            )

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        # load data
        data_fname, _ = self.checklist[1]
        data_fpath = os.fspath(os.path.join(self.root, data_fname))

        with open(data_fpath, "rb") as f:
            self.data = np.fromfile(f, dtype=np.uint8)
            self.data = self.data.reshape((-1, 3, 96, 96))  # BCWH
            self.data = self.data.transpose((0, 3, 2, 1))  # convert to BHWC

        # load targets
        targets_fname, _ = self.checklist[2]
        targets_fpath = os.fspath(os.path.join(self.root, targets_fname))

        with open(targets_fpath, "rb") as f:
            self.targets = [target - 1 for target in np.fromfile(f, dtype=np.uint8)]

        # load labels
        labels_fname, _ = self.META_LIST[0]
        labels_fpath = os.fspath(os.path.join(self.root, labels_fname))

        with open(labels_fpath, "r") as f:
            self.labels = [label for label in f.read().split("\n") if len(label)]

    def _preprocess_data(
        self,
        data: np.ndarray,
    ) -> Image.Image:
        return Image.fromarray(data)
