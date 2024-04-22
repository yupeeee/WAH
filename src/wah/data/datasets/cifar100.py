import os
import pickle

import numpy as np
import torchvision.transforms as tf

from ...typing import (
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
    URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    ROOT = "./datasets/cifar100"
    MODE = "r:gz"

    ZIP_LIST = [
        ("cifar-100-batches-py.tar.gz", "eb9058c3a382ffc7106e4002c42a8d85"),
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

    MEAN = [0.5071, 0.4865, 0.4409]
    STD = [0.2673, 0.2564, 0.2762]
    NORMALIZE = tf.Normalize(MEAN, STD)

    TRANSFORM = {
        "train": tf.Compose(
            [
                tf.RandomHorizontalFlip(),
                tf.RandomCrop(32, 4),
                tf.ToTensor(),
                NORMALIZE,
            ]
        ),
        "test": tf.Compose(
            [
                tf.ToTensor(),
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
            ],
        ] = None,
        target_transform: Union[Optional[Callable], Literal["auto",]] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, split, transform, target_transform, download)

    def initialize(
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
