import os
import pickle

import numpy as np
from PIL import Image
import torchvision.transforms as T

from ...typing import (
    Callable,
    Literal,
    Optional,
    Path,
    Union,
)
from .utils import DNTDataset

__all__ = [
    "CIFAR10",
]


class CIFAR10(DNTDataset):
    URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    ROOT = "./datasets/cifar10"
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
            self.download(self.checklist + self.META_LIST)

        self.initialize()

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
                self.targets.extend(entry["labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # BCHW
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC

        # load labels
        labels_fname, _ = self.META_LIST[0]
        labels_fpath = os.fspath(os.path.join(self.root, labels_fname))

        with open(labels_fpath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.labels = data["label_names"]

    def preprocess_data(
        self,
        data: np.ndarray,
    ) -> Image.Image:
        return Image.fromarray(data)
