import os

import numpy as np
from PIL import Image

from ...typing import (
    Callable,
    Literal,
    Optional,
    Path,
)
from .utils import DNTDataset

__all__ = [
    "STL10",
]


class STL10(DNTDataset):
    URL = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    ROOT = "./datasets/stl10"
    MODE = "r:gz"

    ZIP_LIST = [
        ("stl10_binary.tar.gz", "c58f30108f718f92721af3b95e74349a"),
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

    def __init__(
            self,
            root: Path = ROOT,
            split: Literal["train", "test", ] = "train",
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform)

        self.checklist = self.ZIP_LIST

        if split == "train":
            self.checklist += self.TRAIN_LIST

        elif split == "test":
            self.checklist += self.TEST_LIST

        else:
            raise ValueError(
                f"split must be one of ['train', 'test', ], got {split}")

        if download:
            self.download(self.checklist + self.META_LIST)

        self.initialize()

    def initialize(self, ) -> None:
        # load data
        data_fname, _ = self.checklist[1]
        data_fpath = os.fspath(os.path.join(self.root, data_fname))

        with open(data_fpath, "rb") as f:
            self.data = np.fromfile(f, dtype=np.uint8)
            self.data = self.data.reshape((-1, 3, 96, 96))  # BCWH
            self.data = self.data.transpose((0, 3, 2, 1))   # convert to BHWC

        # load targets
        targets_fname, _ = self.checklist[2]
        targets_fpath = os.fspath(os.path.join(self.root, targets_fname))

        with open(targets_fpath, "rb") as f:
            self.targets = [
                target -
                1 for target in np.fromfile(
                    f,
                    dtype=np.uint8)]

        # load labels
        labels_fname, _ = self.META_LIST[0]
        labels_fpath = os.fspath(os.path.join(self.root, labels_fname))

        with open(labels_fpath, "r") as f:
            self.labels = [label for label in f.read().split("\n")
                           if len(label)]

    def preprocess_data(self, data: np.ndarray, ) -> Image.Image:
        return Image.fromarray(data)
