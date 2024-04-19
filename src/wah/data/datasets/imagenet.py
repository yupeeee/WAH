import os
import pickle

import numpy as np
from PIL import Image

from ...typing import Callable, Optional, Path
from .utils import DNTDataset

__all__ = [
    "ImageNet",
]


class ImageNet(DNTDataset):
    val = {
        "URL": "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        "ROOT": "./datasets/imagenet",
        "MODE": "r:gz",
    }
    ROOT = "./datasets/imagenet"
    MODE = "r:gz"

    ZIP_LIST = [
        ("", ""),
    ]
    TRAIN_LIST = [
        ("", ""),
    ]
    VAL_LIST = [
        ("", ""),
    ]
    TEST_LIST = [
        ("", ""),
    ]
    META_LIST = [
        ("", ""),
    ]

    def __init__(
        self,
        root: Path = ROOT,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform)

        self.checklist = self.ZIP_LIST

        if split == "train":
            self.checklist += self.TRAIN_LIST

        elif split == "val":
            self.checklist += self.VAL_LIST
        
        elif split == "test":
            self.checklist += self.TEST_LIST

        else:
            raise ValueError(
                f"split must be one of ['train', 'val', 'test', ], got {split}")

        if download:
            self.download(self.checklist + self.META_LIST)

        self.initialize()

    def initialize(self, ) -> None:
        # TODO
        self.data = []
        self.targets = []
        self.labels = []

    def preprocess_data(self, data_path: Path, ) -> Image.Image:
        return Image.open(data_path).convert("RGB")
