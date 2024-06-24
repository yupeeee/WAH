import os

import torchvision.transforms as T
from PIL import Image

from ..typing import (
    Callable,
    Literal,
    Optional,
    Path,
    Union,
)
from ..utils.download import check, download_url
from ..utils.lst import load_txt
from ..utils.path import ls
from .base import ClassificationDataset
from .labels import imagenet1k as labels

__all__ = [
    "ImageNetTrain",
    "ImageNetVal",
]


class ImageNetTrain(ClassificationDataset):
    URLS = [
        "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
    ]
    ROOT = os.path.normpath("./datasets/imagenet")

    ZIP_LIST = [
        ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    ]

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NORMALIZE = T.Normalize(MEAN, STD)

    TRANSFORM = {
        "train": T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                NORMALIZE,
            ]
        ),
        "val": T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                NORMALIZE,
            ]
        ),
        "tt": T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ]
        ),
    }
    TARGET_TRANSFORM = {
        "train": None,
        "val": None,
    }

    def __init__(
        self,
        root: Path = ROOT,
        transform: Union[
            Optional[Callable],
            Literal[
                "auto",
                "train",
                "val",
                "tt",
            ],
        ] = None,
        target_transform: Union[Optional[Callable], Literal["auto",]] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform)

        self.checklist = []

        self.checklist += self.ZIP_LIST

        if self.transform == "auto":
            self.transform = self.TRANSFORM["train"]
        elif self.transform is not None:
            self.transform = self.TRANSFORM[self.transform]
        else:
            pass

        if self.target_transform == "auto":
            self.target_transform = self.TARGET_TRANSFORM["val"]
        else:
            pass

        if download:
            # download validation dataset
            self._download(
                urls=self.URLS,
                checklist=self.checklist,
                ext_dir_name="train",
            )

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        # load data (list of path to images)
        data_root = os.path.normpath(os.path.join(self.root, "train"))
        fnames = ls(
            path=data_root,
            fext="JPEG",
            sort=True,
        )

        self.data = []
        for fname in fnames:
            fpath = os.path.normpath(os.path.join(data_root, fname))
            self.data.append(fpath)

        # load targets
        targets_path = os.path.normpath(
            os.path.join(
                self.root,
                "ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt",
            )
        )
        self.targets = load_txt(targets_path, dtype=int)

        # load labels
        self.labels = labels

    def _preprocess_data(
        self,
        fpath: Path,
    ) -> Image.Image:
        return Image.open(fpath).convert("RGB")


class ImageNetVal(ClassificationDataset):
    URLS = [
        "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
        "https://raw.githubusercontent.com/yupeeee/WAH/main/src/wah/datasets/targets/ILSVRC2012_validation_ground_truth.txt",
    ]
    ROOT = os.path.normpath("./datasets/imagenet")

    ZIP_LIST = [
        ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
        ("ILSVRC2012_validation_ground_truth.txt", "f31656d784908741c59ccb6823cf0bea"),
    ]

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NORMALIZE = T.Normalize(MEAN, STD)

    TRANSFORM = {
        "val": T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                NORMALIZE,
            ]
        ),
        "tt": T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
            ]
        ),
    }
    TARGET_TRANSFORM = {
        "val": None,
    }

    def __init__(
        self,
        root: Path = ROOT,
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
        super().__init__(root, transform, target_transform)

        self.checklist = []

        self.checklist += self.ZIP_LIST

        if self.transform == "auto":
            self.transform = self.TRANSFORM["val"]
        elif self.transform == "tt":
            self.transform = self.TRANSFORM["tt"]
        else:
            pass

        if self.target_transform == "auto":
            self.target_transform = self.TARGET_TRANSFORM["val"]
        else:
            pass

        if download:
            # download validation dataset
            self._download(
                urls=self.URLS[:1],
                checklist=self.checklist[:1],
                ext_dir_name="val",
            )
            # download devkit
            fpath = download_url(self.URLS[1], self.root)
            check(fpath, self.checklist[1][1])

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        # load data (list of path to images)
        data_root = os.path.normpath(os.path.join(self.root, "val"))
        fnames = ls(
            path=data_root,
            fext="JPEG",
            sort=True,
        )

        self.data = []
        for fname in fnames:
            fpath = os.path.normpath(os.path.join(data_root, fname))
            self.data.append(fpath)

        # load targets
        targets_path = os.path.normpath(
            os.path.join(
                self.root,
                "ILSVRC2012_validation_ground_truth.txt",
            )
        )
        self.targets = load_txt(targets_path, dtype=int)

        # load labels
        self.labels = labels

    def _preprocess_data(
        self,
        fpath: Path,
    ) -> Image.Image:
        return Image.open(fpath).convert("RGB")
