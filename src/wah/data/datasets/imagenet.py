import torchvision.transforms as T
from PIL import Image

from ...typing import (
    Callable,
    Literal,
    Optional,
    Path,
    Union,
)
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

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NORMALIZE = T.Normalize(MEAN, STD)

    TRANSFORM = {
        "train": T.Compose(
            [
                T.RandomResizedCrop(224),
                T.RandomHorizontalFlip(),
                # T.AutoAugment(T.AutoAugmentPolicy(None)),
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
        "test": T.Compose(
            [
                T.Resize(256),
                T.CenterCrop(224),
                T.ToTensor(),
                NORMALIZE,
            ]
        ),
    }
    TARGET_TRANSFORM = {
        "train": None,
        "val": None,
        "test": None,
    }

    def __init__(
        self,
        root: Path = ROOT,
        split: str = "train",
        transform: Union[
            Optional[Callable],
            Literal[
                "auto",
                "tt",
                "train",
                "val",
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

        elif split == "val":
            self.checklist += self.VAL_LIST

        elif split == "test":
            self.checklist += self.TEST_LIST

        else:
            raise ValueError(
                f"split must be one of ['train', 'val', 'test', ], got {split}"
            )

        if self.transform == "auto":
            self.transform = self.TRANSFORM[split]
        elif self.transform == "tt":
            self.transform = T.Compose(
                [
                    T.Resize(256),
                    T.CenterCrop(224),
                    T.ToTensor(),
                ]
            )
        elif self.transform == "train":
            self.transform = self.TRANSFORM["train"]
        elif self.transform == "val":
            self.transform = self.TRANSFORM["val"]
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
        # TODO
        self.data = []
        self.targets = []
        self.labels = []

    def preprocess_data(
        self,
        data_path: Path,
    ) -> Image.Image:
        return Image.open(data_path).convert("RGB")
