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
from ..utils.zip import extract
from .base import ClassificationDataset
from .labels import imagenet1k as labels

__all__ = [
    "ImageNetTrain",
    "ImageNetVal",
]


class ImageNetTrain(ClassificationDataset):
    """
    [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) train dataset.

    ### Attributes
    - `root` (Path):
      Root directory where the dataset exists or will be saved to.
    - `transform` (Callable, optional):
      A function/transform that takes in the data and transforms it.
      If None, no transformation is performed. Defaults to None.
    - `target_transform` (Callable, optional):
      A function/transform that takes in the target and transforms it.
      If None, no transformation is performed. Defaults to None.
    - `data`:
      Data of the dataset.
    - `targets`:
      Targets of the dataset.
    - `labels`:
      Labels of the dataset.
    - `MEAN` (list):
      mean of dataset; [0.485, 0.456, 0.406].
    - `STD` (list):
      std of dataset; [0.229, 0.224, 0.225].
    - `NORMALIZE` (callable):
      transform for dataset normalization.

    ### Methods
    - `__getitem__`:
      Returns (data, target) of dataset using the specified index.
    - `__len__`:
      Returns the size of the dataset.
    - `set_return_data_only`:
      Sets the flag to return only data without targets.
    - `unset_return_data_only`:
      Unsets the flag to return only data without targets.

    ### Example
    ```python
    dataset = ImageNetTrain(download=True)
    data, target = dataset[0]
    num_data = len(dataset)
    ```
    """

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
        return_data_only: Optional[bool] = False,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform, return_data_only)

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
            # download train dataset
            self._download(
                urls=self.URLS,
                checklist=self.checklist,
                ext_dir_name="train",
            )

            # extract class folders
            data_root = os.path.normpath(os.path.join(self.root, "train"))
            classes = ls(data_root, fext=".tar", sort=True)

            if len(classes):
                classes = [c.split(".tar")[0] for c in classes]

                for c in classes:
                    fpath = os.path.join(data_root, f"{c}.tar")
                    extract(fpath, save_dir=os.path.join(data_root, c))

                    # delete zipfiles
                    os.remove(fpath)

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        # load class folders
        data_root = os.path.normpath(os.path.join(self.root, "train"))
        classes = ls(
            path=data_root,
            fext="dir",
            sort=True,
        )

        # load data (list of path to images) & targets
        self.data = []
        self.targets = []

        for class_idx, c in enumerate(classes):
            class_dir = os.path.normpath(os.path.join(data_root, c))
            fpaths = ls(path=class_dir, fext=".JPEG", sort=True)

            path_to_images = [
                os.path.normpath(os.path.join(data_root, c, fpath)) for fpath in fpaths
            ]

            self.data += path_to_images
            self.targets += [class_idx] * len(fpaths)

        # load labels
        self.labels = labels

    def _preprocess_data(
        self,
        fpath: Path,
    ) -> Image.Image:
        return Image.open(fpath).convert("RGB")


class ImageNetVal(ClassificationDataset):
    """
    [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) validation dataset.

    ### Attributes
    - `root` (Path):
      Root directory where the dataset exists or will be saved to.
    - `transform` (Callable, optional):
      A function/transform that takes in the data and transforms it.
      If None, no transformation is performed. Defaults to None.
    - `target_transform` (Callable, optional):
      A function/transform that takes in the target and transforms it.
      If None, no transformation is performed. Defaults to None.
    - `data`:
      Data of the dataset.
    - `targets`:
      Targets of the dataset.
    - `labels`:
      Labels of the dataset.
    - `MEAN` (list):
      mean of dataset; [0.485, 0.456, 0.406].
    - `STD` (list):
      std of dataset; [0.229, 0.224, 0.225].
    - `NORMALIZE` (callable):
      transform for dataset normalization.

    ### Methods
    - `__getitem__`:
      Returns (data, target) of dataset using the specified index.
    - `__len__`:
      Returns the size of the dataset.
    - `set_return_data_only`:
      Sets the flag to return only data without targets.
    - `unset_return_data_only`:
      Unsets the flag to return only data without targets.

    ### Example
    ```python
    dataset = ImageNetVal(download=True)
    data, target = dataset[0]
    num_data = len(dataset)
    ```
    """

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
        return_data_only: Optional[bool] = False,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform, target_transform, return_data_only)

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
            # download ground truth targets
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
