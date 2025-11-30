import os
import shutil
from typing import Callable, Literal, Optional, Union

import tqdm
from PIL import Image

from ...misc import lists, path, zips
from .base import ClassificationDataset
from .ILSVRC2012 import _ilsvrc2012_labels, _ilsvrc2012_val_meta
from .transform import (
    ClassificationPresetEval,
    ClassificationPresetTrain,
    DeNormalize,
    Normalize,
)

__all__ = [
    "ImageNet",
]


class ImageNet(ClassificationDataset):
    """[ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset.

    ### Args
        - `root` (os.PathLike): Root directory where the dataset exists or will be saved to.
        - `split` (Literal["train", "val"]): The dataset split; supports "train" (default), and "val".
        - `transform` (Union[Optional[Callable], Literal["auto", "tt", "train", "val"]]): A function/transform that takes in the data and transforms it.
        Supports "auto", "tt", "train", "val", and None (default).
            - "auto": Automatically initializes the transform based on the dataset type and `split`.
            - "tt": Converts data into a tensor image.
            - "train": Transform to use in the train stage.
            - "val": Transform to use in the validation stage.
            - None (default): No transformation is applied.
        - `target_transform` (Union[Optional[Callable], Literal["auto"]]): A function/transform that takes in the target and transforms it.
        Supports "auto", and None (default).
            - "auto": Automatically initializes the transform based on the dataset type and `split`.
            - None (default): No transformation is applied.
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
        If the dataset is already downloaded, it is not downloaded again.

    ### Attributes
        - `root` (os.PathLike): Root directory where the dataset exists or will be saved to.
        - `transform` (Callable, optional): A function/transform that takes in the data and transforms it. Defaults to None.
        - `target_transform` (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
        - `data`: Data of the dataset.
        - `targets`: Targets of the dataset.
        - `labels`: Labels of the dataset.
        - `MEAN` (List[float]): Channel-wise mean for normalization
        - `STD` (List[float]): Channel-wise std for normalization
        - `NORMALIZE` (Normalize): Normalization transform
        - `DENORMALIZE` (DeNormalize): De-normalization transform

    ### Example
    ```python
    >>> dataset = ImageNet("path/to/dataset", split="train", transform="auto")
    >>> len(dataset)  # Get dataset size
    1281167
    >>> data, target = dataset[0]  # Get first sample and target
    ```
    """

    _train_urls = [
        "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
    ]
    _val_urls = [
        "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    ]
    _train_list = [
        ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    ]
    _val_list = [
        ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    ]
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NORMALIZE = Normalize(MEAN, STD)
    DENORMALIZE = DeNormalize(MEAN, STD)

    def __init__(
        self,
        root: os.PathLike = "imagenet",
        split: Literal["train", "val"] = "train",
        transform: Union[
            Optional[Callable],
            Literal[
                "auto",
                "train",
                "val",
                "tt",
            ],
        ] = None,
        target_transform: Union[Optional[Callable], Literal["auto"]] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        - `root` (os.PathLike): Root directory where the dataset exists or will be saved to.
        - `split` (Literal["train", "val"]): The dataset split; supports "train" (default), and "val".
        - `transform` (Union[Optional[Callable], Literal["auto", "tt", "train", "val"]]): A function/transform that takes in the data and transforms it.
        Supports "auto", "tt", "train", "val", and None (default).
            - "auto": Automatically initializes the transform based on the dataset type and `split`.
            - "tt": Converts data into a tensor image.
            - "train": Transform to use in the train stage.
            - "val": Transform to use in the validation stage.
            - None (default): No transformation is applied.
        - `target_transform` (Union[Optional[Callable], Literal["auto"]]): A function/transform that takes in the target and transforms it.
        Supports "auto", and None (default).
            - "auto": Automatically initializes the transform based on the dataset type and `split`.
            - None (default): No transformation is applied.
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
        If the dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root,
            transform,
            target_transform,
        )
        self.split = split

        # checklist
        self._urls = []
        self._checklist = []
        if split == "train":
            self._urls += self._train_urls
            self._checklist += self._train_list
        elif split == "val":
            self._urls += self._val_urls
            self._checklist += self._val_list
        else:
            raise ValueError(f"split must be one of ['train', 'val'], got {split}")

        # transform
        if self.transform == "auto" and split == "train" or self.transform == "train":
            kwargs.update({"mean": self.MEAN, "std": self.STD})
            self.transform = ClassificationPresetTrain(**kwargs)
        elif self.transform == "auto" and split == "val" or self.transform == "val":
            kwargs.update({"mean": self.MEAN, "std": self.STD})
            self.transform = ClassificationPresetEval(**kwargs)
        elif self.transform == "tt":
            self.transform = ClassificationPresetEval(**kwargs)
        else:
            pass

        # target_transform
        if self.target_transform == "auto":
            self.target_transform = None
        else:
            pass

        # download
        if download:
            # download train dataset
            self._download(
                urls=self._urls,
                checklist=self._checklist,
                extract_dir=split,
            )

            # extract class folders
            data_root = os.path.join(self.root, split)

            if not lists.eq(
                path.ls(data_root, fext="dir", sort=True, absolute=False),
                lists.sort(list(set(_ilsvrc2012_val_meta))),
            ):
                if split == "train":
                    for wnid in set(_ilsvrc2012_val_meta):
                        fpath = os.path.join(data_root, f"{wnid}.tar")
                        zips.extract(fpath, save_dir=os.path.join(data_root, wnid))

                        # delete zipfiles
                        os.remove(fpath)

                else:
                    for wnid in set(_ilsvrc2012_val_meta):
                        os.makedirs(os.path.join(data_root, wnid), exist_ok=True)

                    for wnid, image_path in zip(
                        _ilsvrc2012_val_meta,
                        path.ls(data_root, fext=".JPEG", sort=True, absolute=True),
                    ):
                        shutil.move(
                            image_path,
                            os.path.join(data_root, wnid, os.path.basename(image_path)),
                        )

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        # load class folders
        data_root = os.path.join(self.root, self.split)
        classes = path.ls(
            path=data_root,
            fext="dir",
            sort=True,
        )

        # load data (list of path to images) & targets
        self.data = []
        self.targets = []

        for class_idx, c in enumerate(
            tqdm.tqdm(
                classes,
                desc=f"Initializing ImageNet dataset ({self.split})",
            )
        ):
            class_dir = os.path.join(data_root, c)
            fpaths = path.ls(path=class_dir, fext=".JPEG", sort=True)

            path_to_images = [os.path.join(data_root, c, fpath) for fpath in fpaths]

            self.data += path_to_images
            self.targets += [class_idx] * len(fpaths)

        # load labels
        self.labels = _ilsvrc2012_labels

    def _preprocess_data(
        self,
        fpath: os.PathLike,
    ) -> Image.Image:
        return Image.open(fpath).convert("RGB")
