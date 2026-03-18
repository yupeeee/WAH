"""Currently only able do download dataset"""

import json
import os
from collections import defaultdict
from typing import Callable, Literal, Optional, Union

from PIL import Image

from .base import DetectionDataset

__all__ = [
    "COCO",
]


class COCO(DetectionDataset):
    """[COCO](https://cocodataset.org/#home) dataset.

    TBD
    """

    _name = "COCO"
    _urls = [
        "http://images.cocodataset.org/zips/train2017.zip",
        "http://images.cocodataset.org/zips/val2017.zip",
        "http://images.cocodataset.org/zips/test2017.zip",
        "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
    ]
    _zip_list = [
        ("train2017.zip", "cced6f7f71b7629ddf16f17bbcfab6b2"),
        ("val2017.zip", "442b8da7639aecaf257c1dceb8ba8c80"),
        ("test2017.zip", "77ad2c53ac5d0aea611d422c0938fb35"),
        ("annotations_trainval2017.zip", "f4bbac642086de4f52a3fdda2de5fa2c"),
    ]

    def __init__(
        self,
        root: os.PathLike = "coco",
        split: Literal["train", "val", "test"] = "train",
        transform: Union[
            Optional[Callable],
            Literal["auto",],
        ] = None,
        target_transform: Union[Optional[Callable], Literal["auto"]] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        TBD
        """
        super().__init__(
            root,
            transform,
            target_transform,
        )
        self.split = split

        # checklist
        self._checklist = self._zip_list
        if split not in ["train", "val", "test"]:
            raise ValueError(
                f"split must be one of ['train', 'val', 'test'], got {split}"
            )

        # transform
        if self.transform == "auto":
            self.transform = None
        else:
            pass

        # download
        if download and not all(
            os.path.exists(os.path.join(self.root, dirname))
            for dirname in ["train2017", "val2017", "test2017", "annotations"]
        ):
            self._download(
                urls=self._urls,
                checklist=self._checklist,
            )

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        # Load annotations
        fpath = os.path.join(
            self.root, "annotations", f"instances_{self.split}2017.json"
        )
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.images = {
            img["id"]: img for img in sorted(data["images"], key=lambda x: x["id"])
        }
        self.annotations = {
            ann["id"]: ann for ann in sorted(data["annotations"], key=lambda x: x["id"])
        }
        self.categories = {
            cat["id"]: cat for cat in sorted(data["categories"], key=lambda x: x["id"])
        }

        self.img2ann = defaultdict(list)
        self.cat2img = defaultdict(list)
        for annotation in self.annotations.values():
            image_id = annotation["image_id"]
            category_id = annotation["category_id"]
            self.img2ann[image_id].append(annotation)
            self.cat2img[category_id].append(image_id)

        # TODO: finish implementation

    def _preprocess_data(
        self,
        fpath: os.PathLike,
    ) -> Image.Image:
        return Image.open(fpath).convert("RGB")
