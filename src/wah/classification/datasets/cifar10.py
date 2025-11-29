import os
import pickle
from typing import Callable, Literal, Optional, Union

import numpy as np
from PIL import Image

from .base import ClassificationDataset
from .transform import (
    ClassificationPresetEval,
    ClassificationPresetTrain,
    DeNormalize,
    Normalize,
)

__all__ = [
    "CIFAR10",
]


class CIFAR10(ClassificationDataset):
    """[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) Dataset.

    ### Args
        - `root` (Path): Root directory where the dataset exists or will be saved to.
        - `split` (Literal["train", "test"]): The dataset split; supports "train" (default), and "test".
        - `transform` (Union[Optional[Callable], Literal["auto", "tt", "train", "test"]]): A function/transform that takes in the data and transforms it.
          Supports "auto", "tt", "train", "test", and None (default).
          - "auto": Automatically initializes the transform based on the dataset type and `split`.
          - "tt": Converts data into a tensor image.
          - "train": Transform to use in the train stage.
          - "test": Transform to use in the test stage.
          - None (default): No transformation is applied.
        - `target_transform` (Union[Optional[Callable], Literal["auto"]]): A function/transform that takes in the target and transforms it.
          Supports "auto", and None (default).
          - "auto": Automatically initializes the transform based on the dataset type and `split`.
          - None (default): No transformation is applied.
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
          If the dataset is already downloaded, it is not downloaded again.

    ### Attributes
        - `root` (Path): Root directory where the dataset exists or will be saved to.
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
    >>> dataset = CIFAR10("path/to/dataset", split="train", transform="auto")
    >>> len(dataset)  # Get dataset size
    50000
    >>> data, target = dataset[0]  # Get first sample and target
    ```
    """

    _urls = [
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    ]
    _zip_list = [
        ("cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a"),
    ]
    _train_list = [
        ("cifar-10-batches-py/data_batch_1", "c99cafc152244af753f735de768cd75f"),
        ("cifar-10-batches-py/data_batch_2", "d4bba439e000b95fd0a9bffe97cbabec"),
        ("cifar-10-batches-py/data_batch_3", "54ebc095f3ab1f0389bbae665268c751"),
        ("cifar-10-batches-py/data_batch_4", "634d18415352ddfa80567beed471001a"),
        ("cifar-10-batches-py/data_batch_5", "482c414d41f54cd18b22e5b47cb7c3cb"),
    ]
    _test_list = [
        ("cifar-10-batches-py/test_batch", "40351d587109b95175f43aff81a1287e"),
    ]
    _meta_list = [
        ("cifar-10-batches-py/batches.meta", "5ff9c542aee3614f3951f8cda6e48888"),
    ]
    MEAN = [0.4914, 0.4822, 0.4465]
    STD = [0.2470, 0.2435, 0.2616]
    NORMALIZE = Normalize(MEAN, STD)
    DENORMALIZE = DeNormalize(MEAN, STD)

    def __init__(
        self,
        root: os.PathLike = "cifar10",
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
        target_transform: Union[
            Optional[Callable],
            Literal["auto",],
        ] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        - `root` (os.PathLike): Root directory where the dataset exists or will be saved to.
        - `split` (Literal["train", "test"]): The dataset split; supports "train" (default), and "test".
        - `transform` (Union[Optional[Callable], Literal["auto", "tt", "train", "test"]]): A function/transform that takes in the data and transforms it.
          Supports "auto", "tt", "train", "test", and None (default).
          - "auto": Automatically initializes the transform based on the dataset type and `split`.
          - "tt": Converts data into a tensor image.
          - "train": Transform to use in the train stage.
          - "test": Transform to use in the test stage.
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

        # checklist
        self._checklist = []
        self._checklist += self._zip_list
        if split == "train":
            self._checklist += self._train_list
        elif split == "test":
            self._checklist += self._test_list
        else:
            raise ValueError(f"split must be one of ['train', 'test'], got {split}")

        # transform
        if self.transform == "auto" and split == "train" or self.transform == "train":
            kwargs.update({"mean": self.MEAN, "std": self.STD})
            self.transform = ClassificationPresetTrain(**kwargs)
        elif self.transform == "auto" and split == "test" or self.transform == "test":
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
            self._download(
                urls=self._urls,
                checklist=self._checklist + self._meta_list,
                extract_dir="",
            )

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        self.data = []
        self.targets = []

        # load data/targets
        for fname, _ in self._checklist[1:]:
            fpath = os.path.join(self.root, fname)

            with open(fpath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # BCHW
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC

        # load labels
        labels_fname, _ = self._meta_list[0]
        labels_fpath = os.path.join(self.root, labels_fname)

        with open(labels_fpath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.labels = data["label_names"]

    def _preprocess_data(
        self,
        data: np.ndarray,
    ) -> Image.Image:
        return Image.fromarray(data)
