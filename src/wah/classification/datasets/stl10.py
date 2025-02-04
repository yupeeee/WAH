import numpy as np
from PIL import Image
from torchvision.transforms import Normalize

from ...misc import path as _path
from ...misc.typing import Callable, Literal, Optional, Path, Union
from .base import ClassificationDataset
from .transforms import ClassificationPresetEval, ClassificationPresetTrain, DeNormalize

__all__ = [
    "STL10",
]


class STL10(ClassificationDataset):
    """[STL-10](https://ai.stanford.edu/~acoates/stl10/) dataset.

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
    >>> dataset = STL10("path/to/dataset", split="train", transform="auto")
    >>> len(dataset)  # Get dataset size
    50000
    >>> data, target = dataset[0]  # Get first sample and target
    ```
    """

    URLS = [
        "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz",
    ]
    ROOT = _path.clean("./datasets/stl10")
    ZIP_LIST = [
        ("stl10_binary.tar.gz", "91f7769df0f17e558f3565bffb0c7dfb"),
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
    ]
    MEAN = [0.4467, 0.4398, 0.4066]
    STD = [0.2603, 0.2566, 0.2713]
    NORMALIZE = Normalize(MEAN, STD)
    DENORMALIZE = DeNormalize(MEAN, STD)

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
        target_transform: Union[Optional[Callable], Literal["auto"]] = None,
        download: bool = False,
        **kwargs,
    ) -> None:
        """
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
        """
        super().__init__(
            root,
            transform,
            target_transform,
        )

        self.checklist = []

        self.checklist += self.ZIP_LIST

        if split == "train":
            self.checklist += self.TRAIN_LIST

        elif split == "test":
            self.checklist += self.TEST_LIST

        else:
            raise ValueError(f"split must be one of ['train', 'test'], got {split}")

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

        if self.target_transform == "auto":
            self.target_transform = None
        else:
            pass

        if download:
            self._download(
                urls=self.URLS,
                checklist=self.checklist + self.META_LIST,
                extract_dir=".",
            )

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        # load data
        data_fname, _ = self.checklist[1]
        data_fpath = _path.join(self.root, data_fname)

        with open(data_fpath, "rb") as f:
            self.data = np.fromfile(f, dtype=np.uint8)
            self.data = self.data.reshape((-1, 3, 96, 96))  # BCWH
            self.data = self.data.transpose((0, 3, 2, 1))  # convert to BHWC

        # load targets
        targets_fname, _ = self.checklist[2]
        targets_fpath = _path.join(self.root, targets_fname)

        with open(targets_fpath, "rb") as f:
            self.targets = [target - 1 for target in np.fromfile(f, dtype=np.uint8)]

        # load labels
        labels_fname, _ = self.META_LIST[0]
        labels_fpath = _path.join(self.root, labels_fname)

        with open(labels_fpath, "r") as f:
            self.labels = [label for label in f.read().split("\n") if len(label)]

    def _preprocess_data(
        self,
        data: np.ndarray,
    ) -> Image.Image:
        return Image.fromarray(data)
