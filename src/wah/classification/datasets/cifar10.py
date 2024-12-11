import pickle

import numpy as np
from PIL import Image
from torchvision.transforms import Normalize

from ... import path as _path
from ...typing import Callable, Literal, Optional, Path, Union
from .base import ClassificationDataset
from .transforms import ClassificationPresetEval, ClassificationPresetTrain, DeNormalize

__all__ = [
    "CIFAR10",
]


class CIFAR10(ClassificationDataset):
    """
    [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset.

    ### Attributes
    - `root` (Path): Root directory where the dataset exists or will be saved to.
    - `transform` (Callable, optional): A function/transform that takes in the data and transforms it. Defaults to None.
    - `target_transform` (Callable, optional): A function/transform that takes in the target and transforms it. Defaults to None.
    - `data`: Data of the dataset.
    - `targets`: Targets of the dataset.
    - `labels`: Labels of the dataset.
    - `MEAN` (list): Mean of dataset; [0.4914, 0.4822, 0.4465].
    - `STD` (list): Standard deviation of dataset; [0.2470, 0.2435, 0.2616].
    - `NORMALIZE` (callable): Transform for dataset normalization.
    - `DENORMALIZE` (callable): Transform for dataset denormalization.

    ### Methods
    - `__getitem__(index) -> Tuple[Any, Any]`: Returns (data, target) of dataset using the specified index.
    - `__len__() -> int`: Returns the size of the dataset.
    - `set_return_data_only() -> None`: Sets the flag to return only data without targets.
    - `unset_return_data_only() -> None`: Unsets the flag to return only data without targets.
    - `set_return_w_index() -> None`: Sets the flag to return data with index.
    - `unset_return_w_index() -> None`: Unsets the flag to return data with index.

    ### Example
    ```python
    import wah

    dataset = wah.datasets.CIFAR10(root="path/to/dataset")
    data, target = dataset[0]
    num_data = len(dataset)
    ```
    """

    URLS = [
        "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz",
    ]
    ROOT = _path.clean("./datasets/cifar10")

    ZIP_LIST = [
        ("cifar-10-python.tar.gz", "c58f30108f718f92721af3b95e74349a"),
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
        target_transform: Union[
            Optional[Callable],
            Literal["auto",],
        ] = None,
        return_data_only: Optional[bool] = False,
        return_w_index: Optional[bool] = False,
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the CIFAR-10 dataset.

        ### Parameters
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
        - `return_data_only` (Optional[bool]): Whether to return only data without targets. Defaults to False.
        - `return_w_index` (Optional[bool]): Whether to return data with index. Defaults to False.
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
          If the dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root,
            transform,
            target_transform,
            return_data_only,
            return_w_index,
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
                ext_dir_name=".",
            )

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        """
        Initializes the CIFAR-10 dataset.

        ### Returns
        - `None`
        """
        self.data = []
        self.targets = []

        # load data/targets
        for fname, _ in self.checklist[1:]:
            fpath = _path.join(self.root, fname)

            with open(fpath, "rb") as f:
                entry = pickle.load(f, encoding="latin1")

                self.data.append(entry["data"])
                self.targets.extend(entry["labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)  # BCHW
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to BHWC

        # load labels
        labels_fname, _ = self.META_LIST[0]
        labels_fpath = _path.join(self.root, labels_fname)

        with open(labels_fpath, "rb") as f:
            data = pickle.load(f, encoding="latin1")
            self.labels = data["label_names"]

    def _preprocess_data(
        self,
        data: np.ndarray,
    ) -> Image.Image:
        """
        Preprocesses the data.

        ### Parameters
        - `data` (np.ndarray): The data to preprocess.

        ### Returns
        - `Image.Image`: The preprocessed data.
        """
        return Image.fromarray(data)
