import shutil

import tqdm
from PIL import Image
from torchvision.transforms import Normalize

from ... import path as _path
from ...typing import Callable, Literal, Optional, Path, Union
from ...utils.zip import extract
from .base import ClassificationDataset
from .ILSVRC2012_meta import _ilsvrc2012_meta
from .labels import imagenet1k as labels
from .transforms import ClassificationPresetEval, ClassificationPresetTrain, DeNormalize

__all__ = [
    "ImageNet",
]


class ImageNetTrain(ClassificationDataset):
    """
    [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) train dataset.

    ### Attributes
    - `root` (Path): Root directory where the dataset exists or will be saved to.
    - `transform` (Callable, optional): A function/transform that takes in the data and transforms it. If None, no transformation is performed. Defaults to None.
    - `target_transform` (Callable, optional): A function/transform that takes in the target and transforms it. If None, no transformation is performed. Defaults to None.
    - `data`: Data of the dataset.
    - `targets`: Targets of the dataset.
    - `labels`: Labels of the dataset.
    - `MEAN` (list): Mean of dataset; [0.485, 0.456, 0.406].
    - `STD` (list): Standard deviation of dataset; [0.229, 0.224, 0.225].
    - `NORMALIZE` (callable): Transform for dataset normalization.
    - `DENORMALIZE` (callable): Transform for dataset denormalization.

    ### Methods
    - `__getitem__(index) -> Tuple[Any, Any]`: Returns (data, target) of dataset using the specified index.
    - `__len__() -> int`: Returns the size of the dataset.
    - `set_return_data_only() -> None`: Sets the flag to return only data without targets.
    - `unset_return_data_only() -> None`: Unsets the flag to return only data without targets.
    - `set_return_w_index() -> None`: Sets the flag to return data with index.
    - `unset_return_w_index() -> None`: Unsets the flag to return data with index.
    """

    URLS = [
        "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar",
    ]
    ROOT = _path.clean("./datasets/imagenet")

    ZIP_LIST = [
        ("ILSVRC2012_img_train.tar", "1d675b47d978889d74fa0da5fadfb00e"),
    ]

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NORMALIZE = Normalize(MEAN, STD)
    DENORMALIZE = DeNormalize(MEAN, STD)

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
        target_transform: Union[Optional[Callable], Literal["auto"]] = None,
        return_data_only: Optional[bool] = False,
        return_w_index: Optional[bool] = False,
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the ImageNet train dataset.

        ### Parameters
        - `root` (Path): Root directory where the dataset exists or will be saved to.
        - `transform` (Union[Optional[Callable], Literal["auto", "train", "val", "tt"]]): A function/transform that takes in the data and transforms it.
          Supports "auto", "train", "val", "tt", and None (default).
          - "auto": Automatically initializes the transform based on the dataset type and `split`.
          - "train": Transform to use in the train stage.
          - "val": Transform to use in the validation stage.
          - "tt": Converts data into a tensor image.
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

        self.checklist = self.ZIP_LIST

        if self.transform == "auto" or self.transform == "train":
            kwargs.update({"mean": self.MEAN, "std": self.STD})
            self.transform = ClassificationPresetTrain(**kwargs)
        elif self.transform == "val":
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
            # download train dataset
            self._download(
                urls=self.URLS,
                checklist=self.checklist,
                ext_dir_name="train",
            )

            # extract class folders
            data_root = _path.join(self.root, "train")
            classes = _path.ls(data_root, fext=".tar", sort=True)

            if len(classes):
                classes = [c.split(".tar")[0] for c in classes]

                for c in classes:
                    fpath = _path.join(data_root, f"{c}.tar")
                    extract(fpath, save_dir=_path.join(data_root, c))

                    # delete zipfiles
                    _path.rmfile(fpath)

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        """
        Initializes the ImageNet train dataset.

        ### Returns
        - `None`
        """
        # load class folders
        data_root = _path.join(self.root, "train")
        classes = _path.ls(
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
                desc="Initializing ImageNet train dataset...",
            )
        ):
            class_dir = _path.join(data_root, c)
            fpaths = _path.ls(path=class_dir, fext=".JPEG", sort=True)

            path_to_images = [_path.join(data_root, c, fpath) for fpath in fpaths]

            self.data += path_to_images
            self.targets += [class_idx] * len(fpaths)

        # load labels
        self.labels = labels

    def _preprocess_data(
        self,
        fpath: Path,
    ) -> Image.Image:
        """
        Preprocesses the data.

        ### Parameters
        - `fpath` (Path): The path to the image file.

        ### Returns
        - `Image.Image`: The preprocessed data.
        """
        return Image.open(fpath).convert("RGB")


class ImageNetVal(ClassificationDataset):
    """
    [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) validation dataset.

    ### Attributes
    - `root` (Path): Root directory where the dataset exists or will be saved to.
    - `transform` (Callable, optional): A function/transform that takes in the data and transforms it. If None, no transformation is performed. Defaults to None.
    - `target_transform` (Callable, optional): A function/transform that takes in the target and transforms it. If None, no transformation is performed. Defaults to None.
    - `data`: Data of the dataset.
    - `targets`: Targets of the dataset.
    - `labels`: Labels of the dataset.
    - `MEAN` (list): Mean of dataset; [0.485, 0.456, 0.406].
    - `STD` (list): Standard deviation of dataset; [0.229, 0.224, 0.225].
    - `NORMALIZE` (callable): Transform for dataset normalization.

    ### Methods
    - `__getitem__(index) -> Tuple[Any, Any]`: Returns (data, target) of dataset using the specified index.
    - `__len__() -> int`: Returns the size of the dataset.
    - `set_return_data_only() -> None`: Sets the flag to return only data without targets.
    - `unset_return_data_only() -> None`: Unsets the flag to return only data without targets.
    - `set_return_w_index() -> None`: Sets the flag to return data with index.
    - `unset_return_w_index() -> None`: Unsets the flag to return data with index.
    """

    URLS = [
        "https://www.image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar",
    ]
    ROOT = _path.clean("./datasets/imagenet")

    ZIP_LIST = [
        ("ILSVRC2012_img_val.tar", "29b22e2961454d5413ddabcf34fc5622"),
    ]

    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    NORMALIZE = Normalize(MEAN, STD)

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
        target_transform: Union[Optional[Callable], Literal["auto"]] = None,
        return_data_only: Optional[bool] = False,
        return_w_index: Optional[bool] = False,
        download: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the ImageNet validation dataset.

        ### Parameters
        - `root` (Path): Root directory where the dataset exists or will be saved to.
        - `transform` (Union[Optional[Callable], Literal["auto", "tt"]]): A function/transform that takes in the data and transforms it.
          Supports "auto", "tt", and None (default).
          - "auto": Automatically initializes the transform based on the dataset type and `split`.
          - "tt": Converts data into a tensor image.
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

        self.checklist = self.ZIP_LIST

        if self.transform == "auto":
            kwargs.update({"mean": self.MEAN, "std": self.STD})
            self.transform = ClassificationPresetEval(**kwargs)
        elif self.transform == "tt":
            self.transform = ClassificationPresetEval(**kwargs)
        elif self.transform == "train":
            raise ValueError(
                "Train transformations cannot be applied to the validation dataset."
            )
        else:
            pass

        if self.target_transform == "auto":
            self.target_transform = None
        else:
            pass

        if download:
            # download validation dataset
            self._download(
                urls=self.URLS[:1],
                checklist=self.checklist[:1],
                ext_dir_name="val",
            )
            # parse
            self._parse_val_archive()

        self._initialize()

    def _parse_val_archive(
        self,
    ) -> None:
        image_paths = _path.ls(
            path=_path.join(self.root, "val"),
            fext=".JPEG",
            sort=True,
            absolute=True,
        )

        wnids = _ilsvrc2012_meta

        for wnid in set(wnids):
            _path.mkdir(_path.join(self.root, "val", wnid))

        for wnid, image_path in zip(wnids, image_paths):
            shutil.move(
                image_path,
                _path.join(self.root, "val", wnid, _path.basename(image_path)),
            )

    def _initialize(
        self,
    ) -> None:
        """
        Initializes the ImageNet validation dataset.

        ### Returns
        - `None`
        """
        # load class folders
        data_root = _path.join(self.root, "val")
        classes = _path.ls(
            path=data_root,
            fext="dir",
            sort=True,
        )

        if len(classes) != 1000:
            self._parse_val_archive()
            classes = _path.ls(
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
                desc="Initializing ImageNet val dataset...",
            )
        ):
            class_dir = _path.join(data_root, c)
            fpaths = _path.ls(path=class_dir, fext=".JPEG", sort=True)

            path_to_images = [_path.join(data_root, c, fpath) for fpath in fpaths]

            self.data += path_to_images
            self.targets += [class_idx] * len(fpaths)

        # load labels
        self.labels = labels

    def _preprocess_data(
        self,
        fpath: Path,
    ) -> Image.Image:
        """
        Preprocesses the data.

        ### Parameters
        - `fpath` (Path): The path to the image file.

        ### Returns
        - `Image.Image`: The preprocessed data.
        """
        return Image.open(fpath).convert("RGB")


def ImageNet(
    root: Path = "./datasets/imagenet",
    split: Literal[
        "train",
        "val",
    ] = "train",
    transform: Union[
        Optional[Callable],
        Literal[
            "auto",
            "tt",
            "train",
            "val",
        ],
    ] = None,
    target_transform: Union[Optional[Callable], Literal["auto"]] = None,
    return_data_only: Optional[bool] = False,
    return_w_index: Optional[bool] = False,
    download: bool = False,
    **kwargs,
) -> Union[ImageNetTrain, ImageNetVal]:
    """
    [ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset.

    ### Parameters
    - `root` (Path): Root directory where the dataset exists or will be saved to.
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
    - `return_data_only` (Optional[bool]): Whether to return only data without targets. Defaults to False.
    - `return_w_index` (Optional[bool]): Whether to return data with index. Defaults to False.
    - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
      If the dataset is already downloaded, it is not downloaded again.

    ### Returns
    - `Union[ImageNetTrain, ImageNetVal]`: ImageNet train/validation dataset.

    ### Example
    ```python
    import wah

    dataset = wah.datasets.ImageNet(root="path/to/dataset")
    data, target = dataset[0]
    num_data = len(dataset)
    ```
    """
    if split == "train":
        return ImageNetTrain(
            root=root,
            transform=transform,
            target_transform=target_transform,
            return_data_only=return_data_only,
            return_w_index=return_w_index,
            download=download,
            **kwargs,
        )

    elif split == "val":
        return ImageNetVal(
            root=root,
            transform="auto" if transform == "val" else transform,
            target_transform=target_transform,
            return_data_only=return_data_only,
            return_w_index=return_w_index,
            download=download,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported dataset split: {split}")
