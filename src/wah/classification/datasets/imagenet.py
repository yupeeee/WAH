import shutil

import tqdm
from PIL import Image
from torchvision.transforms import Normalize

from ... import utils
from ...misc import path as _path
from ...misc.typing import Callable, Literal, Optional, Path, Union
from .base import ClassificationDataset
from .ILSVRC2012_meta import _ilsvrc2012_meta
from .labels import imagenet1k as labels
from .transforms import ClassificationPresetEval, ClassificationPresetTrain, DeNormalize

__all__ = [
    "ImageNet",
]


class ImageNetTrain(ClassificationDataset):
    """[ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset.

    ### Args
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
    >>> dataset = ImageNet("path/to/dataset", split="train", transform="auto")
    >>> len(dataset)  # Get dataset size
    1281167
    >>> data, target = dataset[0]  # Get first sample and target
    ```
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
        download: bool = False,
        **kwargs,
    ) -> None:
        """
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
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
        If the dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root,
            transform,
            target_transform,
        )

        # checklist
        self.checklist = self.ZIP_LIST

        # transform
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

        # target_transform
        if self.target_transform == "auto":
            self.target_transform = None
        else:
            pass

        # download
        if download:
            # download train dataset
            self._download(
                urls=self.URLS,
                checklist=self.checklist,
                extract_dir="train",
            )

            # extract class folders
            data_root = _path.join(self.root, "train")
            classes = _path.ls(data_root, fext=".tar", sort=True)

            if len(classes):
                classes = [c.split(".tar")[0] for c in classes]

                for c in classes:
                    fpath = _path.join(data_root, f"{c}.tar")
                    utils.zips.extract(fpath, save_dir=_path.join(data_root, c))

                    # delete zipfiles
                    _path.rmfile(fpath)

        self._initialize()

    def _initialize(
        self,
    ) -> None:
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
        return Image.open(fpath).convert("RGB")


class ImageNetVal(ClassificationDataset):
    """[ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset.

    ### Args
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
    >>> dataset = ImageNet("path/to/dataset", split="train", transform="auto")
    >>> len(dataset)  # Get dataset size
    1281167
    >>> data, target = dataset[0]  # Get first sample and target
    ```
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
        download: bool = False,
        **kwargs,
    ) -> None:
        """
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
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
        If the dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root,
            transform,
            target_transform,
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
                extract_dir="val",
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
    download: bool = False,
    **kwargs,
) -> Union[ImageNetTrain, ImageNetVal]:
    """[ImageNet](https://www.image-net.org/challenges/LSVRC/2012/index.php) dataset.

    ### Args
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
    >>> dataset = ImageNet("path/to/dataset", split="train", transform="auto")
    >>> len(dataset)  # Get dataset size
    1281167
    >>> data, target = dataset[0]  # Get first sample and target
    ```
    """
    if split == "train":
        return ImageNetTrain(
            root=root,
            transform=transform,
            target_transform=target_transform,
            download=download,
            **kwargs,
        )

    elif split == "val":
        return ImageNetVal(
            root=root,
            transform="auto" if transform == "val" else transform,
            target_transform=target_transform,
            download=download,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported dataset split: {split}")
