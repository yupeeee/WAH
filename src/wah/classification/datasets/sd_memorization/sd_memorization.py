"""
[Memorized]
Detecting, Explaining, and Mitigating Memorization in Diffusion Models
Yuxin Wen, Yuchen Liu, Chen Chen, Lingjuan Lyu
https://github.com/YuxinWenRick/diffusion_memorization/blob/main/examples/sdv1_500_memorized.jsonl

[Normal]
https://github.com/ml-research/localizing_memorization_in_diffusion_models/tree/main/prompts
"""

import json

import requests
from PIL import Image

from ....misc import path as _path
from ....misc.typing import Image as ImageType
from ....misc.typing import List, Path, Tuple
from ..base import ClassificationDataset

__all__ = [
    "SDM1K",
]


def _load_json(
    path: Path,
) -> List[Tuple[str, str, int]]:
    """Load JSON file containing prompts.

    ### Args
        - `path` (Path): Path to JSON file.

    ### Returns
        - List[Tuple[str, str, int]]: List of tuples containing (caption, url, index).
    """
    prompts = []
    with open(path, "r") as f:
        for line in f:
            entry = json.loads(line)
            prompt = (entry["caption"], entry["url"], int(entry["index"]))
            prompts.append(prompt)
    return prompts


class SDM1K(ClassificationDataset):
    """Stable Diffusion Memorization 1K Dataset.

    ### Args
        - `root` (Path): Root directory where the dataset exists or will be saved to.
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
          If the dataset is already downloaded, it is not downloaded again.

    ### Attributes
        - `root` (Path): Root directory where the dataset exists or will be saved to.
        - `data`: Data of the dataset.
        - `targets`: Targets of the dataset.
        - `labels`: Labels of the dataset.

    ### Example
    ```python
    >>> dataset = SDM1K("path/to/dataset", download=True)
    >>> len(dataset)  # Get dataset size
    1000
    >>> caption, image, index = dataset[0]  # Get first sample
    ```
    """

    URLS = [
        "https://raw.githubusercontent.com/yupeeee/WAH/main/src/wah/classification/datasets/sd_memorization/sdv1_500_memorized.jsonl",
        "https://raw.githubusercontent.com/yupeeee/WAH/main/src/wah/classification/datasets/sd_memorization/sdv1_500_normal.jsonl",
    ]
    ROOT = _path.clean("./datasets/sdm1k")

    JSON_LIST = [
        ("sdv1_500_memorized.jsonl", "6526a7f82f6b17e37ea379e14711a53a"),
        ("sdv1_500_normal.jsonl", "038834d49e716cc9a06014c9dbd9bcf0"),
    ]

    def __init__(
        self,
        root: Path = ROOT,
        download: bool = False,
    ) -> None:
        """
        - `root` (Path): Root directory where the dataset exists or will be saved to.
        - `download` (bool): If True, downloads the dataset from the internet and puts it into the `root` directory.
          If the dataset is already downloaded, it is not downloaded again.
        """
        super().__init__(
            root,
            None,
            None,
        )

        self.checklist = self.JSON_LIST

        if download:
            self._download(
                urls=self.URLS,
                checklist=self.checklist,
                extract_dir=".",
            )

        self._initialize()

    def _initialize(
        self,
    ) -> None:
        self.data = []
        self.targets = []

        # load data/targets
        for i, (fname, _) in enumerate(self.checklist):
            fpath = _path.join(self.root, fname)
            prompts = _load_json(fpath)
            self.data.extend(prompts)
            self.targets.extend([i] * len(prompts))

        # load labels
        self.labels = ["memorized", "normal"]

    def _preprocess_data(
        self,
        data: Tuple[str, str, int],
        # ) -> Tuple[str, ImageType, int]:
    ) -> str:
        caption, url, index = data
        # image = Image.open(requests.get(url, stream=True).raw)
        # return caption, image, index
        return caption
