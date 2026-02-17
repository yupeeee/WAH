import os
from typing import Literal

from .base import _WebsterArXiv2023

__all__ = [
    "WebsterArXiv2023",
]

CONFIG = {
    "sdv1": {
        "url": "https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/sdv1_bb_edge_groundtruth.parquet",
        "checksum": "72c2dded275dda496de9c3f7c33c4f3d",
    },
    "sdv2": {
        "url": "https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/sdv2_bb_edge_groundtruth.parquet",
        "checksum": "c700ae1d156a9330940d339a24740ed1",
    },
    # "midjourney": {
    #     "url": "https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/midjourney_groundtruth.parquet",
    #     "checksum": "0c4329ae105832ce94d032e2c3012f25",
    # },
    "deep_if": {
        "url": "https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/deep_if_sdv1_wb_groundtruth.parquet",
        "checksum": "b21661ba27499b23cde515988292266e",
    },
    "realvis": {
        "url": "https://huggingface.co/datasets/fraisdufour/templates-verbs/resolve/main/groundtruth_parquets/realistic_vision_sdv1_edge_groundtruth.parquet",
        "checksum": "d4ce3781e41efb78289c936d87a79edf",
    },
}


class WebsterArXiv2023(_WebsterArXiv2023):
    """
    # A Reproducible Extraction of Training Images from Diffusion Models
    ### Ryan Webster, arXiv 2023

    Paper: https://arxiv.org/abs/2305.08694
    GitHub: https://github.com/ryanwebster90/onestep-extraction
    """

    def __init__(
        self,
        root: os.PathLike,
        pipe: Literal[
            "sdv1",
            "sdv2",
            "deep_if",
            "realvis",
        ] = "sdv1",
    ) -> None:
        super().__init__(
            root=root,
            url=CONFIG[pipe]["url"],
            checksum=CONFIG[pipe]["checksum"],
        )
