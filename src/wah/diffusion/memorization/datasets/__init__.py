import os
from typing import Literal

from . import realvis, sdv1, sdv2
from .base import PromptDataset

__all__ = [
    "MemorizedPrompts",
]

URLS = {
    "sdv1": sdv1.URL,
    "sdv2": sdv2.URL,
    "realvis": realvis.URL,
}
CHECKSUMS = {
    "sdv1": sdv1.CHECKSUM,
    "sdv2": sdv2.CHECKSUM,
    "realvis": realvis.CHECKSUM,
}


class MemorizedPrompts(PromptDataset):
    def __init__(
        self,
        root: os.PathLike = "memorized_prompts",
        pipe: Literal["sdv1", "sdv2", "realvis"] = "sdv1",
    ) -> None:
        super().__init__(
            root=root,
            url=URLS[pipe],
            checksum=CHECKSUMS[pipe],
        )
