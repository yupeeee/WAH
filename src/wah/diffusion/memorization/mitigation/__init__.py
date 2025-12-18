from typing import List, Literal, Optional, Union

import torch
from PIL.Image import Image

from .jain_cvpr2025 import jain_cvpr2025_dynamic as _jain_cvpr2025_dynamic
from .jain_cvpr2025 import jain_cvpr2025_static as _jain_cvpr2025_static
from .ren_eccv2024 import ren_eccv2024 as _ren_eccv2024

__all__ = [
    "MemorizationMitigator",
]


class MemorizationMitigator:
    def __init__(
        self,
        pipe,
        strategy: Literal[
            "ren_eccv2024",
            "jain_cvpr2025_static",
            "jain_cvpr2025_dynamic",
        ] = "jain_cvpr2025_dynamic",
    ) -> None:
        self.pipe = pipe
        self.strategy = strategy
        self.strategies = {
            "ren_eccv2024": _ren_eccv2024,
            "jain_cvpr2025_static": _jain_cvpr2025_static,
            "jain_cvpr2025_dynamic": _jain_cvpr2025_dynamic,
        }

    def to(self, device: Optional[Union[str, torch.device]]) -> "MemorizationMitigator":
        self.pipe.to(device)
        return self

    def __call__(
        self,
        prompt: List[str],
        seed: Optional[Union[int, List[int]]] = None,
        verbose: bool = False,
        **kwargs,
    ) -> List[Image]:
        return self.strategies[self.strategy](
            prompt, self.pipe, seed, verbose, **kwargs
        )
