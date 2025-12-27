from typing import List, Literal, Optional, Union

import torch
from PIL.Image import Image

from .hintersdorf_nips2024 import hintersdorf_nips2024 as _hintersdorf_nips2024
from .jain_cvpr2025 import jain_cvpr2025_dynamic as _jain_cvpr2025_dynamic
from .ren_eccv2024 import ren_eccv2024 as _ren_eccv2024
from .wen_iclr2024 import wen_iclr2024 as _wen_iclr2024

__all__ = [
    "MemorizationMitigator",
]


class MemorizationMitigator:
    def __init__(
        self,
        pipe,
        strategy: Literal[
            "wen_iclr2024",
            "ren_eccv2024",
            "hintersdorf_nips2024",
            "jain_cvpr2025",
        ] = "jain_cvpr2025",
    ) -> None:
        self.pipe = pipe
        self.strategy = strategy
        self.strategies = {
            "wen_iclr2024": _wen_iclr2024,
            "ren_eccv2024": _ren_eccv2024,
            "hintersdorf_nips2024": _hintersdorf_nips2024,
            "jain_cvpr2025": _jain_cvpr2025_dynamic,
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
