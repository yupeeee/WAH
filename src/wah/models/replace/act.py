from torch import nn

from ...typing import (
    Module,
)
from .utils import ReplaceModule

__all__ = [
    "relu2gelu",
    "gelu2relu",
]

kws = {
    "relu": [
        "inplace",
        None,
    ],
    "gelu": [
        None,
        "approximate",
    ],
}


class ReLU2GELU(ReplaceModule):
    target_module_name = "ReLU"
    keymaps = [k for k in zip(kws["relu"], kws["gelu"])]
    _replacement_module = nn.GELU

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, module: Module, use_cuda: bool = None):
        gelu = self.replacement_module(module, use_cuda)
        gelu.approximate = "none"

        return gelu


class GELU2ReLU(ReplaceModule):
    target_module_name = "GELU"
    keymaps = [k for k in zip(kws["gelu"], kws["relu"])]
    _replacement_module = nn.ReLU

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, module: Module, use_cuda: bool = None):
        relu = self.replacement_module(module, use_cuda)
        relu.inplace = True

        return relu


relu2gelu = ReLU2GELU()
gelu2relu = GELU2ReLU()
