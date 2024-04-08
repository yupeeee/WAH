from torch import nn

from ...typing import Module
from ..modules import get_module_name
from .misc import ReplaceModule
from .wrapper import PermuteWrapper

__all__ = [
    "bn2ln",
    "ln2bn",
]

kws = {
    "bn": [
        "num_features",
        "eps",
        "momentum",
        "affine",
        "track_running_stats",
    ],
    "ln": [
        "normalized_shape",
        None,
        None,
        "elementwise_affine",
        None,
    ],
}


class BN2LN(ReplaceModule):
    target_module_name = "BatchNorm2d"
    keymaps = [k for k in zip(kws["bn"], kws["ln"])]
    _replacement_module = nn.LayerNorm

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, module: Module, use_cuda: bool = False):
        ln = self.replacement_module(module, use_cuda)

        return PermuteWrapper(ln, (0, 2, 3, 1))


class LN2BN(ReplaceModule):
    target_module_name = [
        "LayerNorm",
        "LayerNorm2d",
    ]
    keymaps = [k for k in zip(kws["ln"], kws["bn"])]
    _replacement_module = nn.BatchNorm2d

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, module: Module, use_cuda: bool = False):
        module_name = get_module_name(module)
        bn = self.replacement_module(module, use_cuda)

        if module_name == "LayerNorm":
            return PermuteWrapper(bn, (0, 3, 1, 2))

        elif module_name == "LayerNorm2d":
            return bn

        else:
            raise


bn2ln = BN2LN()
ln2bn = LN2BN()
