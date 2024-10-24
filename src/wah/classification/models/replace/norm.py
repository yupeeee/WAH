from torch import nn

from ....module import get_module_name
from ....typing import Module
from .utils import ReplaceModule
from .wrapper import PermuteWrapper

__all__ = ["bn2ln", "ln2bn", "ln2bnx"]

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

    def __init__(self, track_running_stats=True) -> None:
        super().__init__()

        self.track_running_stats = track_running_stats

    def __call__(self, module: Module, use_cuda: bool = False):
        module_name = get_module_name(module)
        bn = self.replacement_module(
            module,
            use_cuda,
            track_running_stats=self.track_running_stats,
        )

        if module_name == "LayerNorm":
            return PermuteWrapper(
                module=bn,
                dims=(0, 3, 1, 2),
                unsqueeze_dim=1,
            )

        elif module_name == "LayerNorm2d":
            return bn

        else:
            raise


bn2ln = BN2LN()
ln2bn = LN2BN(track_running_stats=True)
ln2bnx = LN2BN(track_running_stats=False)
