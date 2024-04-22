from ...typing import (
    Module,
)
from .misc import ReplaceModule, get_module_name
from .wrapper import Conv1x1Wrapper

__all__ = [
    "attn2pattn",
]


class ATTN2PATTN(ReplaceModule):
    target_module_name = [
        "MultiheadAttention",
        "ShiftedWindowAttention",
    ]
    keymaps = None
    _replacement_module = None

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, module: Module, use_cuda: bool = False):
        assert get_module_name(module) in self.target_module_name

        channel_dim = int(str(module).split("in_features=")[1].split(",")[0])

        attn = module
        pattn = Conv1x1Wrapper(
            module=attn,
            channel_dim=channel_dim,
            permute_dims=(0, 3, 1, 2),
        )

        return pattn

    def replacement_module(self, module: Module, use_cuda: bool = False) -> Module:
        raise NotImplementedError


attn2pattn = ATTN2PATTN()
