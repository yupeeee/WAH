import torch

from ...typing import (
    Module,
    Optional,
)
from ..modules import _getattr, get_attrs
from .misc import replace_module

__all__ = [
    "Replacer",
]


class Replacer:
    replacements = {
        "act": [
            ("relu", "gelu"),
            ("gelu", "relu"),
        ],
        "attn": [
            ("attn", "pattn"),
        ],
        "norm": [
            ("bn", "ln"),
            ("ln", "bn"),
        ],
    }

    def __init__(
        self,
        target: str,
        to: str,
        test_replacement: Optional[torch.Tensor] = None,
    ) -> None:
        assert (target, to) in [
            item for sublist in self.replacements.values() for item in sublist
        ], f"{target}->{to} not supported."

        self.target = target
        self.to = to
        self.test_replacement = test_replacement

    def __call__(
        self,
        model: Module,
    ):
        use_cuda = next(model.parameters()).is_cuda

        if (self.target, self.to) in self.replacements["act"]:
            from . import act as lib

            use_cuda = None

        elif (self.target, self.to) in self.replacements["attn"]:
            from . import attn as lib

        elif (self.target, self.to) in self.replacements["norm"]:
            from . import norm as lib

        else:
            raise ValueError

        attrs = get_attrs(model)

        for attr in attrs:
            # check if attribute is valid
            try:
                module = _getattr(model, attr)

            except AttributeError:
                continue

            # replace targets only
            try:
                replace_module(
                    model=model,
                    attr=attr,
                    replace_with=getattr(lib, f"{self.target}2{self.to}")(
                        module,
                        use_cuda=use_cuda,
                    ),
                    test_replacement=None,
                )

            except AssertionError:
                continue

        if self.test_replacement is not None:
            try:
                model = model.eval()
                _ = model(self.test_replacement)

            except BaseException:
                raise

        return model
