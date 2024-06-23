from ...typing import (
    Module,
    Optional,
    Tensor,
)
from ..modules import _getattr, get_attrs
from .utils import replace_module

__all__ = [
    "Replacer",
]


class Replacer:
    """
    A class to replace specific types of layers in a neural network model.

    ### Attributes
    - `replacements` (dict):
      A dictionary mapping types of layers to their possible replacements.

      [**Supported replacements**]() (target -> to):
      - "relu" (ReLU) <-> "gelu" (GELU)
      - "attn" (Self-Attention) -> "pattn" (Conv1x1 + Self-Attention)
      - "bn" (BatchNorm) <-> "ln" (LayerNorm)

    ### Methods
    - `__call__`:
        Replaces specified layers in the model and optionally tests the replacement.
    """

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
        test_replacement: Optional[Tensor] = None,
    ) -> None:
        """
        - `target` (str):
          The type of layer to be replaced.
        - `to` (str):
          The type of layer to replace with.
        - `test_replacement` (Tensor, optional):
          An optional tensor to test the replacement.
          Defaults to None.

        ### Supported replacements (target -> to)
        - "relu" (ReLU) <-> "gelu" (GELU)
        - "attn" (Self-Attention) -> "pattn" (Conv1x1 + Self-Attention)
        - "bn" (BatchNorm) <-> "ln" (LayerNorm)

        ### Raises
        - `AssertionError`:
          If the target and replacement types are not supported.
        """
        assert (target, to) in [
            item for sublist in self.replacements.values() for item in sublist
        ], f"{target}->{to} not supported."

        self.target = target
        self.to = to
        self.test_replacement = test_replacement

    def __call__(
        self,
        model: Module,
    ) -> Module:
        """
        Replaces specified layers in the model and optionally tests the replacement.

        ### Parameters
        - `model` (Module):
          The neural network model to be modified.

        ### Returns
        - `Module`:
          The modified neural network model.

        ### Raises
        - `ValueError`:
          If the target and replacement types are not found in the `replacements` dictionary.
        """
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
