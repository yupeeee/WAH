from .normalization import replace_bn_with_ln as bn_with_ln
from .normalization import replace_ln_with_bn as ln_with_bn

__all__ = [
    "bn_with_ln",
    "ln_with_bn",
]
