from .activation import replace_gelu_with_relu as gelu_with_relu
from .activation import replace_relu_with_gelu as relu_with_gelu
from .normalization import replace_bn_with_ln as bn_with_ln
from .normalization import replace_ln_with_bn as ln_with_bn

__all__ = [
    # activation
    "gelu_with_relu",
    "relu_with_gelu",
    # normalization
    "bn_with_ln",
    "ln_with_bn",
]
