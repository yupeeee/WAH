from torchvision import models

from ..typing import (
    Module,
    Optional,
)
from .modules import *
from .replace import Replacer

__all__ = [
    # "load_model",

    # modules
    "_getattr",
    "get_valid_attr",
    "get_module_name",
    "get_module_params",
    "get_named_modules",
    "get_attrs",

    # replace
    "Replacer",
]


# TODO
# def load_model(
#     net: str,
#     num_classes: int = 1000,
#     image_size: Optional[int] = 224,
#     weights: str = None,
#     **kwargs,
# ) -> Module:
#     model = getattr(models, net)
#     pass
