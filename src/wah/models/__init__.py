from collections import OrderedDict

import torch

from ..typing import (
    Module,
    Path,
    Transform,
    Union,
)
from .feature_extractor import FeatureExtractor
from .modules import *
from .replace import Replacer
from .test import *
from .travel import Traveler

__all__ = [
    "load_state_dict",
    "add_preprocess",

    # feature_extractor
    "FeatureExtractor",

    # modules
    "_getattr",
    "get_valid_attr",
    "get_module_name",
    "get_module_params",
    "get_named_modules",
    "get_attrs",

    # replace
    "Replacer",

    # test
    "AccuracyTest",

    # travel
    "Traveler",
]


def load_state_dict(
    model: Module,
    state_dict_path: Path,
    **kwargs,
) -> None:
    state_dict = torch.load(state_dict_path, **kwargs)["state_dict"]

    for key in state_dict.copy().keys():
        if "model." in key:
            state_dict[key.replace("model.", "")] = state_dict.pop(key)

        elif "feature_extractor." in key:
            del state_dict[key]

        else:
            continue

    model.load_state_dict(state_dict)


def add_preprocess(
    model: Module,
    preprocess: Union[Module, Transform],
) -> Module:
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("preprocess", preprocess),
                ("model", model),
            ]
        )
    )
