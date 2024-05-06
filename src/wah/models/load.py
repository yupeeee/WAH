from collections import OrderedDict
import re

import timm
import torch
from torchvision import models

from ..typing import (
    Module,
    Optional,
    Path,
    Transform,
    Union,
)
from ..utils.lst import load_txt

__all__ = [
    "model_family_name",
    "load_state_dict",
    "load_pytorch_model",
    "load_timm_model",
    "add_preprocess",
]

_timm_family = load_txt("./timm_family.txt")
_timm_need_img_size = load_txt("./timm_need_img_size.txt")


def model_family_name(
    name: str,
) -> str:
    name = name.split("_")[0]

    match = re.match(r"([a-z]+)([0-9]+)", name, re.I)

    if match:
        name, num = match.groups()

        # consider ~v2, ~v3, ~v4 as family name
        if name[-1] == "v" and num in [
            "2",
            "3",
            "4",
        ]:
            name += num

    return name


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


def load_pytorch_model(
    name: str,
    weights: Optional[str] = None,
    num_classes: int = 1000,
    image_size: int = 224,
    **kwargs,
) -> Module:
    assert name in models.list_models(), (
        f"PyTorch does not support {name}. "
        f"Check torchvision.models.list_models() for supported models."
    )

    model_kwargs = {
        "weights": weights,
        "num_classes": num_classes,
    }
    if "vit" in name:
        model_kwargs["image_size"] = image_size

    model_kwargs = {**model_kwargs, **kwargs}

    model = getattr(models, name)(**model_kwargs)

    return model


def load_timm_model(
    name: str,
    pretrained: bool = False,
    num_classes: int = 1000,
    image_size: int = 224,
    num_channels: int = 3,
    **kwargs,
) -> Module:
    assert name in timm.list_models(), (
        f"timm does not support {name}. "
        f"Check timm.list_models() for supported models."
    )

    # model kwargs
    model_kwargs = {
        "model_name": name,
        "pretrained": pretrained,
        "num_classes": num_classes,
        # "img_size": image_size,
        "in_chans": num_channels,
    }

    if name in _timm_need_img_size:
        model_kwargs["img_size"] = image_size

    model_kwargs = {**model_kwargs, **kwargs}

    # load model
    model = timm.create_model(**model_kwargs)

    return model


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
