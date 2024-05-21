from collections import OrderedDict

import timm
import torch
from torchvision import models

from ..typing import (
    Dict,
    Module,
    Optional,
    Path,
    Tensor,
    Transform,
    Union,
)
from .timm_cfg import _timm_need_img_size

__all__ = [
    "load_state_dict",
    "load_pytorch_model",
    "load_timm_model",
    "add_preprocess",
]


def load_state_dict(
    model: Module,
    state_dict_path: Path,
    **kwargs,
) -> None:
    state_dict: Dict[str, Tensor] = torch.load(state_dict_path, **kwargs)

    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    required_keys = list(model.state_dict().keys())

    for key in list(state_dict.keys()):
        if "feature_extractor." in key:
            del state_dict[key]

        elif "model." in key:
            state_dict[key.replace("model.", "")] = state_dict.pop(key)

        else:
            continue

    for key in list(state_dict.keys()):
        if key not in required_keys:
            del state_dict[key]

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
