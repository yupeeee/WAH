from collections import OrderedDict

import timm
import torch
from torchvision import models

from ... import path as _path
from ...typing import (
    Device,
    Dict,
    Literal,
    Module,
    Optional,
    Path,
    Tensor,
    Transform,
    Union,
)
from .timm_cfg import _timm_need_img_size

__all__ = [
    "add_preprocess",
    "load_model",
    "load_state_dict",
]


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


def load_torchvision_model(
    name: str,
    weights: Optional[str] = None,
    num_classes: int = 1000,
    image_size: int = 224,
    **kwargs,
) -> Module:
    assert name in models.list_models(), (
        f"torchvision does not support {name}. "
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


def load_model(
    name: str,
    weights: Optional[str] = None,
    num_classes: int = 1000,
    image_size: int = 224,
    num_channels: int = 3,
    load_from: Literal["timm", "torchvision"] = "timm",
    map_location: Optional[Device] = "cpu",
    make_feature_2d: Optional[bool] = False,
    **kwargs,
) -> Module:
    weights_path: Path = weights if weights is not None else None
    pretrained = True if weights == "auto" else False
    output_size = 2 if make_feature_2d else num_classes

    if load_from == "timm":
        model = load_timm_model(
            name=name,
            pretrained=pretrained,
            num_classes=output_size,
            image_size=image_size,
            num_channels=num_channels,
            **kwargs,
        )

    elif load_from == "torchvision":
        model = load_torchvision_model(
            name=name,
            weights=weights,
            num_classes=output_size,
            image_size=image_size,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported library: {load_from}")

    if make_feature_2d:
        model = torch.nn.Sequential(
            OrderedDict(
                {
                    "model": model,
                    "classifier": torch.nn.Linear(2, num_classes),
                }
            )
        )

    if weights_path is not None:
        load_state_dict(
            model=model,
            state_dict_path=weights_path,
            map_location=map_location,
        )

    return model


def load_state_dict(
    model: Module,
    state_dict_path: Path,
    map_location: Optional[Device] = "cpu",
) -> None:
    assert _path.exists(state_dict_path), f"{state_dict_path} does not exist."

    state_dict: Dict[str, Tensor] = torch.load(
        state_dict_path,
        map_location=map_location,
    )

    # if state_dict is "last.ckpt", i.e., contains other data
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    # state_dict check: remove redundancy in keys
    for key in list(state_dict.keys()):
        if "feature_extractor." in key:
            del state_dict[key]

        # # make_feature_2d = True
        # elif "model.model." in key:
        #     state_dict[key.replace("model.model", "model")] = state_dict.pop(key)

        # elif "model." in key:
        #     state_dict[key.replace("model.", "")] = state_dict.pop(key)

        else:
            continue

    # state_dict check: remove unnecessary keys & values
    required_keys = list(model.state_dict().keys())
    for key in list(state_dict.keys()):
        if key not in required_keys:
            del state_dict[key]

    model.load_state_dict(state_dict)
