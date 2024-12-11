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
    """
    Adds preprocessing to a model pipeline using `torch.nn.Sequential`.

    ### Parameters
    - `model` (Module): The model to which preprocessing is added.
    - `preprocess` (Union[Module, Transform]): The preprocessing transform or module to apply.

    ### Returns
    - `Module`: A sequential model where preprocessing is applied before the model.
    """
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
    """
    Loads a model from the timm library.

    ### Parameters
    - `name` (str): The name of the model to load.
    - `pretrained` (bool, optional): Whether to load a pretrained model. Defaults to `False`.
    - `num_classes` (int, optional): The number of output classes. Defaults to `1000`.
    - `image_size` (int, optional): The input image size. Defaults to `224`.
    - `num_channels` (int, optional): The number of input channels. Defaults to `3`.
    - `**kwargs`: Additional keyword arguments passed to `timm.create_model`.

    ### Returns
    - `Module`: The loaded model.
    """
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
    """
    Loads a model from the torchvision library.

    ### Parameters
    - `name` (str): The name of the model to load.
    - `weights` (Optional[str], optional): Path to the pretrained weights. Defaults to `None`.
    - `num_classes` (int, optional): The number of output classes. Defaults to `1000`.
    - `image_size` (int, optional): The input image size. Defaults to `224`.
    - `**kwargs`: Additional keyword arguments passed to the torchvision model.

    ### Returns
    - `Module`: The loaded model.
    """
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
    **kwargs,
) -> Module:
    """
    Loads a model from either the timm or torchvision library.

    ### Parameters
    - `name` (str): The name of the model to load.
    - `weights` (Optional[str], optional): Path to the pretrained weights. Defaults to `None`.
    - `num_classes` (int, optional): The number of output classes. Defaults to `1000`.
    - `image_size` (int, optional): The input image size. Defaults to `224`.
    - `num_channels` (int, optional): The number of input channels. Defaults to `3`.
    - `load_from` (Literal["timm", "torchvision"], optional): Whether to load the model from timm or torchvision. Defaults to `"timm"`.
    - `map_location` (Optional[Device], optional): The device to load the model on. Defaults to `"cpu"`.
    - `**kwargs`: Additional keyword arguments passed to the model.

    ### Returns
    - `Module`: The loaded model.
    """
    weights_path: Path = weights if weights is not None else None
    pretrained = True if weights == "auto" else False

    if load_from == "timm":
        model = load_timm_model(
            name=name,
            pretrained=pretrained,
            num_classes=num_classes,
            image_size=image_size,
            num_channels=num_channels,
            **kwargs,
        )

    elif load_from == "torchvision":
        model = load_torchvision_model(
            name=name,
            weights=weights,
            num_classes=num_classes,
            image_size=image_size,
            **kwargs,
        )

    else:
        raise ValueError(f"Unsupported library: {load_from}")

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
) -> Module:
    """
    Loads a state dictionary into the model.

    ### Parameters
    - `model` (Module): The model to load the state dictionary into.
    - `state_dict_path` (Path): The path to the state dictionary.
    - `map_location` (Optional[Device], optional): The device to map the model's parameters to. Defaults to `"cpu"`.
    """
    assert _path.exists(state_dict_path), f"{state_dict_path} does not exist."

    state_dict: Dict[str, Tensor] = torch.load(
        state_dict_path,
        map_location=map_location,
        weights_only=True,
    )

    # if state_dict is "last.ckpt", i.e., contains other data
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

        for key in list(state_dict.keys()):
            state_dict[key.replace("model.", "")] = state_dict.pop(key)

    # state_dict check: remove redundancy in keys
    for key in list(state_dict.keys()):
        if "feature_extractor." in key:
            del state_dict[key]

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

    return model
