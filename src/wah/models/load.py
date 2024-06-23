from collections import OrderedDict
import os

import timm
import torch
from torchvision import models

from ..typing import (
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
    "load_state_dict",
    "load_model",
    "add_preprocess",
]


def load_state_dict(
    model: Module,
    state_dict_path: Path,
    map_location: Optional[Device] = None,
) -> None:
    """
    Loads a state dictionary into a model, with optional mapping to a specific device.

    ### Parameters
    - `model` (Module):
      The model into which to load the state dictionary.
    - `state_dict_path` (Path):
      The path to the state dictionary file.
    - `map_location` (Device, optional):
      The device to map the state dictionary to.
      Defaults to None.

    ### Returns
    - `None`

    ### Notes
    - This function handles different key formats in the state dictionary.
    - It removes unnecessary keys and renames keys to match the model's state dictionary.
    """
    state_dict: Dict[str, Tensor] = torch.load(
        state_dict_path,
        map_location=map_location,
    )

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
    - `name` (str):
      The name of the model to load.
    - `pretrained` (bool, optional):
      Whether to load pretrained weights.
      Defaults to False.
    - `num_classes` (int, optional):
      The number of output classes.
      Defaults to 1000 (ImageNet).
    - `image_size` (int, optional):
      The input image size.
      Defaults to 224 (ImageNet).
    - `num_channels` (int, optional):
      The number of input channels.
      Defaults to 3 (RGB).
    - `**kwargs`:
      Additional keyword arguments for the model.

    ### Returns
    - `Module`:
      The loaded model.

    ### Raises
    - `AssertionError`:
      If the model name is not supported by timm.
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
    - `name` (str):
      The name of the model to load.
    - `weights` (str, optional):
      The weights to load.
      Defaults to None.
    - `num_classes` (int, optional):
      The number of output classes.
      Defaults to 1000 (ImageNet).
    - `image_size` (int, optional):
      The input image size.
      Defaults to 224 (ImageNet).
    - `**kwargs`:
      Additional keyword arguments for the model.

    ### Returns
    - `Module`:
      The loaded model.

    ### Raises
    - `AssertionError`:
      If the model name is not supported by torchvision.
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
    map_location: Optional[Device] = None,
    **kwargs,
) -> Module:
    """
    Loads a model from the specified library (timm or torchvision).

    ### Parameters
    - `name` (str):
      The name of the model to load.
    - `weights` (str, optional):
      The path to the weights file, or "auto" to load pretrained weights, or weight specification to load pretrained torchvision model.
      Defaults to None.
    - `num_classes` (int, optional):
      The number of output classes.
      Defaults to 1000 (ImageNet).
    - `image_size` (int, optional):
      The input image size.
      Defaults to 224 (ImageNet).
    - `num_channels` (int, optional):
      The number of input channels.
      Defaults to 3 (RGB).
    - `load_from` (Literal["timm", "torchvision"], optional):
      The library to load the model from.
      Defaults to "timm".
    - `map_location` (Device, optional):
      The device to map the weights to.
      Defaults to None.
    - `**kwargs`:
      Additional keyword arguments for the model.

    ### Returns
    - `Module`:
      The loaded model.

    ### Raises
    - `ValueError`:
      If the specified library (`load_from`) is not supported.

    ### Notes
    - This function supports loading models from either the timm or torchvision library.
    - If weights are specified and exist, the function loads them into the model.
    """
    weights_path: Path = weights if os.path.exists(weights) else None
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


def add_preprocess(
    model: Module,
    preprocess: Union[Module, Transform],
) -> Module:
    """
    Adds a preprocessing step to the model.

    ### Parameters
    - `model` (Module):
      The model to which the preprocessing step will be added.
    - `preprocess` (Union[Module, Transform]):
      The preprocessing step to add.

    ### Returns
    - `Module`:
      The model with the preprocessing step added.

    ### Notes
    - This function creates a sequential module that first applies the preprocessing and then the model.
    """
    return torch.nn.Sequential(
        OrderedDict(
            [
                ("preprocess", preprocess),
                ("model", model),
            ]
        )
    )
