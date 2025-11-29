import inspect
import os
from typing import Dict, Optional

import timm
import torch

__all__ = [
    "_load_model",
    "_load_state_dict",
]


def _load_model(
    name: str,
    pretrained: bool = False,
    num_classes: int = 1000,
    image_size: int = 224,
    num_channels: int = 3,
    **kwargs,
) -> torch.nn.Module:
    """Load a classification model from timm.

    ### Args
        - `name` (str): Name of the model to load
        - `pretrained` (bool): Whether to load pretrained weights. Defaults to `False`.
        - `num_classes` (int): Number of classes for the model head. Defaults to `1000`.
        - `image_size` (int): Size of input images. Defaults to `224`.
        - `num_channels` (int): Number of input channels. Defaults to `3`.
        - `**kwargs`: Additional arguments to pass to `timm.create_model()`.

    ### Returns
        - `torch.nn.Module`: The loaded model

    ### Example
    ```python
    # Load a ViT model with 10 output classes
    >>> model = load_model(
    ...     name="vit_base_patch16_224",
    ...     num_classes=10,
    ...     image_size=32,
    ...     num_channels=3,
    ... )
    ```
    """
    assert (
        name in timm.list_models()
    ), f"timm does not support {name}. Check timm.list_models() for supported models."

    model_loader = timm.create_model

    model_kwargs = {
        "model_name": name,
        "pretrained": pretrained,
        "num_classes": num_classes,
        "in_chans": num_channels,
    }
    model_kwargs = {**model_kwargs, **kwargs}

    # Check if 'img_size' is a valid argument
    signature = inspect.signature(model_loader)
    if "img_size" in signature.parameters:
        model_kwargs["img_size"] = image_size

    model = model_loader(**model_kwargs)

    return model


def _load_state_dict(
    model: torch.nn.Module,
    state_dict_path: os.PathLike,
    map_location: Optional[torch.device] = "cpu",
) -> torch.nn.Module:
    """Load a state dictionary into a model.

    ### Args
        - `model` (torch.nn.Module): Model to load the state dictionary into
        - `state_dict_path` (os.PathLike): Path to the state dictionary file
        - `map_location` (torch.device, optional): Device to load the state dictionary to. Defaults to "cpu".

    ### Returns
        - `torch.nn.Module`: The model with the loaded state dictionary

    ### Example
    ```python
    # Load a state dictionary into a model
    >>> model = load_state_dict(
    ...     model=model,
    ...     state_dict_path="model.pth",
    ...     map_location="cuda",
    ... )
    ```
    """
    assert os.path.exists(state_dict_path), f"{state_dict_path} does not exist."

    state_dict: Dict[str, torch.Tensor] = torch.load(
        state_dict_path,
        map_location=map_location,
        weights_only=True,
    )

    # if state_dict is "last.ckpt", i.e., contains other data
    if "state_dict" in state_dict.keys():
        state_dict = state_dict["state_dict"]

    # remove "model." prefix from keys
    for key in list(state_dict.keys()):
        if "model." in key:
            state_dict[key.replace("model.", "")] = state_dict.pop(key)
        else:
            continue

    # remove unnecessary keys & values
    required_keys = list(model.state_dict().keys())
    for key in list(state_dict.keys()):
        if key not in required_keys:
            del state_dict[key]

    model.load_state_dict(state_dict)

    return model
