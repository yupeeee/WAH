import timm
import torch
from torchvision import models

from ...misc import path as _path
from ...misc.typing import Device, Dict, Literal, Module, Optional, Path, Tensor, Union

__all__ = [
    "load_model",
    "load_state_dict",
]


def load_timm_model(
    name: str,
    pretrained: bool = False,
    num_classes: int = 1000,
    image_size: int = 224,
    num_channels: int = 3,
    **kwargs,
) -> Module:
    """Load a model from timm.

    ### Args
        - `name` (str): Name of the model to load
        - `pretrained` (bool): Whether to load pretrained weights. Defaults to `False`.
        - `num_classes` (int): Number of classes for the model head. Defaults to `1000`.
        - `image_size` (int): Size of input images. Defaults to `224`.
        - `num_channels` (int): Number of input channels. Defaults to `3`.
        - `**kwargs`: Additional arguments to pass to timm.create_model()

    ### Returns
        - `Module`: The loaded model

    ### Example
    ```python
    # Load a ViT model with 10 output classes
    >>> model = load_timm_model(
    ...     name="vit_base_patch16_224",
    ...     num_classes=10,
    ...     image_size=32,
    ...     num_channels=3,
    ... )
    ```
    """
    assert name in timm.list_models(), (
        f"timm does not support {name}. "
        f"Check timm.list_models() for supported models."
    )
    model_kwargs = {
        "model_name": name,
        "pretrained": pretrained,
        "num_classes": num_classes,
        "in_chans": num_channels,
    }
    model_loader = timm.create_model
    try:
        _ = model_loader(model_name=name, img_size=image_size)
        model_kwargs["img_size"] = image_size
    except (TypeError, ValueError):
        pass
    model_kwargs = {**model_kwargs, **kwargs}
    model = model_loader(**model_kwargs)
    return model


def load_torchvision_model(
    name: str,
    weights: Optional[str] = None,
    num_classes: int = 1000,
    image_size: int = 224,
    **kwargs,
) -> Module:
    """Load a model from torchvision.

    ### Args
        - `name` (str): Name of the model to load
        - `weights` (Optional[str]): Weights to load. Can be "auto" or path to weights file. Defaults to `None`.
        - `num_classes` (int): Number of classes for the model head. Defaults to `1000`.
        - `image_size` (int): Size of input images. Defaults to `224`.
        - `**kwargs`: Additional arguments to pass to the model constructor

    ### Returns
        - `Module`: The loaded model

    ### Example
    ```python
    # Load a ViT model with 10 output classes
    >>> model = load_torchvision_model(
    ...     name="vit_b_16",
    ...     num_classes=10,
    ...     image_size=32,
    ... )
    ```
    """
    assert name in models.list_models(), (
        f"torchvision does not support {name}. "
        f"Check torchvision.models.list_models() for supported models."
    )
    model_kwargs = {
        "weights": weights,
        "num_classes": num_classes,
    }
    model_loader = getattr(models, name)
    try:
        _ = model_loader(model_name=name, image_size=image_size)
        model_kwargs["image_size"] = image_size
    except (TypeError, ValueError):
        pass
    model_kwargs = {**model_kwargs, **kwargs}
    model = model_loader(**model_kwargs)
    return model


def load_model(
    name: str,
    weights: Optional[Union[str, Path]] = None,
    num_classes: int = 1000,
    image_size: int = 224,
    num_channels: int = 3,
    load_from: Literal["timm", "torchvision"] = "timm",
    map_location: Optional[Device] = "cpu",
    **kwargs,
) -> Module:
    """Load a model from either timm or torchvision.

    ### Args
        - `name` (str): Name of the model to load
        - `weights` (Union[str, Path], optional): Weights to load. Can be "auto" or path to weights file. Defaults to `None`.
        - `num_classes` (int): Number of classes for the model head. Defaults to `1000`.
        - `image_size` (int): Size of input images. Defaults to `224`.
        - `num_channels` (int): Number of input channels. Defaults to `3`.
        - `load_from` (Literal["timm", "torchvision"]): Library to load model from. Defaults to `"timm"`.
        - `map_location` (Device, optional): Device to load model to. Defaults to `"cpu"`.
        - `**kwargs`: Additional arguments to pass to the model constructor

    ### Returns
        - `Module`: The loaded model

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
    weights_path: Path = None
    if weights is not None and weights != "auto":
        weights_path = weights
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
    """Load a state dictionary into a model.

    ### Args
        - `model` (Module): Model to load the state dictionary into
        - `state_dict_path` (Path): Path to the state dictionary file
        - `map_location` (Device, optional): Device to load the state dictionary to. Defaults to "cpu".

    ### Returns
        - `Module`: The model with the loaded state dictionary

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
