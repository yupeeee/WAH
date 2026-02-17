import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from ..module import getattrs, summary
from .load import _load_model, _load_state_dict

__all__ = [
    "ClassificationModel",
]


class ClassificationModel(nn.Module):
    """Image classification models from timm.

    ### Args
        - `name` (str): Name of the classification model to load (from timm).
        - `weights` (Optional[Union[str, os.PathLike]], optional):
            - If "auto", use pretrained weights provided by timm (default).
            - If path/filename, load from a state dict checkpoint file.
            - If None, initialize randomly.
        - `num_classes` (int, optional): Number of output classes. Defaults to 1000.
        - `image_size` (int, optional): Input image size for the model. Defaults to 224.
        - `num_channels` (int, optional): Number of input channels. Defaults to 3.
        - `**kwargs`: Additional keyword arguments to pass to the timm model constructor.

    ### Attributes
        - `model` (nn.Module): The internal model loaded from timm.
        - `num_classes` (int): Number of output classes.
        - `image_size` (int): Input image size.
        - `num_channels` (int): Number of input channels.
        - `device` (torch.device): Device assigned to the model.

    ### Example
    ```python
    >>> model = ClassificationModel("vit_base_patch16_224", num_classes=10)
    >>> x = torch.randn(1, 3, 224, 224)
    >>> logits = model(x)
    ```
    """

    def __init__(
        self,
        name: str,
        weights: Optional[Union[str, os.PathLike]] = "auto",
        num_classes: int = 1000,
        image_size: int = 224,
        num_channels: int = 3,
        **kwargs,
    ) -> None:
        """
        Initialize a ClassificationModel.

        ### Args
            - `name` (str): Name of the timm model to instantiate.
            - `weights` (Optional[Union[str, os.PathLike]], optional): Pretrained weights ("auto" for timm default, or state dict path).
            - `num_classes` (int, optional): Output class count. Defaults to 1000.
            - `image_size` (int, optional): Model input image size. Defaults to 224.
            - `num_channels` (int, optional): Input channels. Defaults to 3.
            - `**kwargs`: Additional kwargs passed to timm.create_model.
        """
        super().__init__()

        weights_path: os.PathLike = None
        if weights is not None and weights != "auto":
            weights_path = weights
        pretrained = True if weights == "auto" else False

        self._model = _load_model(
            name,
            pretrained,
            num_classes,
            image_size,
            num_channels,
            **kwargs,
        )

        if weights_path is not None:
            self._model = _load_state_dict(
                model=self._model,
                state_dict_path=weights_path,
                map_location="cpu",
            )

        self._num_classes: int = num_classes
        self._image_size: int = image_size
        self._num_channels: int = num_channels

        self._device: torch.device = torch.device("cpu")

    @property
    def model(self) -> nn.Module:
        """Model loaded from timm."""
        return self._model

    @property
    def num_classes(self) -> int:
        """Number of output classes."""
        return self._num_classes

    @property
    def image_size(self) -> int:
        """Input image size."""
        return self._image_size

    @property
    def num_channels(self) -> int:
        """Number of input channels."""
        return self._num_channels

    @property
    def device(self) -> torch.device:
        """Device the model is currently on."""
        return self._device

    def to(
        self,
        device: torch.device,
    ) -> "ClassificationModel":
        """Move the internal model to the specified device.

        ### Args
            - `device` (torch.device): The target device.

        ### Returns
            - `ClassificationModel`: self, for chaining.
        """
        self._device = device
        self._model.to(device)
        return self

    def load_state_dict(
        self,
        state_dict_path: os.PathLike,
    ) -> "ClassificationModel":
        """Load a state dict checkpoint into the internal model.

        ### Args
            - `state_dict_path` (os.PathLike): Path to the state dict file.

        ### Returns
            - `ClassificationModel`: self, for chaining.
        """
        self._model = _load_state_dict(
            model=self._model,
            state_dict_path=state_dict_path,
            map_location=self.device,
        )
        return self

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Run a forward pass through the model.

        ### Args
            - `x` (torch.Tensor): Input tensor of shape (N, C, H, W).

        ### Returns
            - `torch.Tensor`: Output logits (before activation/softmax).
        """
        return self._model(x.to(self.device))

    def layers(
        self,
    ) -> List[str]:
        """Return a list of attribute paths for all layers in the model.

        ### Returns
            - `List[str]`: List of attribute paths for all layers in the model.
        """
        return getattrs(
            module=self._model,
            specify=None,
            max_depth=None,
            skip_dropout=True,
            skip_identity=True,
        )

    def summary(
        self,
        print_summary: bool = True,
    ) -> Dict[str, Dict[str, Union[str, Tuple[int, ...]]]]:
        """Return a summary of the model (layer names, shapes, etc).

        ### Args
            - `print_summary` (bool, optional): Whether to print the summary. Defaults to True.

        ### Returns
            - `Dict[str, Dict[str, Union[str, Tuple[int, ...]]]]`: Summary information per layer.
        """
        return summary(
            model=self._model,
            input_shape=(1, self._num_channels, self._image_size, self._image_size),
            input_dtype=torch.float32,
            eval=True,
            skip_dropout=True,
            skip_identity=True,
            print_summary=print_summary,
        )
