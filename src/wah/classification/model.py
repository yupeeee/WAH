import os
from typing import Dict, List, Optional, Tuple, Union

import torch

from ..module import getattrs, summary
from .load import _load_model, _load_state_dict

__all__ = [
    "ClassificationModel",
]


class ClassificationModel(torch.nn.Module):
    def __init__(
        self,
        name: str,
        weights: Optional[Union[str, os.PathLike]] = None,
        num_classes: int = 1000,
        image_size: int = 224,
        num_channels: int = 3,
        **kwargs,
    ) -> None:
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

        self.device: torch.device = torch.device("cpu")

        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
        self._features: Dict[str, torch.Tensor] = {}

    @property
    def num_classes(self) -> int:
        return self._num_classes

    @property
    def image_size(self) -> int:
        return self._image_size

    @property
    def num_channels(self) -> int:
        return self._num_channels

    def to(
        self,
        device: torch.device,
    ) -> "ClassificationModel":
        self.device = device
        self._model.to(device)
        return self

    def load_state_dict(
        self,
        state_dict_path: os.PathLike,
    ) -> "ClassificationModel":
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
        return self._model(x.to(self.device))

    def layers(
        self,
    ) -> List[str]:
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
        return summary(
            model=self._model,
            input_shape=(1, self._num_channels, self._image_size, self._image_size),
            input_dtype=torch.float32,
            eval=True,
            skip_dropout=True,
            skip_identity=True,
            print_summary=print_summary,
        )
