from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
from PIL.Image import Image
from torch import Tensor
from torchvision import transforms

from ...misc import tensor as _tensor

__all__ = [
    "SSCD",
]


weights_url = (
    "https://dl.fbaipublicfiles.com/sscd-copy-detection/sscd_disc_mixup.torchscript.pt"
)


class SSCD(nn.Module):
    """
    # A Self-Supervised Descriptor for Image Copy Detection (SSCD)
    ### Ed Pizzi, Sreya Dutta Roy, Sugosh Nagavara Ravindra, Priya Goyal, Matthijs Douze, CVPR 2022

    Paper: https://arxiv.org/abs/2202.10261
    GitHub: https://github.com/facebookresearch/sscd-copy-detection
    """

    def __init__(
        self,
        model_path: Path = None,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        if model_path is None:
            self._model = _tensor.load_from_url(weights_url, use_jit=True)
        else:
            self._model = torch.jit.load(model_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize([320, 320]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.to(device)

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def device(self) -> torch.device:
        return self._device

    def to(self, device: torch.device) -> "SSCD":
        self._device = device
        self._model.to(device)
        return self

    def forward(self, images: Union[Image, List[Image]]) -> Tensor:
        if isinstance(images, Image):
            images = [images]
        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0)
        with torch.no_grad():
            embeddings = self._model(images.to(self._device))
        return embeddings.cpu()
