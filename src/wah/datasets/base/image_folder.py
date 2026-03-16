import os
from typing import Callable, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

from ...misc import path as _path

__all__ = [
    "ImageFolder",
]


class ImageFolder(Dataset):
    def __init__(
        self,
        img_dir: os.PathLike,
        fext: Optional[str] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()

        self.paths = _path.ls(
            path=img_dir,
            fext=fext,
            sort=True,
            absolute=True,
        )
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img
