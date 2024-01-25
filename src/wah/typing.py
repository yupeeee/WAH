from typing import Any, Dict, Optional, Union

import os

from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader, Dataset

__all__ = [
    "Any",
    "Config",
    "DataLoader",
    "Dataset",
    "Dict",
    "LRScheduler",
    "Module",
    "Optimizer",
    "Optional",
    "Path",
]

Config = Dict[str, Any]
Path = Union[str, bytes, os.PathLike]
