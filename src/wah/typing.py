from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import os

from torch import device as Device, Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer

# ImportError: cannot import name 'LRScheduler' from 'torch.optim.lr_scheduler'
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import LambdaLR as LRScheduler

from torch.utils.data import DataLoader, Dataset

from torchmetrics import Metric

__all__ = [
    "Any",
    "Callable",
    "Config",
    "DataLoader",
    "Dataset",
    "Device",
    "Dict",
    "List",
    "LRScheduler",
    "Metric",
    "Module",
    "Optimizer",
    "Optional",
    "Parameter",
    "Path",
    "Sequence",
    "Tensor",
    "Tuple",
    "Union",
]

Config = Dict[str, Any]
Path = Union[str, bytes, os.PathLike]
