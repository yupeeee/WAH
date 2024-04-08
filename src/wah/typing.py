import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from typing_extensions import Literal

from torch import Tensor
from torch import device as Device
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
    "Literal",
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
