import os
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Literal,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from matplotlib.pyplot import Axes
from pandas import DataFrame
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
from torchvision.transforms.v2 import Transform

__all__ = [
    "Any",
    "Axes",
    "Callable",
    "Config",
    "DataFrame",
    "DataLoader",
    "Dataset",
    "Device",
    "Devices",
    "Dict",
    "Iterable",
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
    "Transform",
    "Tuple",
    "Union",
]

Config = Dict[str, Any]
Devices = Union[int, str, List[Union[int, str]]]
Path = Union[str, bytes, os.PathLike]
