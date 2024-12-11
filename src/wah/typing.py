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

from lightning import LightningModule, Trainer
from matplotlib.collections import PathCollection
from matplotlib.colors import Colormap
from matplotlib.colors import Normalize as Colornorm
from matplotlib.figure import Figure
from matplotlib.pyplot import Axes
from numpy import ndarray as NDArray
from pandas import DataFrame
from torch import Tensor
from torch import device as Device
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from torch.utils.hooks import RemovableHandle
from torch.utils.tensorboard.writer import SummaryWriter

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
    "AxIndex",
    "Callable",
    "Color",
    "Colormap",
    "Colornorm",
    "Config",
    "DataFrame",
    "DataLoader",
    "Dataset",
    "Device",
    "Devices",
    "Dict",
    "Figure",
    "Iterable",
    "LightningModule",
    "List",
    "Literal",
    "LRScheduler",
    "Metric",
    "Module",
    "NDArray",
    "Optimizer",
    "Optional",
    "Path",
    "PathCollection",
    "RemovableHandle",
    "Sequence",
    "SummaryWriter",
    "Tensor",
    "Trainer",
    "Transform",
    "Tuple",
    "Union",
]

AxIndex = Union[int, Tuple[int, int]]
Config = Dict[str, Any]
Color = Union[Tuple[float, float, float], str]
Devices = Union[int, str, List[Union[int, str]]]
Path = Union[str, bytes, os.PathLike]
