import os
from argparse import Namespace
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

from lightning import LightningModule, Trainer
from matplotlib.axes import Axes
from pandas import DataFrame
from PIL.Image import Image
from torch import Tensor
from torch import device as Device
from torch.nn import Module
from torch.optim import Optimizer

# ImportError: cannot import name 'LRScheduler' from 'torch.optim.lr_scheduler'
try:
    from torch.optim.lr_scheduler import LRScheduler
except ImportError:
    from torch.optim.lr_scheduler import LambdaLR as LRScheduler

from torch.utils.data import DataLoader, Dataset
from torch.utils.hooks import RemovableHandle
from transformers.modeling_outputs import BaseModelOutputWithPooling as CLIPOutput

__all__ = [
    "Any",
    "Axes",
    "Callable",
    "CLIPOutput",
    "Config",
    "DataFrame",
    "DataLoader",
    "Dataset",
    "Device",
    "Devices",
    "Dict",
    "Image",
    "LightningModule",
    "List",
    "Literal",
    "LRScheduler",
    "Module",
    "Namespace",
    "Optimizer",
    "Optional",
    "Path",
    "RemovableHandle",
    "Sequence",
    "Tensor",
    "Trainer",
    "Tuple",
    "Union",
]

Config = Dict[str, Any]
Devices = Union[int, str, List[Union[int, str]]]
Path = Union[str, bytes, os.PathLike]
