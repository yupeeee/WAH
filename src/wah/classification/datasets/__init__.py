from .base import ClassificationDataset
from .cifar10 import CIFAR10
from .cifar100 import CIFAR100
from .imagenet import ImageNet

__all__ = [
    "ClassificationDataset",
    "CIFAR10",
    "CIFAR100",
    "ImageNet",
]
