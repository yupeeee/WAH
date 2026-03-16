from . import base
from .classification import CIFAR10, CIFAR100, ImageNet
from .diffusion import WebsterArXiv2023

__all__ = [
    "base",
    # classification
    "CIFAR10",
    "CIFAR100",
    "ImageNet",
    # diffusion
    "WebsterArXiv2023",
]
