import os

import torch

from .typing import Devices

__all__ = [
    "set_visible_devices",
]


def set_visible_devices(device: Devices) -> None:
    assert isinstance(
        device, str
    ), "device must be a string, e.g. 'cpu', 'auto', 'gpu:0', or 'gpu:0,1,2'"
    # For CPU, disable all CUDA devices
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return
    # For auto, don't modify anything to use all available devices
    if device == "auto":
        return
    # Parse device specification
    if ":" in device:
        # Handle gpu:0,1,2 format
        devices = device.split(":")[-1]
    else:
        # Handle single device number
        devices = device
    # Set primary device and visible devices
    primary_device = int(devices.split(",")[0])
    torch.cuda.set_device(primary_device)
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
