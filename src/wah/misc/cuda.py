import os

import torch

__all__ = [
    "set_visible_devices",
]


def set_visible_devices(device: str) -> str:
    """Set visible CUDA devices and return normalized device string.

    ### Args
        - `device` (str): Device specification string. Can be:
            - 'cpu': Disable all CUDA devices
            - 'auto': Use all available devices
            - '0', '1', etc: Use specific device number
            - 'gpu:0', 'gpu:0,1', etc: Use specific GPU devices

    ### Returns
        - `str`: Normalized device string

    ### Example
    ```python
    >>> set_visible_devices("cpu")
    'cpu'

    >>> set_visible_devices("auto")
    'gpu:0,1'   # if there are 2 GPUs
    'cpu'       # if there are no GPUs

    >>> set_visible_devices("gpu:0,1")
    'gpu:0,1'

    >>> set_visible_devices("2")
    'gpu:0'
    ```
    """
    if not isinstance(device, str):
        raise TypeError("device must be a string")
    # Handle CPU case
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        return device
    # Handle auto case
    if device == "auto":
        num_devices = torch.cuda.device_count()
        device = (
            f"gpu:{','.join(str(i) for i in range(num_devices))}"
            if num_devices > 0
            else "cpu"
        )
        return device
    # Parse device string
    if ":" in device:
        prefix, devices = device.split(":")
    else:
        prefix = "gpu"
        devices = device
    # Validate and normalize device numbers
    try:
        device_ids = [int(d.strip()) for d in devices.split(",")]
        if any(d < 0 for d in device_ids):
            raise ValueError("Device IDs must be non-negative")
    except ValueError:
        raise ValueError("Invalid device specification")
    # Set CUDA devices
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, device_ids))
    normalized_ids = list(range(len(device_ids)))
    torch.cuda.set_device(normalized_ids[0])
    return f"{prefix}:{','.join(map(str, normalized_ids))}"
