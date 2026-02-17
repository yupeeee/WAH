from typing import List, Union

import torch

__all__ = [
    "parse_devices",
]


def parse_devices(devices: Union[str, List[int], int] = "auto") -> List[int]:
    """Parse a device specification into a list of GPU indices.

    ### Args
        - `devices` (str, int, or list of ints): Device specification.
            * Examples:
                - "cpu", "cuda", "gpu", "auto"
                - "cuda:0,1", "gpu:2,3,4"
                - "0,1,2"
                - int (e.g., 0)
                - List[int] (e.g., [0, 1, 2])

    ### Returns
        - `List[int]`: List of GPU indices (empty if using CPU).

    ### Example
    ```python
    >>> parse_devices("cpu")
    []
    >>> parse_devices("cuda:0,1")
    [0, 1]
    >>> parse_devices([0, 1, 2])
    [0, 1, 2]
    >>> parse_devices("auto")
    # Returns all available GPU indices or [] if no GPU present
    ```
    """
    if devices == "auto":
        return (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
    if isinstance(devices, str):
        s = devices.lower().replace(" ", "")
        if s == "cpu":
            return []
        if s == "cuda" or s == "gpu":
            assert torch.cuda.is_available(), "CUDA is not available"
            # Use only the first visible gpu
            return [0]
        # Parse things like "cuda:0,1,2" or "gpu:1,2"
        if s.startswith("cuda:") or s.startswith("gpu:"):
            idxs = s.split(":")[1]
            return [int(i) for i in idxs.split(",") if i.strip() != ""]
        # Parse comma separated digits, like "0,1,2"
        return [int(i) for i in s.split(",") if i.strip() != ""]
    if isinstance(devices, int):
        return [devices]
    if isinstance(devices, (list, tuple)):
        return [int(i) for i in devices]
    raise ValueError(f"Invalid device specification: {devices}")
