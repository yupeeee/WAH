import io
import urllib.request
from typing import Any

import torch

__all__ = [
    "load_from_url",
]


def load_from_url(
    url: str,
    use_jit: bool = False,
) -> Any:
    """Load a PyTorch .pt file from a URL.

    ### Args
        - `url` (str): The URL to the .pt file.
        - `use_jit` (bool, optional): Whether to load as a TorchScript model using `torch.jit.load()`
          (`True`) or via `torch.load()` for regular state_dict/tensor (`False`). Defaults to `False`.

    ### Returns
        - `Any`: The loaded PyTorch object (could be a ScriptModule, state_dict, tensor, etc.).

    ### Example
    ```python
    # Load TorchScript model
    >>> model = load_from_url("https://example.com/model.pt", use_jit=True)
    # Load state_dict or tensor
    >>> obj = load_from_url("https://example.com/model_state.pt", use_jit=False)
    ```
    """
    with urllib.request.urlopen(url) as response:
        buffer = io.BytesIO(response.read())
        if use_jit:
            return torch.jit.load(buffer)
        else:
            return torch.load(buffer)
