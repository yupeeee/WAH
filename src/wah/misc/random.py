import time

import lightning as L

from .typing import Union

__all__ = [
    "seed",
    "unseed",
]


def seed(
    _seed: Union[int, None] = None,
) -> None:
    """Set random seed for reproducibility.

    ### Args
        - `_seed` (Union[int, None]): Random seed to use. If None, seed is set to -1. If >= 0, seed is set using lightning.seed_everything(). Defaults to None.

    ### Returns
        - `None`

    ### Example
    ```python
    >>> seed(42)    # Set random seed to 42
    >>> seed()      # No seeding
    >>> seed(None)  # No seeding
    ```
    """
    if _seed is None:
        _seed = -1
    if _seed >= 0:
        L.seed_everything(_seed)


def unseed() -> None:
    """Randomly seed using current time.

    ### Returns
        - `None`

    ### Example
    ```python
    >>> unseed()  # Set random seed using current time
    ```
    """
    t = 1000 * time.time()  # current time in milliseconds
    _seed = int(t) % 2**32  # seed must be in range [0, 2^32-1]
    L.seed_everything(_seed)
