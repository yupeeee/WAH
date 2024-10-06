import time

import lightning as L

from ..typing import Union

__all__ = [
    "seed",
    "unseed",
]


def seed(
    _seed: Union[int, None] = None,
) -> None:
    """
    Sets the random seed for reproducibility.

    ### Parameters
    - `_seed` (Union[int, None], optional): The seed value to set.
    If not provided, the seed will default to `-1`, meaning no seeding occurs.
    If a value greater than or equal to 0 is provided, the random seed will be set using `L.seed_everything`.

    ### Notes
    - Uses the `seed_everything` function from the `lightning` library to set the seed.
    """
    if _seed is None:
        _seed = -1

    if _seed >= 0:
        L.seed_everything(_seed)


def unseed() -> None:
    """
    Resets the random seed using the current time in milliseconds.

    ### Notes
    - The seed is derived from the current time, ensuring variability across executions.
    - Uses the `seed_everything` function from the `lightning` library to reset the seed.
    """
    t = 1000 * time.time()  # current time in milliseconds
    _seed = int(t) % 2**32  # seed must be in range [0, 2^32-1]

    L.seed_everything(_seed)
