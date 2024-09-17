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
    if _seed is None:
        _seed = -1

    if _seed >= 0:
        L.seed_everything(_seed)


def unseed() -> None:
    t = 1000 * time.time()  # current time in milliseconds
    _seed = int(t) % 2**32  # seed must be in range [0, 2^32-1]

    L.seed_everything(_seed)
