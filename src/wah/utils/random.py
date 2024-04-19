import time

import lightning as L

__all__ = [
    "seed_everything",
    "unseed_everything",
]


def seed_everything(seed: int, ) -> None:
    L.seed_everything(seed if seed >= 0 else None)


def unseed_everything() -> None:
    t = 1000 * time.time()  # current time in milliseconds
    seed = int(t) % 2 ** 32  # seed must be in range [0, 2^32-1]

    L.seed_everything(seed if seed >= 0 else None)
