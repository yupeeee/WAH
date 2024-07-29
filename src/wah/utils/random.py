import time

import lightning as L

__all__ = [
    "seed_everything",
    "unseed_everything",
]


def seed_everything(
    seed: int,
) -> None:
    """
    Sets the seed for all random number generators to ensure reproducibility.

    ### Parameters
    - `seed (int)`: The seed value to set. If `None`, a default value of `-1` is used.

    ### Returns
    - `None`

    ### Notes
    - This function sets the seed for all relevant random number generators using `lightning.seed_everything`.
    - If the seed is `None`, it defaults to `-1`, which indicates no specific seeding.
    """
    if seed is None:
        seed = -1

    if seed >= 0:
        L.seed_everything(seed)


def unseed_everything() -> None:
    """
    Unsets the seed for random number generators by setting a new seed based on the current time.

    ### Parameters
    - `None`

    ### Returns
    - `None`

    ### Notes
    - This function sets a new seed for all relevant random number generators using the current time in milliseconds.
    - The seed is computed to be in the range `[0, 2^32-1]`.
    """
    t = 1000 * time.time()  # current time in milliseconds
    seed = int(t) % 2**32  # seed must be in range [0, 2^32-1]

    L.seed_everything(seed if seed >= 0 else None)
