import torch

__all__ = [
    "_get_generator",
]


def _get_generator(
    seed: int | list[int] | None,
    batch_size: int,
    device: torch.device,
) -> torch.Generator:
    if seed is None:
        generator = None
    else:
        if isinstance(seed, int):
            seed = [seed] * batch_size
        generator = [torch.Generator(device=device).manual_seed(s) for s in seed]

    return generator
