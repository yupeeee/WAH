import torch

from .typing import Sequence, Tensor, Union

__all__ = [
    "broadcasted_elementwise_mul",
    "create_1d_traj",
    "create_2d_grid",
    "flatten_batch",
    "repeat",
    "stretch",
]


def broadcasted_elementwise_mul(
    mat: Tensor,
    vec: Tensor,
    dim: int,
) -> Tensor:
    assert mat.shape[dim] == len(vec)

    # expand vec to match shape with mat
    mat_shape = torch.ones(size=(len(mat.shape),), dtype=int)
    mat_shape[dim] = len(vec)
    vec = vec.view(*mat_shape)

    return mat * vec


def create_1d_traj(
    x: Tensor,
    d: Tensor,
    num_steps: int,
    eps: float,
) -> Tensor:
    shape = (num_steps, *x.shape)

    # reshape x, d
    x = repeat(x, num_steps, dim=0)
    d = repeat(d, num_steps, dim=0)
    x = x.reshape(num_steps, -1)
    d = d.reshape(num_steps, -1)

    # create y along d
    epsilons = torch.linspace(-eps, eps, num_steps)
    epsilons = epsilons.view(-1, 1)
    y = x + d * epsilons
    traj = y.reshape(shape)

    return traj


def create_2d_grid(
    x: Tensor,
    d1: Tensor,
    d2: Tensor,
    num_steps: int,
    eps: float,
) -> Tensor:
    shape = (num_steps * num_steps, *x.shape)

    # reshape x, d1
    x = repeat(x, num_steps, dim=0)
    d1 = repeat(d1, num_steps, dim=0)
    d2 = repeat(d2, num_steps, dim=0)
    x = x.reshape(num_steps, -1)
    d1 = d1.reshape(num_steps, -1)
    d2 = d2.reshape(num_steps, -1)

    # create y along d1
    epsilons = torch.linspace(-eps, eps, num_steps)
    epsilons = epsilons.view(-1, 1)
    y = x + d1 * epsilons

    # reshape y, d2
    y = repeat(y, num_steps, dim=0)
    d2 = repeat(d2, num_steps, dim=0)
    y = y.reshape(num_steps, -1)
    d2 = d2.reshape(num_steps, -1)

    # create grid along d2
    grid = y + d2 * epsilons
    grid = grid.reshape(shape)

    return grid


def flatten_batch(
    batch: Union[Tensor, Sequence[Tensor]],
) -> Tensor:
    if isinstance(batch, Sequence):
        batch = [x for x in batch if x is not None]
        batch = torch.cat(batch, dim=0)

    batch = batch.reshape(len(batch), -1)

    return batch


def repeat(
    x: Tensor,
    repeat: int,
    dim: int = 0,
) -> Tensor:
    x = x.unsqueeze(dim)
    output_shape = torch.ones(size=(len(x.shape),), dtype=int)
    output_shape[dim] = repeat

    return x.repeat(*output_shape)


def stretch(
    x: Tensor,
    strength: int,
    dim: Union[int, Sequence[int]] = 0,
) -> Tensor:
    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        x = x.repeat_interleave(repeats=strength, dim=d)

    return x
