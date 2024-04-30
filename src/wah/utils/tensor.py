import torch

from ..typing import (
    Tensor,
)

__all__ = [
    "isvec",
    "repeat",
    "mat_eprod_vec",
]


def isvec(
    x: Tensor,
) -> bool:
    return len(x.shape) == 1


def repeat(
    x: Tensor,
    repeat: int,
    dim: int = 0,
) -> Tensor:
    x = x.unsqueeze(dim)
    output_shape = torch.ones(size=(len(x.shape),), dtype=int)
    output_shape[dim] = repeat

    return x.repeat(*output_shape)


def mat_eprod_vec(
    mat: Tensor,
    vec: Tensor,
    dim: int,
) -> Tensor:
    assert isvec(vec)
    assert mat.shape[dim] == len(vec)

    # expand vec to match shape with mat
    mat_shape = torch.ones(size=(len(mat.shape),), dtype=int)
    mat_shape[dim] = len(vec)
    vec = vec.view(*mat_shape)

    return mat * vec
