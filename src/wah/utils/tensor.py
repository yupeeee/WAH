import torch

from ..typing import (
    Tensor,
)

__all__ = [
    "isvec",
    "repeat",
    "mat_eprod_vec",
    "flatten_batch",
]


def isvec(
    x: Tensor,
) -> bool:
    """
    Checks if the given tensor is a vector (1-dimensional).

    ### Parameters
    - `x (Tensor)`: The tensor to check.

    ### Returns
    - `bool`: `True` if the tensor is 1-dimensional, otherwise `False`.

    ### Notes
    - A tensor is considered a vector if it has exactly one dimension.
    """
    return len(x.shape) == 1


def repeat(
    x: Tensor,
    repeat: int,
    dim: int = 0,
) -> Tensor:
    """
    Repeats a tensor along a specified dimension.

    ### Parameters
    - `x (Tensor)`: The tensor to repeat.
    - `repeat (int)`: The number of times to repeat the tensor.
    - `dim (int)`: The dimension along which to repeat the tensor. Defaults to 0.

    ### Returns
    - `Tensor`: The repeated tensor.

    ### Notes
    - This function unsqueezes the tensor along the specified dimension before repeating it.
    - The resulting tensor will have the repeated values along the specified dimension.
    """
    x = x.unsqueeze(dim)
    output_shape = torch.ones(size=(len(x.shape),), dtype=int)
    output_shape[dim] = repeat

    return x.repeat(*output_shape)


def mat_eprod_vec(
    mat: Tensor,
    vec: Tensor,
    dim: int,
) -> Tensor:
    """
    Performs element-wise multiplication of a matrix and a vector along a specified dimension.

    ### Parameters
    - `mat (Tensor)`: The matrix tensor.
    - `vec (Tensor)`: The vector tensor.
    - `dim (int)`: The dimension along which to multiply the matrix with the vector.

    ### Returns
    - `Tensor`: The resulting tensor after element-wise multiplication.

    ### Raises
    - `AssertionError`: If `vec` is not a vector or if the size of `vec` does not match the specified dimension of `mat`.

    ### Notes
    - This function expands the vector to match the shape of the matrix along the specified dimension before performing element-wise multiplication.
    """
    assert isvec(vec)
    assert mat.shape[dim] == len(vec)

    # expand vec to match shape with mat
    mat_shape = torch.ones(size=(len(mat.shape),), dtype=int)
    mat_shape[dim] = len(vec)
    vec = vec.view(*mat_shape)

    return mat * vec


def flatten_batch(batch) -> Tensor:
    """
    Flattens a batch tensor.

    ### Parameters
    - `batch (Union[Tensor, tuple])`: The input batch tensor or tuple of tensors to flatten.

    ### Returns
    - `Tensor`: The flattened batch tensor.

    ### Notes
    - If the input `batch` is a tuple, it concatenates the non-`None` elements along the first dimension.
    - The resulting tensor is reshaped such that the first dimension (batch) is preserved and all other dimensions are flattened.
    """
    # (self-attention) if output is tuple, convert to tensor
    if isinstance(batch, tuple):
        batch = [f for f in batch if f is not None]
        batch = torch.cat(batch, dim=0)

    batch = batch.reshape(len(batch), -1)

    return batch
