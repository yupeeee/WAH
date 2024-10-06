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
    """
    Performs element-wise multiplication of a matrix and a vector,
    broadcasting the vector to match the matrix dimensions.

    ### Parameters
    - `mat` (Tensor): The matrix to be multiplied.
    - `vec` (Tensor): The vector to be broadcasted and multiplied.
    - `dim` (int): The dimension along which to broadcast the vector.

    ### Returns
    - `Tensor`: The result of the element-wise multiplication.
    """
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
    """
    Creates a 1D trajectory by perturbing a tensor `x` along the direction `d`
    over a range of `num_steps` steps.

    ### Parameters
    - `x` (Tensor): The starting point of the trajectory.
    - `d` (Tensor): The direction along which to perturb `x`.
    - `num_steps` (int): The number of steps for the trajectory.
    - `eps` (float): The maximum perturbation.

    ### Returns
    - `Tensor`: The 1D trajectory tensor.
    """
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
    """
    Creates a 2D grid by perturbing a tensor `x` along two directions, `d1` and `d2`,
    over a range of `num_steps` steps.

    ### Parameters
    - `x` (Tensor): The starting point of the grid.
    - `d1` (Tensor): The first direction along which to perturb `x`.
    - `d2` (Tensor): The second direction along which to perturb `x`.
    - `num_steps` (int): The number of steps for each direction.
    - `eps` (float): The maximum perturbation.

    ### Returns
    - `Tensor`: The 2D grid tensor.
    """
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
    """
    Flattens a batch of tensors or a sequence of tensors into a single tensor.

    ### Parameters
    - `batch` (Union[Tensor, Sequence[Tensor]]): A batch of tensors or sequence of tensors.

    ### Returns
    - `Tensor`: The flattened batch.
    """
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
    """
    Repeats a tensor along a specified dimension.

    ### Parameters
    - `x` (Tensor): The tensor to be repeated.
    - `repeat` (int): The number of repetitions.
    - `dim` (int, optional): The dimension along which to repeat. Defaults to `0`.

    ### Returns
    - `Tensor`: The repeated tensor.
    """
    x = x.unsqueeze(dim)
    output_shape = torch.ones(size=(len(x.shape),), dtype=int)
    output_shape[dim] = repeat

    return x.repeat(*output_shape)


def stretch(
    x: Tensor,
    strength: int,
    dim: Union[int, Sequence[int]] = 0,
) -> Tensor:
    """
    Stretches a tensor by repeating elements along specified dimensions.

    ### Parameters
    - `x` (Tensor): The tensor to be stretched.
    - `strength` (int): The number of times to repeat elements.
    - `dim` (Union[int, Sequence[int]], optional): The dimension(s) to stretch. Defaults to `0`.

    ### Returns
    - `Tensor`: The stretched tensor.
    """
    if isinstance(dim, int):
        dim = [dim]

    for d in dim:
        x = x.repeat_interleave(repeats=strength, dim=d)

    return x
