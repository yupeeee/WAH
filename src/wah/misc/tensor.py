import torch

from .typing import Literal, Sequence, Tensor, Union

__all__ = [
    "clean",
    "create_1d_traj",
    "create_2d_grid",
    "flatten_batch",
    "mat_elmul_vec",
    "new",
    "repeat",
    "stretch",
]


def clean(
    x: Tensor,
) -> Tensor:
    """
    Cleans a tensor by replacing special values (NaN, +/-inf) with appropriate finite values.

    ### Args
    - `x` (Tensor): The tensor to clean

    ### Returns
    - `Tensor`: The cleaned tensor with special values replaced:
        - NaN -> 0
        - +inf -> dtype max value
        - -inf -> dtype min value

    ### Example
    ```python
    >>> x = torch.tensor([1.0, float('nan'), float('inf'), float('-inf')])
    >>> clean(x)
    tensor([ 1.0000e+00, 0.0000e+00, 3.4028e+38, -3.4028e+38])
    ```
    """
    # Handle all special values
    is_nan = torch.isnan(x)
    is_posinf = torch.isposinf(x)
    is_neginf = torch.isneginf(x)

    # Get dtype limits
    dtype_info = torch.finfo(x.dtype)

    # Create replacement mask and values
    mask = is_nan | is_posinf | is_neginf
    values = torch.zeros_like(x)
    values[is_posinf] = dtype_info.max
    values[is_neginf] = dtype_info.min

    # Apply replacements
    x = torch.where(mask, values, x)

    return x


def create_1d_traj(
    x: Tensor,
    d: Tensor,
    num_steps: int,
    eps: float,
) -> Tensor:
    """
    Creates a 1D trajectory by perturbing a point along a direction vector.

    ### Args
    - `x` (Tensor): The starting point tensor
    - `d` (Tensor): The direction vector tensor
    - `num_steps` (int): Number of points to generate along the trajectory
    - `eps` (float): Maximum perturbation magnitude

    ### Returns
    - `Tensor`: A tensor of shape (num_steps, *x.shape) containing the trajectory points

    ### Example
    ```python
    >>> x = torch.arange(3.)  # Starting point [0,1,2]
    >>> d = torch.linspace(0.1, 0.3, 3)  # Direction vector [0.1,0.2,0.3]
    >>> traj = create_1d_traj(x, d, num_steps=5, eps=0.1)
    >>> traj.shape
    torch.Size([5, 3])
    >>> traj
    tensor([[-0.0100,  0.9800,  1.9700],
            [-0.0050,  0.9900,  1.9850],
            [ 0.0000,  1.0000,  2.0000],
            [ 0.0050,  1.0100,  2.0150],
            [ 0.0100,  1.0200,  2.0300]])
    ```
    """
    # Create evenly spaced perturbations
    epsilons = torch.linspace(-eps, eps, num_steps).view(-1, 1)

    # Expand x and d to match epsilon dimensions
    x_expanded = x.unsqueeze(0).expand(num_steps, *x.shape)
    d_expanded = d.unsqueeze(0).expand(num_steps, *d.shape)

    # Compute trajectory
    traj = x_expanded + d_expanded * epsilons

    return traj


def create_2d_grid(
    x: Tensor,
    d1: Tensor,
    d2: Tensor,
    num_steps: int,
    eps: float,
) -> Tensor:
    """
    Creates a 2D grid by perturbing a point along two direction vectors.

    ### Args
    - `x` (Tensor): The starting point tensor
    - `d1` (Tensor): First direction vector tensor
    - `d2` (Tensor): Second direction vector tensor
    - `num_steps` (int): Number of points to generate along each direction
    - `eps` (float): Maximum perturbation magnitude

    ### Returns
    - `Tensor`: A tensor of shape (num_steps * num_steps, *x.shape) containing the grid points

    ### Example
    ```python
    >>> x = torch.zeros(2)  # Starting point [0,0]
    >>> d1 = torch.tensor([1., 0])  # First direction [1,0]
    >>> d2 = torch.tensor([0, 1.])  # Second direction [0,1]
    >>> grid = create_2d_grid(x, d1, d2, num_steps=3, eps=1.0)
    >>> grid.shape
    torch.Size([9, 2])
    >>> grid
    tensor([[-1., -1.],
            [ 0., -1.],
            [ 1., -1.],
            [-1.,  0.],
            [ 0.,  0.],
            [ 1.,  0.],
            [-1.,  1.],
            [ 0.,  1.],
            [ 1.,  1.]])
    ```
    """
    # Create evenly spaced perturbations for d1 and d2
    epsilons = torch.linspace(-eps, eps, num_steps).view(-1, 1)

    # Expand x and direction vectors to match grid dimensions
    x_expanded = x.unsqueeze(0).expand(num_steps * num_steps, *x.shape)
    d1_expanded = d1.unsqueeze(0).expand(num_steps * num_steps, *d1.shape)
    d2_expanded = d2.unsqueeze(0).expand(num_steps * num_steps, *d2.shape)

    # Create epsilon grids
    eps1 = epsilons.repeat(num_steps, 1)  # Repeat for d1
    eps2 = epsilons.repeat_interleave(num_steps, dim=0)  # Interleave for d2

    # Compute grid points by perturbing along d1 and d2
    grid = x_expanded + d1_expanded * eps1 + d2_expanded * eps2

    return grid


def flatten_batch(
    batch: Union[Tensor, Sequence[Tensor]],
) -> Tensor:
    """Flattens a batch of tensors into a 2D tensor.

    ### Args
    - `batch` (Union[Tensor, Sequence[Tensor]]): A tensor or sequence of tensors to flatten

    ### Returns
    - `Tensor`: A flattened 2D tensor with shape (batch_size, -1)

    ### Example
    ```python
    >>> x = torch.randn(10, 3, 32, 32)  # Single tensor
    >>> flattened = flatten_batch(x)
    >>> flattened.shape
    torch.Size([10, 3072])

    >>> xs = [torch.randn(5, 3, 32, 32), torch.randn(3, 3, 32, 32)]  # Sequence
    >>> flattened = flatten_batch(xs)
    >>> flattened.shape
    torch.Size([8, 3072])
    ```
    """
    if isinstance(batch, Sequence):
        batch = torch.cat([x for x in batch if x is not None], dim=0)
    return batch.view(batch.size(0), -1)


def mat_elmul_vec(
    mat: Tensor,
    vec: Tensor,
    dim: int,
) -> Tensor:
    """Performs element-wise multiplication between a matrix and a vector along a specified dimension.

    ### Args
    - `mat` (Tensor): Input matrix/tensor to multiply
    - `vec` (Tensor): Vector to multiply with
    - `dim` (int): Dimension along which to multiply

    ### Returns
    - `Tensor`: Result of element-wise multiplication

    ### Example
    ```python
    >>> mat = torch.ones(2, 3, 4)  # 2x3x4 tensor of ones
    >>> vec = torch.arange(3)      # Vector [0,1,2]
    >>> result = mat_elmul_vec(mat, vec, dim=1)
    >>> result.shape
    torch.Size([2, 3, 4])
    >>> result[0,0,0]  # 1 * 0 = 0
    tensor(0.)
    >>> result[0,1,0]  # 1 * 1 = 1
    tensor(1.)
    >>> result[0,2,0]  # 1 * 2 = 2
    tensor(2.)
    ```
    """
    # Check dimensions match
    assert (
        mat.shape[dim] == vec.shape[0]
    ), f"Shape mismatch at dim {dim}: {mat.shape[dim]} != {vec.shape[0]}"

    # Create shape for broadcasting
    expand_shape = [1] * len(mat.shape)
    expand_shape[dim] = vec.shape[0]

    # Reshape vec for broadcasting along specified dimension
    vec = vec.view(*expand_shape)

    return mat * vec


def new(
    strategy: Literal[
        "empty",
        "full",
        "ones",
        "rand",
        "randn",
        "zeros",
    ],
    **kwargs,
) -> Tensor:
    """Create a new tensor using the specified strategy.

    ### Args
        - `strategy` (Literal["empty", "full", "ones", "rand", "randn", "zeros"]): Strategy to use for tensor creation
        - `**kwargs`: Additional arguments passed to the tensor creation function

    ### Returns
        - `torch.Tensor`: Newly created tensor

    ### Example
    ```python
    >>> t = new("rand", size=(2, 3))  # Random uniform [0,1] tensor of shape (2,3)
    >>> t = new("randn", size=(2, 3)) # Random normal tensor of shape (2,3)
    >>> t = new("ones", size=(2, 3))  # Tensor of ones with shape (2,3)
    >>> t = new("zeros", size=(2, 3)) # Tensor of zeros with shape (2,3)
    >>> t = new("empty", size=(2, 3))  # Uninitialized tensor of shape (2,3)
    >>> t = new("full", fill_value=5, size=(2, 3)) # Tensor filled with value 5
    ```
    """
    if strategy == "rand":
        return torch.rand(**kwargs)
    elif strategy == "randn":
        return torch.randn(**kwargs)
    elif strategy == "ones":
        return torch.ones(**kwargs)
    elif strategy == "zeros":
        return torch.zeros(**kwargs)
    elif strategy == "empty":
        return torch.empty(**kwargs)
    elif strategy == "full":
        return torch.full(**kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def repeat(
    x: Tensor,
    repeat: int,
    dim: int = 0,
) -> Tensor:
    """Repeats a tensor along a specified dimension.

    ### Args
    - `x` (Tensor): Input tensor to repeat
    - `repeat` (int): Number of times to repeat the tensor
    - `dim` (int, optional): Dimension along which to repeat. Defaults to 0

    ### Returns
    - `Tensor`: Tensor repeated along specified dimension

    ### Example
    ```python
    >>> x = torch.tensor([[1, 2], [3, 4]])
    >>> repeat(x, 3, dim=0)
    tensor([[[1, 2],
            [3, 4]],

            [[1, 2],
            [3, 4]],

            [[1, 2],
            [3, 4]]])
    ```
    """
    x = x.unsqueeze(dim)
    output_shape = [1] * len(x.shape)
    output_shape[dim] = repeat
    return x.repeat(*output_shape)


def stretch(
    x: Tensor,
    strength: int,
    dim: Union[int, Sequence[int]] = 0,
) -> Tensor:
    """Stretches a tensor by repeating each element along specified dimension(s).

    ### Args
    - `x` (Tensor): Input tensor to stretch
    - `strength` (int): Number of times to repeat each element
    - `dim` (Union[int, Sequence[int]], optional): Dimension(s) along which to stretch. Defaults to 0

    ### Returns
    - `Tensor`: Tensor with elements repeated along specified dimension(s)

    ### Example
    ```python
    >>> x = torch.tensor([[1, 2], [3, 4]])
    >>> stretch(x, 2, dim=0)
    tensor([[1, 2],
            [1, 2],
            [3, 4],
            [3, 4]])
    ```
    """
    if isinstance(dim, int):
        dim = [dim]
    for d in dim:
        x = x.repeat_interleave(repeats=strength, dim=d)
    return x
