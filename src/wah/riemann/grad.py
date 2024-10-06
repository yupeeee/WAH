import torch
from torch.func import hessian, jacrev

from ..typing import Device, Module, Optional, Tensor

__all__ = [
    "compute_jacobian",
    "compute_hessian",
]


def compute_jacobian(
    model: Module,
    x: Tensor,
    device: Optional[Device] = "cpu",
    reshape: Optional[bool] = False,
) -> Tensor:
    """
    Computes the Jacobian of the model's output with respect to the input `x`.

    ### Parameters
    - `model` (Module): The model whose Jacobian is to be computed.
    - `x` (Tensor): The input tensor.
    - `device` (Device, optional): The device to use for computation. Defaults to `"cpu"`.
    - `reshape` (bool, optional): If `True`, reshapes the Jacobian into a 2D matrix (output_dim, input_dim). Defaults to `False`.

    ### Returns
    - `Tensor`: The computed Jacobian matrix.
    """
    x_dim = x.numel()
    model = model.eval().to(device)
    x = x.to(device)

    J: Tensor = jacrev(model)(x)
    J = J.detach()

    if reshape:
        J = J.reshape(-1, x_dim)

    return J.to(torch.device("cpu"))


def compute_hessian(
    model: Module,
    x: Tensor,
    func=None,
    args=None,
    device: Optional[Device] = "cpu",
    reshape: Optional[bool] = False,
) -> Tensor:
    """
    Computes the Hessian of the model's output with respect to the input `x`.
    Optionally, a custom function `func` can be used to compute the Hessian.

    ### Parameters
    - `model` (Module): The model whose Hessian is to be computed.
    - `x` (Tensor): The input tensor.
    - `func` (optional): A custom function to compute the Hessian for. Defaults to `None`.
    - `args` (optional): Additional arguments to pass to the function `func`. Defaults to `None`.
    - `device` (Device, optional): The device to use for computation. Defaults to `"cpu"`.
    - `reshape` (bool, optional): If `True`, reshapes the Hessian into a 2D matrix (input_dim, input_dim). Defaults to `False`.

    ### Returns
    - `Tensor`: The computed Hessian matrix.
    """
    x_dim = x.numel()

    model = model.eval().to(device)
    x = x.to(device)
    args = [arg.to(device) for arg in args]

    if func is not None:
        args = [arg.to(device) for arg in args] if args is not None else ()
        H: Tensor = hessian(lambda x: func(model(x), *args))(x)
    else:
        H: Tensor = hessian(model)(x)
    H = H.detach()

    if reshape:
        H = H.reshape(x_dim, x_dim)

    return H.to(torch.device("cpu"))
