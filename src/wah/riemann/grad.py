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
    reshape: Optional[bool] = True,
) -> Tensor:
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
    reshape: Optional[bool] = True,
) -> Tensor:
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
