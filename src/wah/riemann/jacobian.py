import torch

from ..typing import Device, Module, Optional, Tensor

__all__ = [
    "compute_jacobian",
]


def compute_jacobian(
    model: Module,
    x: Tensor,
    device: Optional[Device] = "cpu",
) -> Tensor:
    model = model.eval().to(device)
    x = x.to(device)

    jacobian = torch.autograd.functional.jacobian(model, x)

    return jacobian
