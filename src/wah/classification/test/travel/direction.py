import torch

from ....typing import Device, Module, Optional, Tensor
from ...attacks import FGSM

__all__ = [
    "methods",
    "generate_travel_directions",
]


def normalize_directions(
    directions: Tensor,
) -> Tensor:
    """||d|| ** 2 = dim(d)"""
    dim = directions.flatten().size()[0]
    directions = directions / torch.norm(directions.flatten(), p=2) * dim**0.5

    return directions


def generate_fgsm_directions(
    data: Tensor,
    targets: Tensor,
    model: Module,
    device: Device,
) -> Tensor:
    fgsm = FGSM(model, 1.0, device)
    grads = fgsm.grad(data, targets)

    directions = grads.sign()
    # directions = normalize_directions(directions)

    return directions


def generate_random_directions(
    data: Tensor,
) -> Tensor:
    directions = torch.randn_like(data)
    directions = normalize_directions(directions)

    return directions


def generate_signed_random_directions(
    data: Tensor,
) -> Tensor:
    directions = torch.randn_like(data).sign().float()
    # directions = normalize_directions(directions)

    return directions


methods = [
    "fgsm",
    "random",
    "signed_random",
]


def generate_travel_directions(
    data: Tensor,
    method: str = "random",
    targets: Optional[Tensor] = None,
    model: Optional[Module] = None,
    device: Optional[Device] = "auto",
) -> Tensor:
    """
    Generates travel directions based on the specified method.

    ### Parameters
    - `data` (Tensor): The input data tensor.
    - `method` (str): The method to use for generating directions. Options are "fgsm", "random", and "signed_random". Defaults to "random".
    - `targets` (Optional[Tensor]): The target labels tensor, required if `method` is "fgsm". Defaults to None.
    - `model` (Optional[Module]): The model to be attacked, required if `method` is "fgsm". Defaults to None.
    - `device` (Device): The device to perform the computations on. Defaults to "auto".

    ### Returns
    - `Tensor`: The generated travel directions.

    ### Raises
    - `AssertionError`: If the specified method is not supported, or if required arguments for the selected method are not provided.
    """
    assert method in methods, f"Unsupported travel method: {method}"

    if method == "fgsm":
        assert (
            targets is not None
        ), "No targets are given; unable to generate travel directions using fgsm"
        assert (
            model is not None
        ), "No model is given; unable to generate travel directions using fgsm"

        return generate_fgsm_directions(
            data=data,
            targets=targets,
            model=model,
            device=device,
        )
    elif method == "random":
        return generate_random_directions(
            data=data,
        )
    elif method == "signed_random":
        return generate_signed_random_directions(
            data=data,
        )
    else:
        raise
