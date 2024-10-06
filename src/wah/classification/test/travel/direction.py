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
    """
    Normalizes a tensor of directions such that the L2 norm of the flattened directions is equal to the square root of its dimensions.

    ||d|| ** 2 = dim(d)

    ### Parameters
    - `directions` (Tensor): The tensor to normalize.

    ### Returns
    - `Tensor`: The normalized directions tensor.
    """
    dim = directions.flatten().size()[0]
    directions = directions / torch.norm(directions.flatten(), p=2) * dim**0.5

    return directions


def generate_fgsm_directions(
    data: Tensor,
    targets: Tensor,
    model: Module,
    device: Device,
) -> Tensor:
    """
    Generates adversarial directions using the Fast Gradient Sign Method (FGSM).

    ### Parameters
    - `data` (Tensor): The input data.
    - `targets` (Tensor): The target labels corresponding to the data.
    - `model` (Module): The model used for computing the gradient.
    - `device` (Device): The device on which to perform the computations.

    ### Returns
    - `Tensor`: The FGSM-generated directions.
    """
    fgsm = FGSM(model, 1.0, device)
    grads = fgsm.grad(data, targets)

    directions = grads.sign()
    # directions = normalize_directions(directions)

    return directions


def generate_random_directions(
    data: Tensor,
) -> Tensor:
    """
    Generates random directions using a normal distribution.

    ### Parameters
    - `data` (Tensor): The input data tensor to base the random directions on.

    ### Returns
    - `Tensor`: Randomly generated directions with the same shape as the input data.
    """
    directions = torch.randn_like(data)
    directions = normalize_directions(directions)

    return directions


def generate_signed_random_directions(
    data: Tensor,
) -> Tensor:
    """
    Generates random signed directions (i.e., -1 or 1 values).

    ### Parameters
    - `data` (Tensor): The input data tensor to base the signed random directions on.

    ### Returns
    - `Tensor`: Randomly generated signed directions with the same shape as the input data.
    """
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

    The supported methods are:
    - `fgsm`: Uses the Fast Gradient Sign Method (FGSM) to compute adversarial directions.
    - `random`: Generates random directions from a normal distribution.
    - `signed_random`: Generates signed random directions (-1 or 1 values).

    ### Parameters
    - `data` (Tensor): The input data for which to generate travel directions.
    - `method` (str, optional): The method to use for generating directions. Defaults to `"random"`.
    - `targets` (Optional[Tensor], optional): The target labels (required for `fgsm` method). Defaults to `None`.
    - `model` (Optional[Module], optional): The model used for `fgsm` method. Defaults to `None`.
    - `device` (Optional[Device], optional): The device for performing computations. Defaults to `"auto"`.

    ### Returns
    - `Tensor`: The generated travel directions based on the chosen method.

    ### Raises
    - `AssertionError`: If an unsupported method is chosen or required parameters are missing for `fgsm`.
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
