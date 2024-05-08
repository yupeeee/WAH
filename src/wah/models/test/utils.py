import torch

from ...attacks.fgsm import FGSM
from ...typing import (
    Literal,
    Module,
    Optional,
    Tensor,
)
from ...utils.random import seed_everything

__all__ = [
    "travel_methods",
    "DirectionGenerator",
]

travel_methods = [
    "fgsm",
]


class DirectionGenerator:
    def __init__(
        self,
        model: Module,
        method: Literal["fgsm",] = "fgsm",
        seed: Optional[int] = -1,
        use_cuda: Optional[bool] = False,
    ) -> None:
        assert (
            method in travel_methods
        ), f"Expected method to be one of {travel_methods}, got {method}"

        self.model = model
        self.method = method
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)

        seed_everything(seed)

    def __call__(
        self,
        data: Tensor,
        targets: Tensor,
    ) -> Tensor:
        if self.method == "fgsm":
            signed_grads = (
                FGSM(
                    model=self.model,
                    epsilon=-1.0,  # dummy value
                    use_cuda=self.use_cuda,
                )
                .grad(data, targets)
                .sign()
            )

            directions = signed_grads

        else:
            raise

        return directions.to(torch.device("cpu"))
