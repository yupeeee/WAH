from typing import List

import torch

__all__ = [
    "BlockActivations",
]


class BlockActivations:
    """
    Hook that blocks the activations of a linear layer by setting them to 0.0.
    """

    def __init__(
        self,
        indices: List[int],
    ) -> None:
        self.indices = indices
        self.active = True

    def __call__(
        self,
        module: torch.nn.Module,
        input: torch.Tensor,
        output: torch.Tensor,
    ) -> torch.Tensor:
        if self.active is False or len(self.indices) == 0:
            return output

        output[:, :, self.indices] = 0.0

        return output

    def activate(self) -> None:
        self.active = True

    def deactivate(self) -> None:
        self.active = False

    def set_indices(self, indices: List[int]) -> None:
        self.indices = indices
