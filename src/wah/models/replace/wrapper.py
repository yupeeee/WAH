from torch import nn

from ...typing import (
    Module,
    Optional,
    Sequence,
    Tensor,
)

__all__ = [
    "PermuteWrapper",
    "Conv1x1Wrapper",
]


class PermuteWrapper(Module):
    def __init__(
        self,
        module: Module,
        dims: Sequence[int],
    ) -> None:
        super().__init__()

        self.module = module
        self.dims = dims

        if hasattr(module, "weight"):
            self.weight = self.module.weight

        if hasattr(module, "bias"):
            self.bias = self.module.bias

    def forward(
        self,
        x: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        squeeze = False

        if x.dim() == 3:
            x = x.unsqueeze(dim=0)
            squeeze = True

        x = x.permute(*self.dims)
        x = self.module(x, *args, **kwargs)
        x = x.permute(*self.original_dims())

        if squeeze:
            x = x[0]

        return x

    def original_dims(
        self,
    ) -> Sequence[int]:
        return sorted(range(len(self.dims)), key=lambda i: self.dims[i])


class Conv1x1Wrapper(Module):
    def __init__(
        self,
        module: Module,
        channel_dim: int,
        permute_dims: Optional[Sequence[int]] = None,
    ) -> None:
        super().__init__()

        self.module = module
        self.permute_dims = permute_dims

        self.conv1x1 = nn.Conv2d(
            in_channels=channel_dim,
            out_channels=channel_dim,
            kernel_size=1,
        )

        if permute_dims is not None:
            self.conv1x1 = PermuteWrapper(
                module=self.conv1x1,
                dims=permute_dims,
            )

        if hasattr(module, "weight"):
            self.weight = self.module.weight

        if hasattr(module, "bias"):
            self.bias = self.module.bias

    def forward(
        self,
        x: Tensor,
        *args,
        **kwargs,
    ) -> Tensor:
        x = self.conv1x1(x)
        x = self.module(x, *args, **kwargs)

        return x
