import torch

from ...typing import List, Optional, Sequence, Tensor

__all__ = [
    "process_gathered_data",
]


def process_gathered_data(
    data: List[Tensor],
    unsqueeze_until: int = 1,
    cat_along: int = -1,
    permute_dims: Optional[Sequence[int]] = None,
) -> Tensor:
    assert (
        data[0].dim() <= unsqueeze_until
    ), f"x.dim() in data must be smaller than or equal to unsqueeze_until, got {data[0].dim()}"

    while data[0].dim() != unsqueeze_until:
        data: List[Tensor] = [x.unsqueeze(0) for x in data]

    data: Tensor = torch.cat(data, dim=cat_along)

    if permute_dims is not None:
        data = data.permute(*permute_dims)

    data = data.flatten()

    return data.cpu()
