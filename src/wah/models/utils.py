import torch

from ..typing import (
    Tensor,
)

__all__ = [
    "flatten_feature",
]


def flatten_feature(feature, batch_size: int, ) -> Tensor:
    # vit: self_attention
    if isinstance(feature, tuple):
        feature = [f for f in feature if f is not None]
        feature = torch.cat(feature, dim=0)

    feature = feature.reshape(batch_size, -1)

    return feature
