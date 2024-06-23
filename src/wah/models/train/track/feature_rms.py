import math

import torch

from ....typing import (
    Dict,
    List,
    Module,
    SummaryWriter,
    Tensor,
    Tuple,
)
from ...feature_extractor import FeatureExtractor
from ....utils.tensor import flatten_batch

__all__ = [
    "init",
    "compute",
    "track",
]


def init(
    model: Module,
) -> Tuple[
    FeatureExtractor,
    Dict[str, List[Tensor]],
    Dict[str, List[Tensor]],
]:
    feature_extractor = FeatureExtractor(model)

    train_rms: Dict[str, List[Tensor]] = {}
    val_rms: Dict[str, List[Tensor]] = {}

    return feature_extractor, train_rms, val_rms


def compute(
    data: Tensor,
    feature_extractor: FeatureExtractor,
    feature_rms_dict: Dict[str, List[Tensor]],
) -> None:
    with torch.no_grad():
        features: Dict[str, Tensor] = feature_extractor(data)

    for i_layer, feature in features.items():
        feature = flatten_batch(feature)

        f_rms = torch.norm(feature, p=2, dim=-1) / math.sqrt(feature.size(-1))
        feature_rms_dict[i_layer].append(f_rms)

        del feature
        torch.cuda.empty_cache()


def track(
    epoch: int,
    tensorboard: SummaryWriter,
    feature_rms_dict: Dict[str, List[Tensor]],
    header: str,
) -> None:
    for i_layer, f_rms in feature_rms_dict.items():
        f_rms = torch.cat(f_rms)
        tensorboard.add_histogram(
            tag=f"{header}/{i_layer}",
            values=f_rms,
            global_step=epoch,
        )

        feature_rms_dict[i_layer].clear()
