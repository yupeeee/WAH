import torch

from ....tensor import flatten_batch
from ....typing import Dict, List, Module, SummaryWriter, Tensor, Tuple
from ...models.feature_extraction import FeatureExtractor

__all__ = [
    "init",
    "compute",
    "track",
    "reset",
]


def init(
    model: Module,
) -> Tuple[
    FeatureExtractor,
    Dict[str, List[Tensor]],
    Dict[str, List[Tensor]],
]:
    feature_extractor = FeatureExtractor(model)

    train_sign: Dict[str, List[Tensor]] = {}
    val_sign: Dict[str, List[Tensor]] = {}

    return feature_extractor, train_sign, val_sign


def compute(
    data: Tensor,
    feature_extractor: FeatureExtractor,
    feature_sign_dict: Dict[str, List[Tensor]],
) -> None:
    with torch.no_grad():
        features: Dict[str, Tensor] = feature_extractor(data)

    for i_layer, feature in features.items():
        feature = flatten_batch(feature)

        f_sign = torch.sum((feature < 0).int(), dim=-1) / feature.size(-1)
        feature_sign_dict[i_layer].append(f_sign)

        del feature
        torch.cuda.empty_cache()


def track(
    epoch: int,
    tensorboard: SummaryWriter,
    feature_sign_dict: Dict[str, List[Tensor]],
    header: str,
) -> None:
    for i_layer, f_sign in feature_sign_dict.items():
        f_sign = torch.cat(f_sign)
        tensorboard.add_histogram(
            tag=f"{header}/{i_layer}",
            values=f_sign,
            global_step=epoch,
        )


def reset(
    feature_sign_dict: Dict[str, List[Tensor]],
) -> None:
    for i_layer, _ in feature_sign_dict.items():
        feature_sign_dict[i_layer].clear()
