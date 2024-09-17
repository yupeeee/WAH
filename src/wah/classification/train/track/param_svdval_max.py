import torch

from ....typing import Dict, List, Module, SummaryWriter, Tensor

__all__ = [
    "init",
    "compute",
    "track",
    "reset",
]


def init(
    model: Module,
) -> Dict[str, List[Tensor]]:
    param_svdval_max_dict: Dict[str, List[Tensor]] = {}

    for i, (layer, _) in enumerate(model.named_parameters()):
        param_svdval_max_dict[f"{i}_{layer}"] = []

    return param_svdval_max_dict


def compute(
    model: Module,
    param_svdval_max_dict: Dict[str, List[Tensor]],
) -> None:
    for i, (layer, param) in enumerate(model.named_parameters()):
        if len(param.shape) == 1:
            param = param.unsqueeze(dim=0)

        svdvals = torch.linalg.svdvals(param)
        svdval_max = torch.max(svdvals).unsqueeze(dim=0)
        param_svdval_max_dict[f"{i}_{layer}"].append(svdval_max)


def track(
    epoch: int,
    tensorboard: SummaryWriter,
    param_svdval_max_dict: Dict[str, List[Tensor]],
    header: str,
) -> None:
    for i_layer, svdval_max in param_svdval_max_dict.items():
        svdval_max = torch.cat(svdval_max)
        tensorboard.add_scalar(
            tag=f"{header}/{i_layer}",
            scalar_value=svdval_max,
            global_step=epoch,
        )


def reset(
    param_svdval_max_dict: Dict[str, List[Tensor]],
) -> None:
    for i_layer, _ in param_svdval_max_dict.items():
        param_svdval_max_dict[i_layer].clear()
