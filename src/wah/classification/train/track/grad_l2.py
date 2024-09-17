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
    grad_l2_dict: Dict[str, List[Tensor]] = {}

    for i, (layer, _) in enumerate(model.named_parameters()):
        grad_l2_dict[f"{i}_{layer}"] = []

    return grad_l2_dict


def compute(
    model: Module,
    grad_l2_dict: Dict[str, List[Tensor]],
) -> None:
    for i, (layer, param) in enumerate(model.named_parameters()):
        grad_l2 = torch.norm(param.grad.flatten(), p=2)
        grad_l2_dict[f"{i}_{layer}"].append(grad_l2.view(1))


def track(
    epoch: int,
    tensorboard: SummaryWriter,
    grad_l2_dict: Dict[str, List[Tensor]],
    header: str,
) -> None:
    for i_layer, l2 in grad_l2_dict.items():
        l2 = torch.cat(l2)
        tensorboard.add_histogram(
            tag=f"{header}/{i_layer}",
            values=l2,
            global_step=epoch,
        )


def reset(
    grad_l2_dict: Dict[str, List[Tensor]],
) -> None:
    for i_layer, _ in grad_l2_dict.items():
        grad_l2_dict[i_layer].clear()
