import lightning as L
import torch
from torch.nn import CrossEntropyLoss, Softmax

from .... import utils
from ....typing import (
    Config,
    Dataset,
    Devices,
    Dict,
    List,
    Module,
    Optional,
    Tensor,
    Tuple,
)
from ...datasets import to_dataloader
from ...train.utils import load_accelerator_and_devices
from .direction import generate_travel_directions

__all__ = [
    "TravelTest",
]


def test_data(
    data: Tensor,
    targets: Tensor,
    model: Module,
) -> Tuple[List[bool], List[float]]:
    with torch.no_grad():
        outputs: Tensor = model(data)

    confs: Tensor = Softmax(dim=-1)(outputs)
    preds: Tensor = torch.argmax(confs, dim=-1)

    results: Tensor = torch.eq(preds, targets)
    signed_confs: Tensor = confs[:, targets].diag() * (results.int() - 0.5).sign()

    return (
        [bool(result) for result in results.cpu()],
        [float(conf) for conf in signed_confs.cpu()],
    )


def travel_data(
    x: Tensor,
    t: Tensor,
    d: Tensor,
    model: Module,
    params: Dict[str, float],
) -> float:
    x = torch.unsqueeze(x, dim=0)
    _x = x

    # init params
    """
    epsilon: travel length
    stride: travel stride
    _stride: original stride
    stride_decay: ratio to decay stride
    correct: current result of travel
    _correct: previous result of travel
    flag:
        0: must travel further
        1: crossed the decision boundary; go back
        -1: diverged
    """
    eps: float = params["init_eps"]
    stride: float = params["stride"]
    _stride: float = params["stride"]
    stride_decay: float = params["stride_decay"]
    correct: bool = True
    _correct: bool = None
    flag: int = 0

    bound = params["bound"]
    tol = params["tol"]
    max_iter = params["max_iter"]
    turnaround = params["turnaround"]

    for i in range(max_iter):
        _correct = correct

        _x = x + d * eps

        if bound:
            _x = _x.clamp(0, 1)

        correct = test_data(_x, t, model)[0][0]
        # print(f'[{self.epsilon}, {self.stride}] {bool(self._correct)}->{bool(self.correct)}')
        (
            eps,
            stride,
            _stride,
            stride_decay,
            correct,
            _correct,
            flag,
        ) = update(
            i,
            eps,
            stride,
            _stride,
            stride_decay,
            correct,
            _correct,
            flag,
            max_iter,
            turnaround,
        )

        # diverged
        if flag == -1:
            return -1.0

        # finished travel
        if stride < tol and not correct:
            return eps

    # not fully converged
    return -eps


def update(
    iteration: int,
    eps: float,
    stride: float,
    _stride: float,
    stride_decay: float,
    correct: bool,
    _correct: bool,
    flag: int,
    max_iter: int,
    turnaround: int,
) -> Tuple[float, float, float, float, bool, bool, int]:
    # just crossed the decision boundary
    if correct != _correct:
        flag = 1

        eps = eps - stride  # go back
        stride = stride * stride_decay  # and travel again with smaller stride

    # not reached the decision boundary yet
    else:
        flag = 0

        eps = eps + stride  # travel further

    # epsilon must be over 0
    if eps < 0.0:
        eps = 0.0

    # divergence test
    if (iteration + 1) == max_iter // (1 / turnaround) and stride == _stride:
        flag = -1

    return (
        eps,
        stride,
        _stride,
        stride_decay,
        correct,
        _correct,
        flag,
    )


def work(
    i: int,
    data: Tensor,
    targets: Tensor,
    directions: Tensor,
    results: List[bool],
    signed_confs: List[float],
    model: Module,
    params: Dict[str, float],
) -> Tuple[int, float, float]:
    gt = int(targets[i])
    correct = results[i]
    conf = signed_confs[i]
    x = data[i]
    t = targets[i]
    d = directions[i]

    eps = 0.0

    if not correct:
        return (gt, conf, eps)

    eps = travel_data(x, t, d, model, params)

    return (gt, conf, eps)


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        config: Config,
        res_dict: Dict[str, List],
    ) -> None:
        super().__init__()

        self.model = model
        self.config = config
        self.res_dict = res_dict

        self.criterion = CrossEntropyLoss(
            reduction="none",
            label_smoothing=self.config["label_smoothing"],
        )
        self.softmax = Softmax(dim=-1)
        self.method = self.config["method"]

    def test_step(self, batch, batch_idx):
        data: Tensor
        targets: Tensor

        data, targets = batch
        directions = generate_travel_directions(
            data=data,
            method=self.method,
            targets=targets,
            model=self.model,
            device=self.global_rank,
        )

        outputs: Tensor = self.model(data)

        losses: Tensor = self.criterion(outputs, targets)

        confs: Tensor = self.softmax(outputs)
        preds: Tensor = torch.argmax(confs, dim=-1)

        results: Tensor = torch.eq(preds, targets)
        signed_confs: Tensor = confs[:, targets].diag() * (results.int() - 0.5).sign()

        self.res_dict["gt"].append(targets.cpu())
        self.res_dict["pred"].append(preds.cpu())
        self.res_dict["loss"].append(losses.cpu())
        self.res_dict["conf"].append(signed_confs.cpu())

    def on_test_epoch_end(self) -> None:
        pass
