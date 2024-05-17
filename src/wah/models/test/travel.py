import torch
import tqdm
from torch import nn

from ...typing import (
    Any,
    DataLoader,
    Device,
    Dict,
    List,
    Literal,
    Module,
    Optional,
    Tensor,
    Tuple,
)
from .utils import DirectionGenerator

__all__ = [
    "Traveler",
]


def test_data(
    data: Tensor,
    targets: Tensor,
    model: Module,
    device: Device,
) -> Tuple[List[bool], List[float]]:
    with torch.no_grad():
        outputs: Tensor = model(data.to(device))

    confs: Tensor = nn.Softmax(dim=-1)(outputs)
    preds: Tensor = torch.argmax(confs, dim=-1)

    results: Tensor = torch.eq(preds, targets.to(device))
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
    device: Device,
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

        correct = test_data(_x, t, model, device)[0][0]
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
    eps,
    stride,
    _stride,
    stride_decay,
    correct,
    _correct,
    flag,
    max_iter: int,
    turnaround: int,
) -> None:
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
    data,
    targets,
    directions,
    results,
    signed_confs,
    model,
    device,
    params,
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

    eps = travel_data(x, t, d, model, device, params)

    return (gt, conf, eps)


class Traveler:
    def __init__(
        self,
        model: Module,
        method: Literal["fgsm",] = "fgsm",
        seed: Optional[int] = 0,
        use_cuda: Optional[bool] = False,
        init_eps: float = 1.0e-3,
        stride: float = 1.0e-3,
        stride_decay: float = 0.5,
        bound: bool = True,
        tol: float = 1.0e-10,
        max_iter: int = 10000,
        turnaround: float = 0.1,
    ) -> None:
        assert 0 < stride, f"Expected 0 < stride, got {stride}"
        assert (
            0 < stride_decay < 1
        ), f"Expected 0 < stride_decay < 1, got {stride_decay}"
        assert 0 < turnaround <= 1, f"Expected 0 < turnaround <= 1, got {turnaround}"

        self.model = model
        self.method = method
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)

        self.softmax = nn.Softmax(dim=-1)

        self.direction_generator = DirectionGenerator(model, method, seed, use_cuda)

        # travel hyperparameters
        """
        init_eps: initial length of travel
        stride: travel stride
        stride decay: ratio to decay stride
        bound: if bound, clip data to [0, 1]
        tol: tolerance for early stop; if stride < tol, stop travel
        max_iter: maximum number of iterations
        turnaround: if travel does not cross the decision boundary
                    at turnaround*100% of max_iter, stop travel (diverged)
        """
        self.params: Dict[str, Any] = {
            "init_eps": init_eps,
            "stride": stride,
            "stride_decay": stride_decay,
            "bound": bound,
            "tol": tol,
            "max_iter": max_iter,
            "turnaround": turnaround,
        }

    def travel(
        self,
        dataloader: DataLoader,
        verbose: Optional[bool] = True,
    ) -> Dict[str, List[float]]:
        traveled_res: Dict[str, List[float]] = {
            "gt": [],
            "conf": [],
            "eps": [],
        }

        if verbose:
            print(f"{self.method} travel of {len(dataloader.dataset)} data")

        for batch_idx, (data, targets) in enumerate(dataloader):
            results, signed_confs = test_data(data, targets, self.model, self.device)
            directions = self.direction_generator(data, targets)

            for i in tqdm.trange(
                len(data),
                desc=f"BATCH {batch_idx + 1}/{len(dataloader)}",
                disable=not verbose,
            ):
                gt, conf, eps = work(
                    i,
                    data,
                    targets,
                    directions,
                    results,
                    signed_confs,
                    self.model,
                    self.device,
                    self.params,
                )

                traveled_res["gt"].append(gt)
                traveled_res["conf"].append(conf)
                traveled_res["eps"].append(eps)

        return traveled_res
