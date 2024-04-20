import torch
import tqdm
from torch import nn

from ..attacks.fgsm import FGSM
from ..typing import (
    DataLoader,
    Dict,
    List,
    Literal,
    Module,
    Optional,
    Tensor,
    Tuple,
)
from ..utils.random import seed_everything

__all__ = [
    "Traveler",
]

methods = [
    "fgsm",
]


class DirectionGenerator:
    def __init__(
        self,
        model: Module,
        method: Literal["fgsm", ] = "fgsm",
        seed: Optional[int] = 0,
        use_cuda: Optional[bool] = False,
    ) -> None:
        assert method in methods, \
            f"Expected method to be one of {methods}, got {method}"

        self.model = model
        self.method = method
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)

        seed_everything(seed)

    def __call__(
        self,
        data: Tensor,
        targets: Tensor,
    ) -> Tensor:
        if self.method == "fgsm":
            signed_grads = FGSM(
                model=self.model,
                epsilon=-1.,                # dummy value
                use_cuda=self.use_cuda,
            ).grad(data, targets).sign()

            directions = signed_grads

        else:
            raise

        return directions.to(torch.device("cpu"))


class Traveler:
    def __init__(
        self,
        model: Module,
        method: Literal["fgsm", ] = "fgsm",
        seed: Optional[int] = 0,
        use_cuda: Optional[bool] = False,
        init_eps: float = 1.e-3,
        stride: float = 1.e-3,
        stride_decay: float = 0.5,
        tol: float = 1.e-10,
        max_iter: int = 10000,
        turnaround: float = 0.1,
    ) -> None:
        assert 0 < stride, \
            f"Expected 0 < stride, got {stride}"
        assert 0 < stride_decay < 1, \
            f"Expected 0 < stride_decay < 1, got {stride_decay}"
        assert 0 < turnaround <= 1, \
            f"Expected 0 < turnaround <= 1, got {turnaround}"

        self.model = model
        self.method = method
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)

        self.softmax = nn.Softmax(dim=-1)

        self.direction_generator = \
            DirectionGenerator(model, method, seed, use_cuda)

        # travel hyperparameters
        self.params: Dict[str, float] = {
            "init_eps": init_eps,
            "stride": stride,
            "stride_decay": stride_decay,
        }
        self.epsilon: float = init_eps              # travel length
        self.stride: float = stride                 # travel stride
        self._stride: float = stride                # original stride
        self.stride_decay: float = stride_decay     # ratio to decay stride
        self.tol: int = tol                         # tolerance for early stop; if stride < tol, stop travel
        self.max_iter: int = max_iter               # maximum number of iterations
        self.turnaround: float = turnaround         # if travel does not cross the decision boundary
                                                    # at turnaround*100% of max_iter, stop travel (diverged)

        self.correct: bool = None                   # current result of travel
        self._correct: bool = None                  # previous result of travel
        self.flag: int = 0                          # flag for travel
                                                    # 0: must travel further
                                                    # 1: crossed the decision boundary; go back
                                                    # -1: diverged

    def travel(
        self,
        dataloader: DataLoader,
        verbose: Optional[bool] = True,
    ) -> Dict[str, float]:
        traveled_res: Dict[str, float] = {
            "gt": [],
            "conf": [],
            "eps": [],
        }

        if verbose:
            print(f"{self.method} travel of {len(dataloader.dataset)} data")

        for batch_idx, (data, targets) in enumerate(dataloader):
            results, signed_confs = self.test_data(data, targets)
            directions = self.direction_generator(data, targets)

            for (target, correct, conf, d, t, direction) in tqdm.tqdm(
                zip(targets, results, signed_confs, data, targets, directions),
                desc=f"BATCH {batch_idx}/{len(dataloader)}",
                disable=not verbose,
            ):
                traveled_res["gt"].append(int(target))
                traveled_res["conf"].append(conf)

                if not correct:
                    traveled_res["eps"].append(0.)
                    continue

                eps = self.travel_data(d, t, direction)
                traveled_res["eps"].append(eps)

                self.reset()

        return traveled_res

    def test_data(
        self,
        data: Tensor,
        targets: Tensor,
    ) -> Tuple[List[bool], List[float]]:
        outputs = self.model(data.to(self.device)).to(torch.device("cpu"))
        confs = self.softmax(outputs.detach())
        preds = torch.argmax(confs, dim=-1)

        results = torch.eq(preds, targets)
        signed_confs = confs[:, targets].diag() * (results.int() - 0.5).sign()

        return [bool(result) for result in results], [float(conf) for conf in signed_confs]

    def travel_data(
        self,
        data: Tensor,
        target: Tensor,
        direction: Tensor,
    ) -> float:
        data = torch.unsqueeze(data, dim=0)
        _data = data

        for i in range(self.max_iter):
            self._correct = self.correct

            _data = data + direction * self.epsilon
            _data = _data.clamp(0, 1)

            self.correct = self.test_data(_data, target)[0][0]
            # print(f'[{self.epsilon}, {self.stride}] {bool(self._correct)}->{bool(self.correct)}')
            self.update(i)

            # diverged
            if self.flag == -1:
                return -1.

            # finished travel
            if self.stride < self.tol and not self.correct:
                return self.epsilon

        # not fully converged
        return -self.epsilon

    def reset(self, ) -> None:
        self.epsilon = self.params["init_eps"]
        self.stride = self.params["stride"]
        self._stride = self.params["stride"]
        self.stride_decay = self.params["stride_decay"]

    def update(self, iteration: int, ) -> None:
        # just crossed the decision boundary
        if self.correct != self._correct:
            self.flag = 1

            self.epsilon = self.epsilon - self.stride       # go back
            self.stride = self.stride * self.stride_decay   # and travel again with smaller stride

        # not reached the decision boundary yet
        else:
            self.flag = 0

            self.epsilon = self.epsilon + self.stride       # travel further

        # epsilon must be over 0
        if self.epsilon < 0.:
            self.epsilon = 0.

        # divergence test
        if (iteration + 1) == self.max_iter // (1 / self.turnaround) \
                and self.stride == self._stride:
            self.flag = -1
