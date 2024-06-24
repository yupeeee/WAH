import torch
import tqdm
from torch import nn

from ...attacks.fgsm import FGSM
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
from ...utils.random import (
    seed_everything,
    unseed_everything,
)

__all__ = [
    "Traveler",
]

travel_methods = [
    "fgsm",
]


class DirectionGenerator:
    """
    Generates travel directions using specified methods.
    For further understanding of the idea of [*travel*]() and [*direction*](),
    refer to [**link**](https://arxiv.org/abs/2210.05742).

    ### Parameters
    - `model` (Module):
      The model to generate directions for.
    - `method` (Literal["fgsm"], optional):
      The method to use for generating travel directions.
      Defaults to "fgsm".
    - `seed` (int, optional):
      The seed for random number generation.
      Defaults to -1 (No seeding).
    - `use_cuda` (bool, optional):
      Whether to use CUDA for computation.
      Defaults to False.

    ### Methods
    - `__call__`:
      Generates travel directions for the given data and targets.
    """

    def __init__(
        self,
        model: Module,
        method: Literal["fgsm",] = "fgsm",
        seed: Optional[int] = -1,
        use_cuda: Optional[bool] = False,
    ) -> None:
        """
        - `model` (Module):
          The model to generate directions for.
        - `method` (Literal["fgsm"], optional):
          The method to use for generating travel directions.
          Defaults to "fgsm".
        - `seed` (int, optional):
          The seed for random number generation.
          Defaults to -1 (No seeding).
        - `use_cuda` (bool, optional):
          Whether to use CUDA for computation.
          Defaults to False.
        """
        assert (
            method in travel_methods
        ), f"Expected method to be one of {travel_methods}, got {method}"

        self.model = model
        self.method = method
        self.seed = seed
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.model.to(self.device)

        seed_everything(self.seed)

    def __call__(
        self,
        data: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Generates travel directions for the given data and targets.

        ### Parameters
        - `data` (Tensor):
        Data for travel.
        - `targets` (Tensor):
        Targets for travel.

        ### Returns
        - `Tensor`:
          Generated travel directions.
        """
        if self.method == "fgsm":
            signed_grads = (
                FGSM(
                    model=self.model,
                    epsilon=-1.0,  # dummy value
                    use_cuda=self.use_cuda,
                )
                .grad(data, targets)
                .sign()
            )

            directions = signed_grads

        else:
            raise

        if self.seed > -1:
            unseed_everything()

        return directions.to(torch.device("cpu"))


def test_data(
    data: Tensor,
    targets: Tensor,
    model: Module,
    device: Device,
) -> Tuple[List[bool], List[float]]:
    """
    Tests the data on the model and returns the results and confidences.

    ### Parameters
    - `data` (Tensor):
      The input data.
    - `targets` (Tensor):
      The target labels.
    - `model` (Module):
      The model to test the data on.
    - `device` (Device):
      The device to run the model on.

    ### Returns
    - `Tuple[List[bool], List[float]]`:
      A tuple containing a list of boolean results indicating correctness and a list of signed confidences.

    ### Notes
    - This function uses softmax to get the confidence scores and computes the results by comparing predictions with targets.
    """
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
    """
    Travels the data in the direction to find the decision boundary.

    ### Parameters
    - `x` (Tensor):
      The input data.
    - `t` (Tensor):
      The target label.
    - `d` (Tensor):
      The direction tensor.
    - `model` (Module):
      The model to test the data on.
    - `device` (Device):
      The device to run the model on.
    - `params` (Dict[str, float]):
      Parameters for the travel algorithm.

    ### Returns
    - `float`:
      The distance traveled to the decision boundary.

    ### Notes
    - This function iteratively updates the distance traveled to find the decision boundary.
    - It uses the `test_data` function to check the correctness of the data at each step.
    """
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
    """
    Updates the travel parameters based on the current state.

    ### Parameters
    - `iteration` (int):
      The current iteration number.
    - `eps` (float):
      The current epsilon value.
    - `stride` (float):
      The current stride value.
    - `_stride` (float):
      The original stride value.
    - `stride_decay` (float):
      The ratio to decay the stride.
    - `correct` (bool):
      The current correctness of the travel.
    - `_correct` (bool):
      The previous correctness of the travel.
    - `flag` (int):
      The current flag indicating the travel state.
    - `max_iter` (int):
      The maximum number of iterations.
    - `turnaround` (int):
      The ratio to determine if the travel has diverged.

    ### Returns
    - `Tuple[float, float, float, float, bool, bool, int]`:
      Updated values for epsilon, stride, original stride, stride decay, current correctness, previous correctness, and flag.

    ### Notes
    - This function adjusts the travel parameters based on whether the decision boundary was crossed.
    - It also checks for divergence based on the turnaround ratio.
    """
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
    """
    Performs the travel work for a single data point.

    ### Parameters
    - `i` (int):
      The index of the data point.
    - `data` (Tensor):
      The input data.
    - `targets` (Tensor):
      The target labels.
    - `directions` (Tensor):
      The direction tensor.
    - `results` (List[bool]):
      The list of correctness results.
    - `signed_confs` (List[float]):
      The list of signed confidences.
    - `model` (Module):
      The model to test the data on.
    - `device` (Device):
      The device to run the model on.
    - `params` (Dict[str, float]):
      Parameters for the travel algorithm.

    ### Returns
    - `Tuple[int, float, float]`:
      A tuple containing the ground truth label, confidence, and epsilon distance.

    ### Notes
    - This function handles the travel process for a single data point, updating the travel distance as needed.
    """
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
    """
    Class to handle traveling data to find decision boundaries.
    For further understanding of the idea of [*travel*](),
    refer to [**link**](https://arxiv.org/abs/2210.05742).

    ### Parameters
    - `model` (Module):
      The model to test the data on.
    - `method` (Literal["fgsm"], optional):
      The method to use for generating directions.
      Defaults to "fgsm".
    - `seed` (int, optional):
      The seed for random number generation.
      Defaults to 0.
    - `use_cuda` (bool, optional):
      Whether to use CUDA for computation.
      Defaults to False.
    - `init_eps` (float, optional):
      Initial length of travel.
      Defaults to 1.0e-3.
    - `stride` (float, optional):
      Travel stride.
      Defaults to 1.0e-3.
    - `stride_decay` (float, optional):
      Ratio to decay stride.
      Defaults to 0.5.
    - `bound` (bool, optional):
      Whether to clip data to [0, 1].
      Defaults to True.
    - `tol` (float, optional):
      Tolerance for early stop.
      Defaults to 1.0e-10.
    - `max_iter` (int, optional):
      Maximum number of iterations.
      Defaults to 10000.
    - `turnaround` (float, optional):
      Ratio to determine if travel has diverged.
      Defaults to 0.1.

    ### Methods
    - `__call__`:
      Travels through the data to find decision boundaries.

    ### Notes
    - This class utilizes `DirectionGenerator` for generating adversarial directions.
    - It defines various hyperparameters for the traveling process.
    """

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
        """
        - `model` (Module):
          The model to test the data on.
        - `method` (Literal["fgsm"], optional):
          The method to use for generating directions.
          Defaults to "fgsm".
        - `seed` (int, optional):
          The seed for random number generation.
          Defaults to 0.
        - `use_cuda` (bool, optional):
          Whether to use CUDA for computation.
          Defaults to False.
        - `init_eps` (float, optional):
          Initial length of travel.
          Defaults to 1.0e-3.
        - `stride` (float, optional):
          Travel stride.
          Defaults to 1.0e-3.
        - `stride_decay` (float, optional):
          Ratio to decay stride.
          Defaults to 0.5.
        - `bound` (bool, optional):
          Whether to clip data to [0, 1].
          Defaults to True.
        - `tol` (float, optional):
          Tolerance for early stop.
          Defaults to 1.0e-10.
        - `max_iter` (int, optional):
          Maximum number of iterations.
          Defaults to 10000.
        - `turnaround` (float, optional):
          Ratio to determine if travel has diverged.
          Defaults to 0.1.
        """
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

    def __call__(
        self,
        dataloader: DataLoader,
        verbose: Optional[bool] = True,
    ) -> Dict[str, List[float]]:
        """
        Travels through the data to find decision boundaries.

        ### Parameters
        - `dataloader` (DataLoader):
          The data loader containing the data to be traveled.
        - `verbose` (bool, optional):
          Whether to print progress. Defaults to True.

        ### Returns
        - `Dict[str, List[float]]`:
          A dictionary containing ground truth labels, confidences, and epsilon distances.

        ### Notes
        - This method iterates over the data in the dataloader, generating directions and traveling each data point to find decision boundaries.
        """
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
