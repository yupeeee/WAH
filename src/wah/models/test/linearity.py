# TODO: finish linearity test

from typing import Iterable, List

import numpy as np
import torch
import tqdm

from .utils import _desc

__all__ = [
    "LinearityTest",
]


class LinearityTest:
    def __init__(
        self,
        epsilons: Iterable[float],
        delta: float = 1e-4,
        bound: bool = False,
        use_cuda: bool = False,
        verbose: bool = False,
    ) -> None:
        if not isinstance(epsilons, list):
            epsilons = [float(eps) for eps in epsilons]

        self.epsilons = epsilons
        self.delta = delta
        self.bound = bound
        self.use_cuda = use_cuda
        self.machine = "cuda" if use_cuda else "cpu"
        self.verbose = verbose

        self.sims = list()

    def __call__(
        self,
        model,
        x: torch.Tensor,
        d: torch.Tensor,
    ) -> List[float]:
        self.init_dict()

        for epsilon in tqdm.tqdm(
            self.epsilons,
            desc=_desc("Linearity test", model),
            disable=not self.verbose,
        ):
            self.compute_sim(model, x, d, epsilon)

        return self.sims

    def init_dict(
        self,
    ) -> None:
        self.sims = list()

    def move(
        self,
        x: torch.Tensor,
        d: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        _x = x + eps * d

        if self.bound:
            _x = _x.clamp(0, 1)

        return _x

    def compute_sim(
        self,
        model,
        x: torch.Tensor,
        d: torch.Tensor,
        epsilon: float,
    ) -> None:
        def _sim(
            i: torch.Tensor,
            f1: torch.Tensor,
            f2: torch.Tensor,
            eps: float = 1e-7,
        ) -> float:
            assert len(i.shape) * len(f1.shape) * len(f2.shape) == 1

            v1 = f1 - i
            v2 = f2 - i

            v1 = v1 / (torch.norm(v1, p="fro") + eps)
            v2 = v2 / (torch.norm(v2, p="fro") + eps)

            return float(torch.dot(v1, v2).clamp(-1, 1))

        x_eps = self.move(x, d, epsilon)
        x_eps_l = self.move(x, d, epsilon - self.delta)
        x_eps_r = self.move(x, d, epsilon + self.delta)

        y_eps = model(x_eps.to(self.machine))
        y_eps_l = model(x_eps_l.to(self.machine))
        y_eps_r = model(x_eps_r.to(self.machine))

        sim = _sim(
            i=y_eps.reshape(-1),
            f1=y_eps_l.reshape(-1),
            f2=y_eps_r.reshape(-1),
        )

        self.sims.append(np.pi - sim)
