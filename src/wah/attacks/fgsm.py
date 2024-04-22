import torch
from torch import nn

from ..typing import (
    Module,
    Optional,
    Tensor,
)

__all__ = [
    "FGSM",
    "IFGSM",
]


class IFGSM:
    def __init__(
        self,
        model: Module,
        epsilon: float,
        iteration: int,
        alpha: Optional[float] = None,
        use_cuda: Optional[bool] = False,
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.iteration = iteration
        self.alpha = epsilon / iteration if alpha is None else alpha
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.criterion = nn.CrossEntropyLoss()

    def __call__(
        self,
        data: Tensor,
        targets: Tensor,
    ) -> Tensor:
        if self.epsilon == 0.0:
            return data

        _data = data.detach()

        for _ in range(self.iteration):
            _data = self.fgsm(_data, targets, self.alpha)

        return _data.to(torch.device("cpu"))

    def grad(
        self,
        data: Tensor,
        targets: Tensor,
    ) -> Tensor:
        data = data.to(self.device)
        targets = targets.to(self.device)

        data = data.detach()
        data.requires_grad = True

        outputs = self.model(data)
        self.model.zero_grad()
        loss = self.criterion(outputs, targets)
        loss.backward()

        with torch.no_grad():
            grads = data.grad.data

        return grads.detach()

    def fgsm(
        self,
        data: Tensor,
        targets: Tensor,
        epsilon: float,
    ) -> Tensor:
        if epsilon == 0.0:
            return data

        signed_grads = -self.grad(data, targets).sign()

        perturbations = epsilon * signed_grads
        perturbations = perturbations.clamp(-epsilon, epsilon)

        _data = data + perturbations
        _data = _data.clamp(0, 1)

        return _data


class FGSM(IFGSM):
    def __init__(
        self,
        model: Module,
        epsilon: float,
        use_cuda: Optional[bool] = False,
    ) -> None:
        super().__init__(model, epsilon, 1, None, use_cuda)
