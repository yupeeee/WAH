import torch
from torch import nn

from ..typing import (
    Device,
    Module,
    Optional,
    Tensor,
)

__all__ = [
    "FGSM",
    "IFGSM",
]


class IFGSM:
    """
    [Iterative Fast Gradient Sign Method (IFGSM)](https://openreview.net/forum?id=BJm4T4Kgx) for generating adversarial examples.

    ### Attributes
    - `model` (Module):
      The neural network model to attack.
    - `epsilon` (float):
      The maximum perturbation allowed.
    - `iteration` (int):
      The number of iterations to perform.
    - `alpha` (float):
      The step size for each iteration.
    - `device` (Device):
      The device (CPU or GPU) used for computation.
    - `criterion` (nn.CrossEntropyLoss):
      The loss function used for computing gradients.

    ### Methods
    - `__call__`:
      Generates adversarial examples for the given data and targets.
    - `grad`:
      Computes the gradients of the loss with respect to the input data.

    ### Example
    ```python
    import wah

    dataset = Dataset(...)
    data, target = dataset[0]

    attack = wah.attacks.IFGSM(...)
    data_adv = attack(data.unsqueeze(dim=0), target.unsqueeze(dim=0))
    ```
    """

    def __init__(
        self,
        model: Module,
        epsilon: float,
        iteration: int,
        alpha: Optional[float] = None,
        device: Optional[Device] = "cpu",
    ) -> None:
        """
        - `model` (Module):
          The neural network model to attack.
        - `epsilon` (float):
          The maximum perturbation allowed.
        - `iteration` (int):
          The number of iterations to perform.
        - `alpha` (float, optional):
          The step size for each iteration. Defaults to epsilon/iteration.
        - `device` (Device, optional):
          The device (CPU or GPU) used for computation. Defaults to "cpu".
        """
        self.model = model
        self.epsilon = epsilon
        self.iteration = iteration
        self.alpha = epsilon / iteration if alpha is None else alpha
        self.device = torch.device(device)

        self.criterion = nn.CrossEntropyLoss()

    def __call__(
        self,
        data: Tensor,
        targets: Tensor,
    ) -> Tensor:
        """
        Generates adversarial examples for the given data and targets.

        ### Parameters
        - `data` (Tensor):
          The input data.
        - `targets` (Tensor):
          The target labels.

        ### Returns
        - `Tensor`:
          The adversarial examples generated from the input data.
        """
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
        """
        Computes the gradients of the loss with respect to the input data.

        ### Parameters
        - `data` (Tensor):
          The input data.
        - `targets` (Tensor):
          The target labels.

        ### Returns
        - `Tensor`:
          The gradients of the loss with respect to the input data.
        """
        data = data.to(self.device)
        targets = targets.to(self.device)

        data = data.detach()
        data.requires_grad = True

        outputs = self.model(data)
        self.model.zero_grad()
        loss: Tensor = self.criterion(outputs, targets)
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
        """
        Applies the [Fast Gradient Sign Method (FGSM)](https://arxiv.org/abs/1412.6572) to the input data.

        ### Parameters
        - `data` (Tensor):
          The input data.
        - `targets` (Tensor):
          The target labels.
        - `epsilon` (float):
          The perturbation magnitude.

        ### Returns
        - `Tensor`:
          The perturbed data.
        """
        if epsilon == 0.0:
            return data

        signed_grads = -self.grad(data, targets).sign()

        perturbations = epsilon * signed_grads
        perturbations = perturbations.clamp(-epsilon, epsilon)

        _data = data + perturbations
        _data = _data.clamp(0, 1)

        return _data


class FGSM(IFGSM):
    """
    [Fast Gradient Sign Method (FGSM)](https://arxiv.org/abs/1412.6572) for generating adversarial examples.

    ### Attributes
    - `model` (Module):
      The neural network model to attack.
    - `epsilon` (float):
      The maximum perturbation allowed.
    - `device` (Device):
      The device (CPU or GPU) used for computation.
    - `criterion` (nn.CrossEntropyLoss):
      The loss function used for computing gradients.

    ### Methods
    - `__call__`:
      Generates adversarial examples for the given data and targets.
    - `grad`:
      Computes the gradients of the loss with respect to the input data.

    ### Example
    ```python
    import wah

    dataset = Dataset(...)
    data, target = dataset[0]

    attack = wah.attacks.FGSM(...)
    data_adv = attack(data.unsqueeze(dim=0), target.unsqueeze(dim=0))
    ```
    """

    def __init__(
        self,
        model: Module,
        epsilon: float,
        device: Optional[Device] = "cpu",
    ) -> None:
        """
        - `model` (Module):
          The neural network model to attack.
        - `epsilon` (float):
          The maximum perturbation size allowed.
        - `device` (Device, optional):
          The device (CPU or GPU) used for computation. Defaults to "cpu".
        """
        super().__init__(model, epsilon, 1, None, device)
