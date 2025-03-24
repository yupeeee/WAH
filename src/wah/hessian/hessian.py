# https://github.com/amirgholami/PyHessian
import numpy as np
import torch

from ..misc.typing import Device, List, Module, Tensor, Tuple

__all__ = [
    "Hessian",
]


def group_product(
    xs: List[Tensor],
    ys: List[Tensor],
) -> Tensor:
    """Compute inner product between two lists of tensors.

    ### Args
        - `xs` (List[Tensor]): First list of tensors
        - `ys` (List[Tensor]): Second list of tensors

    ### Returns
        - `Tensor`: Inner product as a scalar tensor

    ### Example
    ```python
    >>> xs = [torch.tensor([1., 2.]), torch.tensor([3., 4.])]
    >>> ys = [torch.tensor([5., 6.]), torch.tensor([7., 8.])]
    >>> prod = group_product(xs, ys)
    >>> print(prod)
    tensor(70.)  # (1*5 + 2*6) + (3*7 + 4*8)
    ```
    """
    return sum(torch.sum(x * y) for (x, y) in zip(xs, ys))


def group_add(
    xs: List[Tensor],
    x_updates: List[Tensor],
    alpha: float = 1,
) -> List[Tensor]:
    """Update parameters with scaled update: xs = xs + alpha * x_updates.

    ### Args
        - `xs` (List[Tensor]): List of parameter tensors to update
        - `x_updates` (List[Tensor]): List of update tensors
        - `alpha` (float): Scaling factor. Defaults to 1.

    ### Returns
        - `List[Tensor]`: Updated parameters

    ### Example
    ```python
    >>> xs = [torch.ones(2), torch.ones(2)]
    >>> x_updates = [torch.ones(2), torch.ones(2)]
    >>> new_xs = group_add(xs, x_updates, alpha=0.1)
    >>> print(new_xs)
    [tensor([1.1, 1.1]), tensor([1.1, 1.1])]
    ```
    """
    for x, x_update in zip(xs, x_updates):
        x.data.add_(x_update * alpha)
    return xs


def normalization(
    xs: List[Tensor],
) -> List[Tensor]:
    """Normalize a list of vectors.

    ### Args
        - `xs` (List[Tensor]): List of vectors to normalize

    ### Returns
        - `List[Tensor]`: Normalized vectors

    ### Example
    ```python
    >>> xs = [torch.tensor([3., 4.]), torch.zeros(2)]
    >>> norm_xs = normalization(xs)
    >>> print(norm_xs)
    [tensor([0.6, 0.8]), tensor([0., 0.])]
    ```
    """
    norm = torch.sqrt(group_product(xs, xs)).item()
    return [x / (norm + 1e-6) for x in xs]


def get_params_and_grads(
    model: Module,
    loss: Tensor,
) -> Tuple[List[Tensor], List[Tensor]]:
    """Get model parameters and their gradients.

    ### Args
        - `model` (Module): PyTorch model
        - `loss` (Tensor): Loss tensor

    ### Returns
        - `Tuple[List[Tensor], List[Tensor]]`: Tuple of (parameters, gradients)

    ### Example
    ```python
    >>> model = torch.nn.Linear(10, 1)
    >>> loss = model(torch.randn(5, 10)).sum()
    >>> params, grads = get_params_and_grads(model, loss)
    ```
    """
    with torch.autograd.set_grad_enabled(True):
        params = list(model.parameters())
        grads = torch.autograd.grad(loss, params, create_graph=True)

    return params, grads


def hvp(
    gradsH: List[Tensor],
    params: List[Tensor],
    v: List[Tensor],
) -> Tuple[Tensor]:
    """Compute Hessian-vector product.

    ### Args
        - `gradsH` (List[Tensor]): Gradients at current point
        - `params` (List[Tensor]): Model parameters
        - `v` (List[Tensor]): Vector to compute product with

    ### Returns
        - `Tuple[Tensor]`: Hessian-vector product

    ### Example
    ```python
    >>> model = torch.nn.Linear(10, 1)
    >>> x = torch.randn(5, 10)
    >>> loss = model(x).sum()
    >>> params, grads = get_params_and_grads(model, loss)
    >>> v = [torch.randn_like(p) for p in params]
    >>> Hv = hvp(grads, params, v)
    ```
    """
    with torch.autograd.set_grad_enabled(True):
        return torch.autograd.grad(
            gradsH,
            params,
            grad_outputs=v,
            only_inputs=True,
            retain_graph=True,
        )


def orthnormal(
    w: List[Tensor],
    v_list: List[List[Tensor]],
) -> List[Tensor]:
    """Make vector w orthogonal to each vector in v_list and normalize.

    ### Args
        - `w` (List[Tensor]): Vector to orthogonalize
        - `v_list` (List[List[Tensor]]): List of vectors to orthogonalize against

    ### Returns
        - `List[Tensor]`: Orthonormalized vector

    ### Example
    ```python
    >>> w = [torch.tensor([1., 1.]), torch.zeros(2)]
    >>> v = [torch.tensor([1., 0.]), torch.zeros(2)]
    >>> w_ortho = orthnormal(w, [v])
    >>> print(group_product(w_ortho, v))  # Should be close to 0
    tensor(0.)
    ```
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)


class Hessian:
    """Class for computing Hessian properties of neural networks.

    ### Args
        - `model` (Module): Neural network model
        - `criterion` (Module): Loss function
        - `device` (Device): Device to use. Defaults to CPU.

    ### Attributes
        - `model` (Module): Neural network model in eval mode
        - `criterion` (Module): Loss function
        - `device` (Device): Device being used

    ### Example
    ```python
    >>> model = torch.nn.Linear(10, 1)
    >>> criterion = torch.nn.MSELoss()
    >>> x = torch.randn(5, 10)
    >>> y = torch.randn(5, 1)
    >>> hessian = Hessian(model, criterion)
    >>> eigenvals, eigenvecs = hessian.eigenvalues(x, y, top_n=5)
    >>> trace = hessian.trace(x, y)
    >>> density = hessian.density(x, y)
    ```
    """

    def __init__(
        self,
        model: Module,
        criterion: Module,
        device: Device = torch.device("cpu"),
    ) -> None:
        self.model = model.eval()
        self.criterion = criterion
        self.device = device
        self.to(self.device)

    def to(self, device: Device) -> "Hessian":
        self.device = device
        self.model = self.model.to(device)
        self.criterion = self.criterion.to(device)
        return self

    def params_and_grads(
        self,
        inputs: Tensor,
        targets: Tensor,
        **kwargs,
    ) -> Tuple[List[Tensor], List[Tensor]]:
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        outputs = self.model(inputs, **kwargs)
        loss = self.criterion(outputs, targets)
        with torch.autograd.set_grad_enabled(True):
            params = list(self.model.parameters())
            grads = torch.autograd.grad(loss, params, create_graph=True)
        return params, grads

    def eigenvalues(
        self,
        inputs: Tensor,
        targets: Tensor,
        top_n: int = 1,
        max_iter: int = 100,
        tol: float = 1e-3,
        **kwargs,
    ) -> Tuple[List[float], List[List[Tensor]]]:
        """Compute top eigenvalues using power iteration.

        ### Args
            - `inputs` (Tensor): Input tensor
            - `targets` (Tensor): Target tensor
            - `top_n` (int): Number of eigenvalues to compute. Defaults to 1.
            - `max_iter` (int): Maximum iterations per eigenvalue. Defaults to 100.
            - `tol` (float): Convergence tolerance. Defaults to 1e-3.

        ### Returns
            - `Tuple[List[float], List[List[Tensor]]]`: Tuple of (eigenvalues, eigenvectors)

        ### Example
        ```python
        >>> eigenvals, eigenvecs = hessian.eigenvalues(x, y, top_n=5)
        >>> print(f"Top 5 eigenvalues: {eigenvals}")
        ```
        """
        assert top_n >= 1

        params, gradsH = self.params_and_grads(inputs, targets, **kwargs)

        eigenvalues = []
        eigenvectors = []

        for _ in range(top_n):
            v = [torch.randn_like(p, device=self.device) for p in params]
            v = normalization(v)
            eigenvalue = None

            for _ in range(max_iter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                Hv = hvp(gradsH, params, v)
                tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue is None:
                    eigenvalue = tmp_eigenvalue
                elif abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) + 1e-6) < tol:
                    break
                else:
                    eigenvalue = tmp_eigenvalue

            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)

        return eigenvalues, eigenvectors

    def trace(
        self,
        inputs: Tensor,
        targets: Tensor,
        max_iter: int = 100,
        tol: float = 1e-3,
        **kwargs,
    ) -> List[float]:
        """Estimate trace using Hutchinson's method.

        ### Args
            - `inputs` (Tensor): Input tensor
            - `targets` (Tensor): Target tensor
            - `max_iter` (int): Maximum iterations. Defaults to 100.
            - `tol` (float): Convergence tolerance. Defaults to 1e-3.

        ### Returns
            - `List[float]`: List of trace estimates

        ### Example
        ```python
        >>> trace_estimates = hessian.trace(x, y)
        >>> print(f"Final trace estimate: {np.mean(trace_estimates)}")
        ```
        """
        params, gradsH = self.params_and_grads(inputs, targets, **kwargs)

        trace_vHv = []
        trace = 0.0

        for _ in range(max_iter):
            self.model.zero_grad()
            v = [torch.randint_like(p, high=2, device=self.device) for p in params]
            for v_i in v:
                v_i[v_i == 0] = -1

            Hv = hvp(gradsH, params, v)

            trace_vHv.append(group_product(Hv, v).cpu().item())

            if abs(np.mean(trace_vHv) - trace) / (abs(trace) + 1e-6) < tol:
                return trace_vHv
            trace = np.mean(trace_vHv)

        return trace_vHv

    def density(
        self,
        inputs: Tensor,
        targets: Tensor,
        iter: int = 100,
        n_v: int = 1,
        **kwargs,
    ) -> Tuple[List[List[float]], List[List[float]]]:
        """Estimate eigenvalue density using stochastic Lanczos algorithm.

        ### Args
            - `inputs` (Tensor): Input tensor
            - `targets` (Tensor): Target tensor
            - `iter` (int): Number of Lanczos iterations. Defaults to 100.
            - `n_v` (int): Number of random vectors to use. Defaults to 1.

        ### Returns
            - `Tuple[List[List[float]], List[List[float]]]`: Tuple of (eigenvalues, weights) for density estimation

        ### Example
        ```python
        >>> eigenvals, weights = hessian.density(x, y, iter=50, n_v=5)
        >>> plt.hist(eigenvals[0], weights=weights[0])  # Plot density
        ```
        """
        params, gradsH = self.params_and_grads(inputs, targets, **kwargs)

        eigen_list_full = []
        weight_list_full = []

        for _ in range(n_v):
            v = [torch.randint_like(p, high=2, device=self.device) for p in params]
            for v_i in v:
                v_i[v_i == 0] = -1
            v = normalization(v)

            v_list = [v]
            w_list = []
            alpha_list = []
            beta_list = []

            for i in range(iter):
                self.model.zero_grad()
                w_prime = [torch.zeros_like(p, device=self.device) for p in params]

                if i == 0:
                    w_prime = hvp(gradsH, params, v)
                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w = group_add(w_prime, v, alpha=-alpha)
                    w_list.append(w)
                else:
                    beta = torch.sqrt(group_product(w, w))
                    beta_list.append(beta.cpu().item())

                    if beta_list[-1] != 0.0:
                        v = orthnormal(w, v_list)
                    else:
                        w = [torch.randn_like(p, device=self.device) for p in params]
                        v = orthnormal(w, v_list)
                    v_list.append(v)

                    w_prime = hvp(gradsH, params, v)

                    alpha = group_product(w_prime, v)
                    alpha_list.append(alpha.cpu().item())
                    w_tmp = group_add(w_prime, v, alpha=-alpha)
                    w = group_add(w_tmp, v_list[-2], alpha=-beta)

            T = torch.zeros(iter, iter, device=self.device)
            for i in range(len(alpha_list)):
                T[i, i] = alpha_list[i]
                if i < len(alpha_list) - 1:
                    T[i + 1, i] = beta_list[i]
                    T[i, i + 1] = beta_list[i]

            eigenvalues, eigenvectors = torch.linalg.eig(T)
            eigen_list = eigenvalues.real
            weight_list = eigenvectors[0, :].pow(2)

            eigen_list_full.append(eigen_list.cpu().numpy().tolist())
            weight_list_full.append(weight_list.cpu().numpy().tolist())

        return eigen_list_full, weight_list_full
