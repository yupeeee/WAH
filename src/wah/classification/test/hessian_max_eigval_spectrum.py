import lightning as L
import torch
import tqdm

from ... import utils
from ...classification.datasets import to_dataloader
from ...classification.train.utils import load_accelerator_and_devices
from ...typing import Config, Dataset, Devices, Dict, List, Module, Optional, Tensor

__all__ = [
    "HessianMaxEigValSpectrum",
]


def _hvp(
    model: Module,
    data: Tensor,
    targets: Tensor,
    criterion: Module,
    v: Tensor,
) -> Tensor:
    """
    Computes the Hessian-vector product (Hvp) for a given model, data, and vector `v`.

    ### Parameters
    - `model` (Module): The model whose Hessian is being computed.
    - `data` (Tensor): Input data batch.
    - `targets` (Tensor): Target labels for the data.
    - `criterion` (Module): The loss function used to compute the Hessian.
    - `v` (Tensor): The vector to multiply with the Hessian.

    ### Returns
    - `Tensor`: The result of Hessian-vector multiplication.
    """
    model.eval()
    model.zero_grad()

    # Forward pass
    outputs = model(data)
    loss = criterion(outputs, targets)

    # Compute gradient (Jacobian) of the loss with respect to the parameters
    params = list(model.parameters())
    J = torch.autograd.grad(loss, params, create_graph=True)

    # Perform element-wise product of Jacobian and vector
    Hv = torch.autograd.grad(J, params, grad_outputs=v)

    return Hv


def _flatten(
    v: Tensor,
) -> Tensor:
    """
    Flattens a list of tensors into a single 1D tensor.

    ### Parameters
    - `v` (Tensor): A list of tensors to flatten.

    ### Returns
    - `Tensor`: The flattened 1D tensor.
    """
    return torch.cat([x.view(-1) for x in v], dim=0)


def _compute_hessian_max_eigval(
    model: Module,
    data: Tensor,
    targets: Tensor,
    criterion: Tensor,
    max_iter: int,
    tol: float,
    verbose: bool = False,
) -> Tensor:
    """
    Computes the maximum eigenvalue of the Hessian matrix using power iteration.

    ### Parameters
    - `model` (Module): The model whose Hessian matrix is being computed.
    - `data` (Tensor): Input data batch.
    - `targets` (Tensor): Target labels for the data.
    - `criterion` (Tensor): The loss function used to compute the Hessian.
    - `max_iter` (int): Maximum number of iterations for the power method.
    - `tol` (float): Convergence tolerance.
    - `verbose` (bool, optional): Whether to show progress bars. Defaults to `False`.

    ### Returns
    - `Tensor`: The maximum eigenvalue of the Hessian matrix.
    """
    # Randomly initialize a vector for power iteration
    v = [torch.randn_like(p) for p in list(model.parameters())]
    max_eigval: Tensor
    shapes = [p.shape for p in v]

    for _ in tqdm.trange(
        max_iter,
        disable=not verbose,
    ):
        # Compute Hessian-vector product
        Hv = _hvp(model, data, targets, criterion, v)

        # Normalize the vector
        Hv_flat = _flatten(Hv)
        max_eigval = torch.norm(Hv_flat)
        Hv_flat = Hv_flat / max_eigval

        # Check for convergence
        if torch.allclose(_flatten(v), Hv_flat, atol=tol):
            break

        # update v
        i = 0
        v = []
        for shape in shapes:
            numel = torch.prod(
                torch.tensor(shape)
            ).item()  # Number of elements in the tensor
            hv = Hv_flat[i : i + numel].view(shape)  # Reshape to the original size
            v.append(hv)
            i += numel

    return max_eigval


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        config: Config,
        res_dict: Dict[str, List],
        criterion: str = "CrossEntropyLoss",
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> None:
        super().__init__()

        self.model = model
        self.config = config
        self.res_dict = res_dict
        self.criterion = getattr(torch.nn, criterion)(
            label_smoothing=self.config["label_smoothing"]
        )
        self.max_iter = max_iter
        self.tol = tol

        self.max_eigvals: List[Tensor] = []

    def test_step(self, batch: Tensor, batch_idx):
        data, targets = batch

        # enable gradients
        with torch.inference_mode(False):
            data = data.clone().detach()
            targets = targets.clone().detach()

            max_eigval = _compute_hessian_max_eigval(
                model=self.model,
                data=data,
                targets=targets,
                criterion=self.criterion,
                max_iter=self.max_iter,
                tol=self.tol,
            )
            self.max_eigvals.append(max_eigval.detach().cpu())

    def on_test_epoch_end(self) -> None:
        max_eigvals: List[Tensor] = self.all_gather(self.max_eigvals)
        if max_eigvals[0].dim() == 0:
            max_eigvals = [e.unsqueeze(0) for e in max_eigvals]
        max_eigvals = torch.cat(max_eigvals, dim=-1).flatten()

        self.res_dict["max_eigvals"] = [float(e) for e in max_eigvals]


class HessianMaxEigValSpectrum:
    """
    A class for computing the maximum eigenvalue of the Hessian matrix for a model on a given dataset using PyTorch Lightning.

    ### Attributes
    - `batch_size` (int): The batch size for the test.
    - `num_workers` (int): The number of workers for the DataLoader.
    - `mixup_alpha` (Optional[float]): Alpha value for Mixup data augmentation. Defaults to `0.0`.
    - `cutmix_alpha` (Optional[float]): Alpha value for CutMix data augmentation. Defaults to `0.0`.
    - `label_smoothing` (Optional[float]): The amount of label smoothing to apply to the loss. Defaults to `0.0`.
    - `seed` (Optional[int]): Random seed for deterministic behavior. Defaults to `None`.
    - `devices` (Optional[Devices]): The devices to run the test on. Defaults to `"auto"`.
    - `verbose` (Optional[bool]): Whether to show progress bar and model summary during testing. Defaults to `True`.

    ### Methods
    - `__call__(model: Module, dataset: Dataset, criterion: str = "CrossEntropyLoss", max_iter: int = 1000, tol: float = 1e-8) -> List[float]`:
      Runs the Hessian maximum eigenvalue computation and returns the maximum eigenvalues.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        mixup_alpha: Optional[float] = 0.0,
        cutmix_alpha: Optional[float] = 0.0,
        label_smoothing: Optional[float] = 0.0,
        seed: Optional[int] = None,
        devices: Optional[Devices] = "auto",
        verbose: Optional[bool] = True,
    ) -> None:
        """
        - `batch_size` (int): The batch size for the test.
        - `num_workers` (int): The number of workers for the DataLoader.
        - `mixup_alpha` (Optional[float], optional): Alpha value for Mixup data augmentation. Defaults to `0.0`.
        - `cutmix_alpha` (Optional[float], optional): Alpha value for CutMix data augmentation. Defaults to `0.0`.
        - `label_smoothing` (Optional[float], optional): The amount of label smoothing to apply to the loss. Defaults to `0.0`.
        - `seed` (Optional[int]): Random seed for deterministic behavior. Defaults to `None`.
        - `devices` (Optional[Devices]): The devices to run the test on. Defaults to `"auto"`.
        - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.devices = devices
        self.verbose = verbose

        utils.random.seed(self.seed)
        accelerator, devices = load_accelerator_and_devices(self.devices)

        self.runner = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=False,
            max_epochs=1,
            log_every_n_steps=None,
            deterministic="warn" if self.seed is not None else False,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
        )
        self.config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "mixup_alpha": self.mixup_alpha,
            "cutmix_alpha": self.cutmix_alpha,
            "label_smoothing": self.label_smoothing,
        }

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
        criterion: str = "CrossEntropyLoss",
        max_iter: int = 1000,
        tol: float = 1e-8,
    ) -> List[float]:
        """
        Runs the Hessian maximum eigenvalue computation for the given model and dataset.

        ### Parameters
        - `model` (Module): The model to be evaluated.
        - `dataset` (Dataset): The dataset to compute the Hessian matrix on.
        - `criterion` (str, optional): The loss function to use when computing the Hessian. Defaults to `"CrossEntropyLoss"`.
        - `max_iter` (int, optional): Maximum number of iterations for the power method. Defaults to `1000`.
        - `tol` (float, optional): Convergence tolerance. Defaults to `1e-8`.

        ### Returns
        - `List[float]`: The computed maximum eigenvalues of the Hessian matrix for the dataset.
        """
        res_dict = {}
        model = Wrapper(model, self.config, res_dict, criterion, max_iter, tol)
        dataloader = to_dataloader(
            dataset=dataset,
            train=False,
            **self.config,
        )

        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )

        return res_dict["max_eigvals"]
