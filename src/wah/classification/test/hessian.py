import lightning as L
import torch
from torch.func import hessian

from ... import utils
from ...classification.datasets import to_dataloader
from ...classification.train.utils import load_accelerator_and_devices
from ...typing import Dataset, Devices, Dict, List, Literal, Module, Optional, Tensor

__all__ = [
    "HessianSigVals",
]


def compute_loss_wrt_params(params, model, data, targets, criterion):
    torch.nn.utils.vector_to_parameters(params, model.parameters())
    outputs = model(data)
    return criterion(outputs, targets)


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        res_dict: Dict[str, List],
        criterion: str = "CrossEntropyLoss",
        wrt: Literal[
            "param",
            "input",
        ] = "param",
        use_half: Optional[bool] = False,
    ) -> None:
        assert wrt in [
            "param",
            "input",
        ], f"wrt must be one of 'param' or 'input', got {wrt}"

        super().__init__()

        self.model = model
        self.res_dict = res_dict
        self.criterion = getattr(torch.nn, criterion)()
        self.wrt = wrt
        self.use_half = use_half

        if self.use_half:
            self.model = model.half()

        self.num_data = []
        self.sigval = []

    def test_step(self, batch: Tensor, batch_idx):
        data, targets = batch

        if self.use_half:
            data = data.half()
            targets = targets.half()

        batch_size = len(data)
        input_dim = batch[0].numel()

        # enable gradients
        with torch.inference_mode(False):
            if self.wrt == "param":
                H: Tensor = hessian(
                    lambda p: self.criterion(self.model(data), targets)
                )(torch.nn.utils.parameters_to_vector(self.model.parameters()))
            elif self.wrt == "input":
                data = data.clone().detach().requires_grad_(True)
                H: Tensor = hessian(lambda x: self.criterion(self.model(x), targets))(
                    data
                )
            else:
                raise

        assert H.requires_grad
        H = H.detach().reshape(batch_size, input_dim, input_dim)

        sigvals: Tensor = torch.linalg.svdvals(H)

        self.num_data.append(batch_size)
        self.sigval.append(sigvals.cpu())

    def on_test_epoch_end(self) -> None:
        num_data: List[Tensor] = self.all_gather(self.num_data)
        sigval: List[Tensor] = self.all_gather(self.sigval)

        num_data = torch.cat(num_data, dim=0).flatten().sum()
        sigval: Tensor = torch.cat(sigval, dim=1).permute(1, 0, 2).reshape(num_data, -1)

        self.res_dict["sigval"] = sigval


class HessianSigVals:
    """
    A class for computing the singular values of the Hessian matrix for a model on a given dataset using PyTorch Lightning.

    ### Attributes
    - `batch_size` (int): The batch size for the test.
    - `num_workers` (int): The number of workers for the DataLoader.
    - `seed` (Optional[int]): Random seed for deterministic behavior. Defaults to `None`.
    - `devices` (Optional[Devices]): The devices to run the test on. Defaults to `"auto"`.
    - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.

    ### Methods
    - `__call__(model: Module, dataset: Dataset, criterion: str = "CrossEntropyLoss", wrt: Literal['param', 'input'] = 'param') -> Tensor`:
    Runs the Hessian singular value computation and returns the singular values.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        seed: Optional[int] = None,
        devices: Optional[Devices] = "auto",
        verbose: Optional[bool] = True,
    ) -> None:
        """
        - `batch_size` (int): The batch size for the test.
        - `num_workers` (int): The number of workers for the DataLoader.
        - `seed` (Optional[int]): Random seed for deterministic behavior. Defaults to `None`.
        - `devices` (Optional[Devices]): The devices to run the test on. Defaults to `"auto"`.
        - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        }

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
        criterion: str = "CrossEntropyLoss",
        wrt: Literal[
            "param",
            "input",
        ] = "param",
        use_half: Optional[bool] = False,
    ) -> Tensor:
        """
        Runs the Hessian singular value computation for the given model and dataset.

        ### Parameters
        - `model` (Module): The model to be evaluated.
        - `dataset` (Dataset): The dataset to compute the Hessian matrix on.
        - `criterion` (str, optional): The loss function to use when computing the Hessian. Defaults to `"CrossEntropyLoss"`.
        - `wrt` (Literal["param", "input"], optional): Specifies whether to compute the Hessian with respect to model parameters (`"param"`) or input data (`"input"`). Defaults to `"param"`.
        - `use_half` (Optional[bool]): Whether to use half-precision (float16) for the model and input data. Defaults to `False`.

        ### Returns
        - `Tensor`: The computed singular values of the Hessian matrix for the dataset.
        """
        res_dict = {}
        model = Wrapper(model, res_dict, criterion, wrt, use_half)
        dataloader = to_dataloader(
            dataset=dataset,
            train=False,
            **self.config,
        )

        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )

        return res_dict["sigval"].to(torch.device("cpu"))
