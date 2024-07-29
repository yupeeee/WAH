import os

import torch
import tqdm

from ...typing import (
    DataLoader,
    Dataset,
    Devices,
    Dict,
    List,
    Literal,
    Module,
    Optional,
    Path,
    Tensor,
    Union,
)
from ...datasets.base import ClassificationDataset
from ...utils import dist
from ...utils.path import ls, rmdir
from ...utils.time import current_time
from ..load import load_state_dict

__all__ = [
    "LossTest",
    "LossTests",
]

temp_dir = "wahtmpdir@LossTest"


def set_loss_fn(
    loss: Literal["ce"],
    **kwargs,
) -> Module:
    """
    Sets the loss function for evaluation.

    ### Parameters
    - `loss (Literal["ce"])`: The type of loss function to use. Currently only "ce" (CrossEntropyLoss) is supported.
    - `**kwargs`: Additional keyword arguments for the loss function.

    ### Returns
    - `Module`: The loss function module.

    ### Raises
    - `ValueError`: If an unsupported loss type is provided.
    """
    if loss == "ce":
        return torch.nn.CrossEntropyLoss(reduction="none", **kwargs)
    else:
        raise ValueError(f"Unsupported loss: {loss}")


def run(
    rank: int,
    nprocs: int,
    model: Module,
    dataset: Dataset,
    loss_fn: Module,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = False,
    desc: Optional[str] = None,
) -> None:
    """
    Runs the loss computation on the given dataset.

    ### Parameters
    - `rank (int)`: The rank of the current process. Use -1 for CPU.
    - `nprocs (int)`: The total number of processes.
    - `model (Module)`: The PyTorch model to evaluate.
    - `dataset (Dataset)`: The dataset to evaluate the model on.
    - `loss_fn (Module)`: The loss function to use for evaluation.
    - `batch_size (int)`: The batch size for the DataLoader. Defaults to 1.
    - `num_workers (int)`: The number of workers for the DataLoader. Defaults to 0.
    - `verbose (bool)`: Whether to enable verbose output. Defaults to False.
    - `desc (Optional[str])`: Description for the progress bar. Defaults to None.

    ### Returns
    - `None`
    """
    # if rank == -1: CPU
    if rank == -1:
        rank = torch.device("cpu")

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        model = model.to(rank)

    # else: GPU
    else:
        dist.init_dist(rank, nprocs)

        dataloader = dist.load_dataloader(
            rank,
            nprocs,
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        model = dist.load_model(rank, model)

    # compute loss
    os.makedirs(temp_dir, exist_ok=True)

    for batch_idx, (indices, (data, targets)) in enumerate(
        tqdm.tqdm(
            dataloader,
            desc=f"LossTest" if desc is None else desc,
            disable=not verbose,
        )
    ):
        data: Tensor = data.to(rank)
        targets: Tensor = targets.to(rank)

        with torch.no_grad():
            outputs: Tensor = model(data)

        loss: Tensor = loss_fn(outputs, targets)

        torch.save(
            (indices, loss.to(torch.device("cpu"))),
            os.path.join(
                temp_dir,
                f"{batch_idx}-{current_time()}.pt",
            ),
        )

    # DDP cleanup
    if rank != torch.device("cpu"):
        dist.cleanup()


class LossTest:
    """
    A class for testing the loss of a PyTorch model on a given dataset.

    ### Attributes
    - `loss_fn (Module)`: The loss function module.
    - `batch_size (int)`: The batch size for the DataLoader. Defaults to 1.
    - `num_workers (int)`: The number of workers for the DataLoader. Defaults to 0.
    - `use_cuda (bool)`: Whether to use CUDA for evaluation. Defaults to False.
    - `devices (Optional[Devices])`: The devices to use for evaluation. Defaults to "auto".

    ### Methods
    - `__call__(model, dataset, verbose, desc) -> Dict[str, Union[int, float]]`: Evaluates the model on the given dataset and returns the computed loss.
    """

    def __init__(
        self,
        loss: Literal["ce"] = "ce",
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        Initialize the LossTest class.

        ### Parameters
        - `loss (Literal["ce"])`: The type of loss function to use. Currently only "ce" (CrossEntropyLoss) is supported.
        - `batch_size (int)`: The batch size for the DataLoader. Defaults to 1.
        - `num_workers (int)`: The number of workers for the DataLoader. Defaults to 0.
        - `use_cuda (bool)`: Whether to use CUDA for evaluation. Defaults to False.
        - `devices (Optional[Devices])`: The devices to use for evaluation. Defaults to "auto".
        """
        self.loss_fn = set_loss_fn(loss=loss)
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.use_cuda = use_cuda
        self.devices = "cpu"
        if self.use_cuda:
            self.devices = dist.parse_devices(devices)
            dist.init_os_env(self.devices)

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> Dict[str, Union[int, float]]:
        """
        Evaluates the model on the given dataset and returns the computed loss.

        ### Parameters
        - `model (Module)`: The PyTorch model to evaluate.
        - `dataset (Dataset)`: The dataset to evaluate the model on.
        - `verbose (bool)`: Whether to enable verbose output. Defaults to False.
        - `desc (Optional[str])`: Description for the progress bar. Defaults to None.

        ### Returns
        - `Dict[str, Union[int, float]]`: The computed loss and indices.
        """
        model.eval()
        if isinstance(dataset, ClassificationDataset):
            dataset.set_return_w_index()

        # GPU
        if self.use_cuda:
            nprocs = len(self.devices)

            dist.run_fn(
                fn=run,
                args=(
                    nprocs,
                    model,
                    dataset,
                    self.loss_fn,
                    self.batch_size,
                    self.num_workers,
                    verbose,
                    desc,
                ),
                nprocs=nprocs,
            )

        # CPU
        else:
            nprocs = 1

            run(
                rank=-1,
                nprocs=nprocs,
                model=model,
                dataset=dataset,
                loss_fn=self.loss_fn,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                verbose=verbose,
                desc=desc,
            )

        # loss
        indices: List[int] = []
        loss: List[float] = []

        file_names = ls(temp_dir)

        for file_name in file_names:
            i, l = torch.load(os.path.join(temp_dir, file_name))
            indices.append(i)
            loss.append(l)

        indices = torch.cat(indices, dim=0)
        loss = torch.cat(loss, dim=0)

        loss = loss[indices]
        indices, _ = indices.sort()

        rmdir(temp_dir)

        return {
            "idx": [int(i) for i in indices],
            "loss": [float(l) for l in loss],
        }


class LossTests:
    """
    A class for running loss tests across multiple checkpoints of a PyTorch model.

    ### Attributes
    - `test (LossTest)`: An instance of the LossTest class.

    ### Methods
    - `__call__(model, dataset, ckpt_dir, verbose, desc) -> float`: Evaluates the model across multiple checkpoints and returns the computed losses.
    """

    def __init__(
        self,
        loss: Literal["ce"] = "ce",
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        Initialize the LossTests class.

        ### Parameters
        - `loss (Literal["ce"])`: The type of loss function to use. Currently only "ce" (CrossEntropyLoss) is supported.
        - `batch_size (int)`: The batch size for the DataLoader. Defaults to 1.
        - `num_workers (int)`: The number of workers for the DataLoader. Defaults to 0.
        - `use_cuda (bool)`: Whether to use CUDA for evaluation. Defaults to False.
        - `devices (Optional[Devices])`: The devices to use for evaluation. Defaults to "auto".
        """
        self.test = LossTest(
            loss=loss,
            batch_size=batch_size,
            num_workers=num_workers,
            use_cuda=use_cuda,
            devices=devices,
        )

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
        ckpt_dir: Path,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> float:
        """
        Evaluates the model across multiple checkpoints and returns the computed losses.

        ### Parameters
        - `model (Module)`: The PyTorch model to evaluate.
        - `dataset (Dataset)`: The dataset to evaluate the model on.
        - `ckpt_dir (Path)`: The directory containing the model checkpoints.
        - `verbose (bool)`: Whether to enable verbose output. Defaults to False.
        - `desc (Optional[str])`: Description for the progress bar. Defaults to None.

        ### Returns
        - `float`: The computed losses across checkpoints.
        """
        ckpt_fnames = ls(ckpt_dir, fext=".ckpt", sort=True)
        ckpt_fnames = [fname for fname in ckpt_fnames if "epoch=" in fname]

        loss = []

        for epoch, ckpt_fname in enumerate(ckpt_fnames):
            epoch_id = f"epoch={epoch}"

            load_state_dict(
                model,
                os.path.join(ckpt_dir, ckpt_fname),
                map_location="cpu",
            )

            loss_per_epoch = self.test(
                model=model,
                dataset=dataset,
                verbose=verbose,
                desc=f"LossPlot ({epoch_id})" if desc is None else desc,
            )

            loss.append(loss_per_epoch["loss"])

        loss = torch.Tensor(loss)

        return loss
