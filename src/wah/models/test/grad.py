import os

import torch
import tqdm

from ...typing import (
    DataLoader,
    Dataset,
    Devices,
    List,
    Module,
    Optional,
    Tensor,
)
from ...datasets.base import ClassificationDataset
from ...utils import dist
from ...utils.path import ls, rmdir
from ...utils.time import current_time

__all__ = [
    "GradTest",
]

temp_dir = f"wahtmpdir@GradTest{current_time()}"


def run(
    rank: int,
    nprocs: int,
    model: Module,
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = False,
    desc: Optional[str] = None,
) -> None:
    """
    Runs the gradient computation on the given dataset.

    ### Parameters
    - `rank (int)`: The rank of the current process. Use -1 for CPU.
    - `nprocs (int)`: The total number of processes.
    - `model (Module)`: The PyTorch model to evaluate.
    - `dataset (Dataset)`: The dataset to evaluate the model on.
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

    # compute grad
    os.makedirs(temp_dir, exist_ok=True)

    for batch_idx, (indices, data) in enumerate(
        tqdm.tqdm(
            dataloader,
            desc=f"GradTest" if desc is None else desc,
            disable=not verbose,
        )
    ):
        data: Tensor = data.to(rank)
        data.requires_grad = True

        model.zero_grad()

        outputs: Tensor = model(data)
        outputs.sum().backward()

        grads = data.grad

        torch.save(
            (indices, grads.detach().to(torch.device("cpu"))),
            os.path.join(
                temp_dir,
                f"{batch_idx}-{current_time()}.pt",
            ),
        )

    # DDP cleanup
    if rank != torch.device("cpu"):
        dist.cleanup()


class GradTest:
    """
    A class for testing the gradients of a PyTorch model on a given dataset.

    ### Attributes
    - `batch_size (int)`: The batch size for the DataLoader. Defaults to 1.
    - `num_workers (int)`: The number of workers for the DataLoader. Defaults to 0.
    - `use_cuda (bool)`: Whether to use CUDA for evaluation. Defaults to False.
    - `devices (Optional[Devices])`: The devices to use for evaluation. Defaults to "auto".

    ### Methods
    - `__call__(model, dataset, verbose, desc) -> Tensor`: Evaluates the model on the given dataset and returns the computed gradients.
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        Initialize the GradTest class.

        ### Parameters
        - `batch_size (int)`: The batch size for the DataLoader. Defaults to 1.
        - `num_workers (int)`: The number of workers for the DataLoader. Defaults to 0.
        - `use_cuda (bool)`: Whether to use CUDA for evaluation. Defaults to False.
        - `devices (Optional[Devices])`: The devices to use for evaluation. Defaults to "auto".
        """
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
    ) -> Tensor:
        """
        Evaluates the model on the given dataset and returns the computed gradients.

        ### Parameters
        - `model (Module)`: The PyTorch model to evaluate.
        - `dataset (Dataset)`: The dataset to evaluate the model on.
        - `verbose (bool)`: Whether to enable verbose output. Defaults to False.
        - `desc (Optional[str])`: Description for the progress bar. Defaults to None.

        ### Returns
        - `Tensor`: The computed gradients.
        """
        model.eval()
        if isinstance(dataset, ClassificationDataset):
            dataset.set_return_data_only()
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
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                verbose=verbose,
                desc=desc,
            )

        # grad
        indices: List[int] = []
        grads: List[Tensor] = []

        file_names = ls(temp_dir)

        for file_name in file_names:
            i, g = torch.load(os.path.join(temp_dir, file_name))
            indices.append(i)
            grads.append(g)

        indices = torch.cat(indices, dim=0)
        grads = torch.cat(grads, dim=0)

        grads = grads[indices]
        indices, _ = indices.sort()

        rmdir(temp_dir)

        return grads
