import torch
import tqdm

from ...typing import (
    DataLoader,
    Dataset,
    Devices,
    Module,
    Optional,
    ResQueue,
    Tensor,
)
from ...utils import dist


def run(
    rank: int,
    nprocs: int,
    res_queue: ResQueue,
    model: Module,
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = False,
    desc: Optional[str] = None,
) -> None:
    if hasattr(dataset, "targets"):
        dataset.set_return_data_only()

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

    # compute
    num_data = 0
    cumulative_mean = None
    cumulative_m2 = None

    for data in tqdm.tqdm(
        dataloader,
        desc=f"linearity@input_grad" if desc is None else desc,
        disable=not verbose,
    ):
        data: Tensor = data.to(rank)
        data.requires_grad()

        # forward pass
        outputs: Tensor = model(data)

        # backward pass
        outputs.backward(torch.ones_like(outputs))

        # gradients
        grads = data.grad.detach()

        # batch size, mean, var
        batch_size = grads.size(0)
        grads = grads.reshape(batch_size, -1)

        batch_mean = torch.mean(grads, dim=0)
        batch_var = torch.var(grads, dim=0, unbiased=False)

        # save results
        num_data += batch_size

        delta = batch_mean - cumulative_mean
        cumulative_mean += delta * (batch_size / num_data)

        cumulative_m2 += (
            batch_size * batch_var
            + (delta**2) * batch_size * (num_data - batch_size) / num_data
        )

    res_queue.put((num_data, cumulative_mean, cumulative_m2))

    # DDP cleanup
    if rank != torch.device("cpu"):
        dist.cleanup()


class GradLinearityTest:
    """
    A class for testing the top-k accuracy of a PyTorch model on a given dataset.

    ### Parameters
    - `top_k` (int):
      The top-k accuracy to compute.
      Defaults to 1.
    - `batch_size` (int):
      The batch size for the DataLoader.
      Defaults to 1.
    - `num_workers` (int):
      The number of workers for the DataLoader.
      Defaults to 0.
    - `use_cuda` (bool):
      Whether to use CUDA for evaluation.
      Defaults to False.
    - `devices` (Union[str, List[int]], optional):
      The devices to use for evaluation.
      Defaults to "auto".

    ### Methods
    - `__call__`:
      Evaluates the model on the given dataset and returns the accuracy.
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        - `batch_size` (int):
        The batch size for the DataLoader.
        Defaults to 1.
        - `num_workers` (int):
        The number of workers for the DataLoader.
        Defaults to 0.
        - `use_cuda` (bool):
        Whether to use CUDA for evaluation.
        Defaults to False.
        - `devices` (Union[str, List[int]], optional):
        The devices to use for evaluation.
        Defaults to "auto".
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
    ) -> float:
        model.eval()
        res_queue = dist.Queue()

        # GPU
        if self.use_cuda:
            nprocs = len(self.devices)

            dist.run_fn(
                fn=run,
                args=(
                    nprocs,
                    res_queue,
                    model,
                    dataset,
                    self.top_k,
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
                nprocs=1,
                res_queue=res_queue,
                model=model,
                dataset=dataset,
                top_k=self.top_k,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                verbose=verbose,
                desc=desc,
            )

        # acc
        N = 0
        mean = None
        m2 = None

        for _ in range(nprocs):
            num_data, cumulative_mean, cumulative_m2 = res_queue.get()

            N += num_data
            mean += cumulative_mean
            m2 += cumulative_m2

        overall_mean = self.cumulative_mean
        overall_var = self.cumulative_m2 / self.total_samples
        overall_std = torch.sqrt(overall_var)

        return overall_mean, overall_std

        corrects: int = 0
        

        acc: float = corrects / len(dataset)

        return acc
