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

__all__ = [
    "AccuracyTest",
]


def run(
    rank: int,
    nprocs: int,
    res_queue: ResQueue,
    model: Module,
    dataset: Dataset,
    top_k: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = False,
    desc: Optional[str] = None,
) -> None:
    """
    Runs the evaluation of the model on the given dataset and computes top-k accuracy.

    ### Parameters
    - `rank` (int):
      The rank of the current process.
      Use -1 for CPU.
    - `nprocs` (int):
      The total number of processes.
    - `res_queue` (Queue):
      A multiprocessing queue to store the results.
    - `model` (Module):
      The PyTorch model to evaluate.
    - `dataset` (Dataset):
      The dataset to evaluate the model on.
    - `top_k` (int):
      The top-k accuracy to compute.
      Defaults to 1.
    - `batch_size` (int):
      The batch size for the DataLoader.
      Defaults to 1.
    - `num_workers` (int):
      The number of workers for the DataLoader.
      Defaults to 0.
    - `verbose` (bool):
      Whether to enable verbose output.
      Defaults to False.
    - `desc` (str, optional):
      Description for the progress bar.
      Defaults to None.

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

    # compute accuracy
    corrects: int = 0

    for data, targets in tqdm.tqdm(
        dataloader,
        desc=f"acc@{top_k}" if desc is None else desc,
        disable=not verbose,
    ):
        data: Tensor = data.to(rank)
        targets: Tensor = targets.to(rank)

        outputs: Tensor = model(data)

        _, preds = outputs.topk(k=top_k, dim=-1)

        for k in range(top_k):
            corrects += int(preds[:, k].eq(targets).detach().sum())

    res_queue.put(corrects)

    # DDP cleanup
    if rank != torch.device("cpu"):
        dist.cleanup()


class AccuracyTest:
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
        top_k: int = 1,
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
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
        """
        self.top_k = top_k
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
        """
        Evaluates the model on the given dataset and returns the accuracy.

        ### Parameters
        - `model` (Module):
          The PyTorch model to evaluate.
        - `dataset` (Dataset):
          The dataset to evaluate the model on.
        - `verbose` (bool):
          Whether to enable verbose output. Defaults to False.
        - `desc` (str, optional):
          Description for the progress bar. Defaults to None.

        ### Returns
        - `float`:
          The computed accuracy.
        """
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
                nprocs=nprocs,
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
        corrects: int = 0
        for _ in range(nprocs):
            corrects += res_queue.get()

        acc: float = corrects / len(dataset)

        return acc
