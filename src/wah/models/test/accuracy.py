import torch
import tqdm

from ...typing import (
    DataLoader,
    Dataset,
    Devices,
    Module,
    Optional,
    Tensor,
)
from ...utils import dist

__all__ = [
    "AccuracyTest",
]


def cpu_run(
    model: Module,
    dataset: Dataset,
    top_k: int = 1,
    batch_size: int = 1,
    verbose: bool = False,
):
    device = torch.device("cpu")

    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = model.to(device)

    acc = 0.0

    for data, targets in tqdm.tqdm(
        dataloader,
        desc=f"acc@{top_k}",
        disable=not verbose,
    ):
        data: Tensor = data.to(device)
        targets: Tensor = targets.to(device)

        with torch.no_grad():
            outputs: Tensor = model(data)

        _, preds = outputs.topk(k=top_k, dim=-1)

        for k in range(top_k):
            acc += float(preds[:, k].eq(targets).sum())

    acc = acc / len(dataloader.dataset)

    return acc


def dist_run(
    rank: int,
    nprocs: int,
    model: Module,
    dataset: Dataset,
    top_k: int = 1,
    batch_size: int = 1,
    verbose: bool = False,
):
    dist.init_dist(rank, nprocs)

    dataloader = dist.load_dataloader(rank, nprocs, dataset, batch_size=batch_size)
    model = dist.load_model(rank, model)

    acc = torch.zeros(size=(1,)).to(rank)

    for data, targets in tqdm.tqdm(
        dataloader,
        desc=f"acc@{top_k}",
        disable=not verbose,
    ):
        data: Tensor = data.to(rank)
        targets: Tensor = targets.to(rank)

        with torch.no_grad():
            outputs: Tensor = model(data)

        _, preds = outputs.topk(k=top_k, dim=-1)

        for k in range(top_k):
            acc += preds[:, k].eq(targets).sum()

    acc = acc / len(dataloader.dataset)

    return float(acc.to(torch.device("cpu")))


class AccuracyTest:
    def __init__(
        self,
        top_k: int = 1,
        batch_size: int = 1,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        self.top_k = top_k
        self.batch_size = batch_size

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
    ) -> float:
        if self.use_cuda:
            nprocs = len(self.devices)

            acc = dist.run_fn(
                fn=dist_run,
                args=(nprocs, model, dataset, self.top_k, self.batch_size, verbose),
                nprocs=nprocs,
            )
        
        else:
            acc = cpu_run(model, dataset, self.top_k, self.batch_size, verbose)
        
        return acc
