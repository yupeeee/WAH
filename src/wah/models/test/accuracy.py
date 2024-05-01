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


def cpu_run(
    model: Module,
    dataset: Dataset,
    top_k: int = 1,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = False,
    desc: Optional[str] = None,
):
    device = torch.device("cpu")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model = model.to(device)

    acc = 0.0

    for data, targets in tqdm.tqdm(
        dataloader,
        desc=f"acc@{top_k}" if desc is None else desc,
        disable=not verbose,
    ):
        data: Tensor = data.to(device)
        targets: Tensor = targets.to(device)

        outputs: Tensor = model(data)

        _, preds = outputs.topk(k=top_k, dim=-1)

        for k in range(top_k):
            acc += float(preds[:, k].eq(targets).detach().sum())

    acc = acc / len(dataset)

    del dataloader

    return acc


def dist_run(
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
):
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

    del dataloader
    dist.cleanup()


class AccuracyTest:
    def __init__(
        self,
        top_k: int = 1,
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
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
        if self.use_cuda:
            nprocs = len(self.devices)
            res_queue = dist.Queue()

            dist.run_fn(
                fn=dist_run,
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

            corrects: int = 0

            for _ in range(nprocs):
                corrects += res_queue.get()

            acc: float = corrects / len(dataset)

        else:
            acc: float = cpu_run(
                model,
                dataset,
                self.top_k,
                self.batch_size,
                self.num_workers,
                verbose,
                desc,
            )

        return acc
