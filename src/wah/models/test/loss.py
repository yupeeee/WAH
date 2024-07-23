import os

import torch
import tqdm

from ...typing import (
    DataLoader,
    Dataset,
    Devices,
    List,
    Literal,
    Module,
    Optional,
    Path,
    Tensor,
)
from ...datasets.base import ClassificationDataset
from ...utils import dist
from ...utils.path import ls, rmdir
from ...utils.time import current_time
from ..load import load_state_dict

__all__ = [
    "LossTest",
    "LossPlot",
]

temp_dir = "wahtmpdir@LossTest"


def set_loss_fn(
    loss: Literal["ce",],
    **kwargs,
) -> Module:
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
    def __init__(
        self,
        loss: Literal["ce",] = "ce",
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
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
    ) -> float:
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


class LossPlot:
    def __init__(
        self,
        loss: Literal["ce",] = "ce",
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
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
