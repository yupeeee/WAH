import os
from typing import List, Tuple

import lightning as L
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from ..misc import path

__all__ = [
    "FID",
]


def _load_inception_v3() -> nn.Module:
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.eval()
    model.dropout = nn.Identity()
    model.fc = nn.Identity()
    return model


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super().__init__()

        self.model = model

        self.register_buffer("sum", torch.zeros(2048))
        self.register_buffer("sum_sq", torch.zeros(2048))
        self.register_buffer("num_examples", torch.tensor(0, dtype=torch.long))
        self.register_buffer("mean", torch.zeros(2048))
        self.register_buffer("std", torch.zeros(2048))

    def on_test_start(self) -> None:
        self.model.to(self.device)
        self.sum.zero_()
        self.sum_sq.zero_()
        self.num_examples.zero_()
        self.mean.zero_()
        self.std.zero_()

    def test_step(self, batch, batch_idx):
        # batch: [B, 3, 299, 299]
        activations = self.model(batch)  # [B, 2048]

        batch_sum = activations.sum(dim=0)
        batch_sum_sq = (activations**2).sum(dim=0)
        batch_count = activations.size(0)

        # Accumulate stats
        self.sum += batch_sum
        self.sum_sq += batch_sum_sq
        self.num_examples += batch_count

        return {}

    def on_test_end(self) -> None:
        # Gather sum, sum_sq, num_examples from all devices
        if self.trainer.world_size > 1:
            # For DDP, sync across all processes
            sum = self.sum.clone()
            sum_sq = self.sum_sq.clone()
            num_examples = self.num_examples.clone()
            torch.distributed.all_reduce(sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(sum_sq, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(
                num_examples, op=torch.distributed.ReduceOp.SUM
            )
        else:
            # Single-device
            sum = self.sum.clone()
            sum_sq = self.sum_sq.clone()
            num_examples = self.num_examples.clone()

        mean = sum / num_examples
        mean_sq = sum_sq / num_examples
        std = (mean_sq - mean**2).sqrt()
        self.mean = mean
        self.std = std

        return {}


def _load_accelerator_and_devices(
    devices: str = "auto",
) -> Tuple[str, List[int]]:
    if isinstance(devices, str) and "cuda" in devices:
        devices = devices.replace("cuda", "gpu")
    devices_cfg: List[str] = devices.split(":")
    accelerator = "auto"
    devices = "auto"
    if len(devices_cfg) == 1:
        accelerator = devices_cfg[0]
    else:
        accelerator = devices_cfg[0]
        devices = [int(d) for d in devices_cfg[1].split(",")]
    return accelerator, devices


def _load_runner(
    devices: str = "auto",
    use_half: bool = True,
) -> L.Trainer:
    # Load accelerator and devices
    accelerator, devices = _load_accelerator_and_devices(devices)
    # Set precision
    if use_half:
        torch.set_float32_matmul_precision("medium")
    # Load runner
    runner = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if use_half else "32-true",
        logger=False,
        callbacks=None,
        max_epochs=1,
        log_every_n_steps=None,
        enable_checkpointing=False,
        deterministic=False,
    )
    # Return runner
    return runner


class Runner:
    def __init__(
        self,
        batch_size: int = 128,
        num_workers: int = 4,
        devices: str = "auto",
        use_half: bool = True,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.runner: L.Trainer = _load_runner(devices, use_half)
        self.model: nn.Module = _load_inception_v3()

    def run(
        self,
        dataset: Dataset,
    ) -> None:
        dataloader: DataLoader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
        model: L.LightningModule = Wrapper(self.model)
        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )
        return model.mean.cpu(), model.std.cpu()


class ImageOnlyDataset(Dataset):
    def __init__(self, img_dir: os.PathLike) -> None:
        self.paths = path.ls(
            path=img_dir,
            absolute=True,
        )
        self.transform = models.Inception_V3_Weights.DEFAULT.transforms()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Image.Image:
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def _prepare_dataset(
    dataset,
) -> Dataset:
    if isinstance(dataset, str):
        if os.path.isdir(dataset):
            return ImageOnlyDataset(dataset)
        elif dataset in []:
            pass
        else:
            raise ValueError(f"Invalid dataset: {dataset}")
    elif isinstance(dataset, Dataset):
        return dataset
    else:
        raise ValueError(f"Invalid dataset type: {type(dataset)}")


def _compute_fid(
    mu1: torch.Tensor,
    sigma1: torch.Tensor,
    mu2: torch.Tensor,
    sigma2: torch.Tensor,
) -> float:
    # The Fréchet Inception Distance (FID)
    # between two multivariate Gaussians X_1 ~ N(mu_1, C_1) and X_2 ~ N(mu_2, C_2) is
    # d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    # Compute covariance matrices
    C1 = torch.diag(sigma1**2)
    C2 = torch.diag(sigma2**2)

    # Compute FID
    diff = mu1 - mu2
    sqrt_C1_C2 = torch.diag((C1 * C2).sqrt())
    fid = diff.dot(diff) + torch.trace(C1 + C2 - 2 * sqrt_C1_C2)

    return fid.item()


class FID:
    """Fréchet Inception Distance (FID) metric.

    ### Args
        - `batch_size` (int): Batch size.
        - `num_workers` (int): Number of workers.
        - `devices` (str): Devices to use.
        - `use_half` (bool): Whether to use half precision.

    ### Example
    ```python
    >>> fid = FID(batch_size=16, num_workers=4, devices="auto", use_half=True)
    >>> fid(dataset1, dataset2)
    ```
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        devices: str = "auto",
        use_half: bool = True,
    ) -> None:
        """
        - `batch_size` (int): Batch size.
        - `num_workers` (int): Number of workers.
        - `devices` (str): Devices to use.
        - `use_half` (bool): Whether to use half precision.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.devices = devices
        self.use_half = use_half

    def __call__(
        self,
        dataset1,
        dataset2,
    ) -> float:
        """
        Compute FID score between two datasets.

        ### Args
        - `dataset1` (Dataset): First dataset.
        - `dataset2` (Dataset): Second dataset.

        ### Returns
        - `fid` (float): FID score.
        """
        dataset1 = _prepare_dataset(dataset1)
        dataset2 = _prepare_dataset(dataset2)

        runner = Runner(
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            devices=self.devices,
            use_half=self.use_half,
        )
        mean1, std1 = runner.run(dataset1)
        mean2, std2 = runner.run(dataset2)

        fid = _compute_fid(mean1, std1, mean2, std2)

        return fid
