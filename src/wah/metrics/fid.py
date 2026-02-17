import os
from typing import Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import models

from ..misc import path as _path

__all__ = [
    "FID",
]


def _load_inception_v3() -> nn.Module:
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.eval()
    model.dropout = nn.Identity()
    model.fc = nn.Identity()
    return model


class _StatsAccumulator:
    def __init__(self, device: torch.device):
        self.sum = torch.zeros(2048, device=device)
        self.sum_sq = torch.zeros(2048, device=device)
        self.num_examples = 0

    def update(self, batch: torch.Tensor):
        self.sum += batch.sum(dim=0)
        self.sum_sq += (batch**2).sum(dim=0)
        self.num_examples += batch.shape[0]

    def finalize(self):
        mean = self.sum / self.num_examples
        mean_sq = self.sum_sq / self.num_examples
        std = (mean_sq - mean**2).sqrt()
        return mean.cpu(), std.cpu()


class ImageOnlyDataset(Dataset):
    def __init__(self, img_dir: os.PathLike) -> None:
        self.paths = _path.ls(
            path=img_dir,
            absolute=True,
        )
        self.transform = models.Inception_V3_Weights.DEFAULT.transforms()

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> torch.Tensor:
        img = Image.open(self.paths[index]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def _prepare_dataset(
    dataset,
) -> Dataset:
    if isinstance(dataset, Dataset):
        return dataset
    if isinstance(dataset, str):
        img_dir = _path.clean(dataset)
        if os.path.isdir(img_dir):
            return ImageOnlyDataset(img_dir)
        else:
            raise ValueError(f"dataset must be a path to a directory, got {dataset}")
    raise ValueError(
        f"Invalid dataset type: {type(dataset)} (expected str path or PyTorch Dataset)"
    )


def _run_single_dataset(
    dataset: Dataset,
    batch_size: int = 128,
    num_workers: int = 4,
    device: torch.device = torch.device("cpu"),
    use_half: bool = True,
) -> Tuple[Tensor, Tensor]:
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    model = _load_inception_v3().to(device)
    if use_half:
        model = model.half()
    model.eval()

    stats = _StatsAccumulator(device)

    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                batch = batch[0]
            batch = batch.to(device)
            if use_half:
                batch = batch.half()
            activations = model(batch)
            stats.update(activations)

    return stats.finalize()


def _compute_fid(
    mu1: Tensor,
    sigma1: Tensor,
    mu2: Tensor,
    sigma2: Tensor,
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
        - `use_half` (bool): Whether to use half precision.

    ### Example
    ```python
    >>> fid = FID(batch_size=16, num_workers=4, device="cuda", use_half=True)
    >>> fid(dataset1, dataset2)
    ```
    """

    def __init__(
        self,
        batch_size: int = 16,
        num_workers: int = 4,
        device: torch.device = torch.device("cpu"),
        use_half: bool = True,
    ) -> None:
        """
        - `batch_size` (int): Batch size.
        - `num_workers` (int): Number of workers.
        - `device` (torch.device): Device to use.
        - `use_half` (bool): Whether to use half precision.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
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

        # Prepare datasets
        dataset1 = _prepare_dataset(dataset1)
        dataset2 = _prepare_dataset(dataset2)

        mean1, std1 = _run_single_dataset(
            dataset1,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            use_half=self.use_half,
        )
        mean2, std2 = _run_single_dataset(
            dataset2,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            device=self.device,
            use_half=self.use_half,
        )

        fid = _compute_fid(mean1, std1, mean2, std2)
        return fid
