import os

import torch
import tqdm

from ....datasets.base import ClassificationDataset
from ....typing import (
    Dataset,
    DataLoader,
    Device,
    Devices,
    Dict,
    List,
    Module,
    Optional,
    Tensor,
)
from ....utils import dist
from ....utils.path import ls, rmdir
from ....utils.random import seed_everything
from ....utils.time import current_time
from ...feature_extractor import (
    flatten_batch,
    FeatureExtractor,
)

__all__ = [
    "PCALinearityTest",
]

temp_dir = f"wahtmpdir@PCALinearityTest{current_time()}"
default_pca_dim = 10


def compute_first_eigval_ratios(
    feature_extractor: Module,
    data: Tensor,
    device: Device,
    pca_dim: int = default_pca_dim,
) -> Dict[str, List[float]]:
    feature_extractor.to(device)
    data = data.to(device)

    # compute features
    with torch.no_grad():
        features: Dict[str, Tensor] = feature_extractor(data)

    # pca
    first_eigval_ratios = {}

    for layer, feature in features.items():
        feature = flatten_batch(feature)

        _, eigvals, _ = torch.pca_lowrank(feature, q=pca_dim)
        eigvals, _ = torch.sort(eigvals, descending=True)

        first_eigval_ratio = (eigvals / eigvals.sum())[0].item()

        first_eigval_ratios[layer] = [first_eigval_ratio]

    return first_eigval_ratios


def run(
    rank: int,
    nprocs: int,
    feature_extractor: Module,
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    pca_dim: int = default_pca_dim,
    verbose: bool = False,
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
        feature_extractor = feature_extractor.to(rank)

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
        feature_extractor = dist.load_model(rank, feature_extractor)

    # compute pca
    os.makedirs(temp_dir, exist_ok=True)

    for batch_idx, data in enumerate(
        tqdm.tqdm(
            dataloader,
            desc=f"PCALinearityTest",
            disable=not verbose,
        )
    ):
        first_eigval_ratios = compute_first_eigval_ratios(
            feature_extractor=feature_extractor,
            data=data,
            device=rank,
            pca_dim=pca_dim,
        )

        torch.save(
            first_eigval_ratios,
            os.path.join(
                temp_dir,
                f"{batch_idx}-{current_time()}.pt",
            ),
        )

    # DDP cleanup
    if rank != torch.device("cpu"):
        dist.cleanup()


class PCALinearityTest:
    def __init__(
        self,
        batch_size: int = -1,
        num_workers: int = 0,
        pca_dim: int = default_pca_dim,
        seed: Optional[int] = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pca_dim = pca_dim
        self.seed = seed

        seed_everything(seed)

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.devices = dist.parse_devices(devices)
            dist.init_os_env(self.devices)

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
        verbose: bool = False,
    ) -> Dict[float, List[float]]:
        if self.batch_size < 0:
            self.batch_size = len(dataset)

        feature_extractor = FeatureExtractor(model.eval())
        if isinstance(dataset, ClassificationDataset):
            dataset.set_return_data_only()

        # GPU
        if self.use_cuda:
            nprocs = len(self.devices)

            dist.run_fn(
                fn=run,
                args=(
                    nprocs,
                    feature_extractor,
                    dataset,
                    self.batch_size,
                    self.num_workers,
                    self.pca_dim,
                    verbose,
                ),
                nprocs=nprocs,
            )

        # CPU
        else:
            nprocs = 1

            run(
                rank=-1,
                nprocs=nprocs,
                feature_extractor=feature_extractor,
                dataset=dataset,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pca_dim=self.pca_dim,
                verbose=verbose,
            )

        # first eigenvalue ratios
        first_eigval_ratios = dict()

        first_eigval_ratios_fnames = ls(
            path=temp_dir,
            fext=".pt",
            sort=True,
        )

        for fname in first_eigval_ratios_fnames:
            first_eigval_ratios_batch: Dict[str, List[float]] = torch.load(
                os.path.join(temp_dir, fname)
            )

            for layer, ratios in first_eigval_ratios_batch.items():
                if layer in first_eigval_ratios.keys():
                    first_eigval_ratios[layer] += ratios
                else:
                    first_eigval_ratios[layer] = ratios

        rmdir(temp_dir)

        dataset.unset_return_data_only()

        return first_eigval_ratios
