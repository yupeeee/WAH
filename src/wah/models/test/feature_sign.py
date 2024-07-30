import os

import torch
import tqdm

from ...datasets.base import ClassificationDataset
from ...typing import (
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
from ...utils import dist
from ...utils.path import ls, rmdir
from ...utils.random import seed_everything
from ...utils.time import current_time
from ..feature_extractor import (
    flatten_batch,
    FeatureExtractor,
)

__all__ = [
    "FeatureSignTest",
]

temp_dir = "wahtmpdir@FeatureSignTest"


def compute_negative_ratios(
    feature_extractor: Module,
    data: Tensor,
    device: Device,
) -> Dict[str, List[float]]:
    """
    Computes the negative ratios of features extracted by the feature extractor.

    ### Parameters
    - `feature_extractor (Module)`: The feature extractor model.
    - `data (Tensor)`: The input data tensor.
    - `device (Device)`: The device to perform computation on.

    ### Returns
    - `Dict[str, List[float]]`: A dictionary with layer names as keys and lists of negative ratios as values.
    """
    feature_extractor.to(device)
    data = data.to(device)

    # compute features
    with torch.no_grad():
        features: Dict[str, Tensor] = feature_extractor(data)

    # compute negative ratios of features
    negative_ratios = {}

    for layer, feature in features.items():
        feature = flatten_batch(feature)
        f_sign = torch.sum((feature < 0).int(), dim=-1) / feature.size(-1)

        negative_ratios[layer] = [float(f) for f in f_sign]

    return negative_ratios


def run(
    rank: int,
    nprocs: int,
    feature_extractor: Module,
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = False,
) -> None:
    """
    Runs the feature extraction and computes negative ratios across dataset.

    ### Parameters
    - `rank (int)`: The rank of the current process. Use -1 for CPU.
    - `nprocs (int)`: The total number of processes.
    - `feature_extractor (Module)`: The feature extractor model.
    - `dataset (Dataset)`: The dataset to evaluate.
    - `batch_size (int)`: The batch size for the DataLoader. Defaults to 1.
    - `num_workers (int)`: The number of workers for the DataLoader. Defaults to 0.
    - `verbose (bool)`: Whether to enable verbose output. Defaults to False.

    ### Returns
    - `None`
    """
    # if rank == -1: CPU
    if rank == -1:
        rank = "cpu"

        dataloader = DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )
        feature_extractor = feature_extractor.to(torch.device(rank))

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

    # compute negative ratios
    os.makedirs(temp_dir, exist_ok=True)

    for batch_idx, data in enumerate(
        tqdm.tqdm(
            dataloader,
            desc=f"FeatureSignTest",
            disable=not verbose,
        )
    ):
        negative_ratios = compute_negative_ratios(
            feature_extractor=feature_extractor,
            data=data,
            device=rank,
        )

        torch.save(
            negative_ratios,
            os.path.join(
                temp_dir,
                f"{batch_idx}-{current_time()}.pt",
            ),
        )

    # DDP cleanup
    if rank != torch.device("cpu"):
        dist.cleanup()


class FeatureSignTest:
    """
    A class for testing the feature sign of a PyTorch model on a given dataset.

    ### Attributes
    - `batch_size (int)`: The batch size for the DataLoader. Defaults to 1.
    - `num_workers (int)`: The number of workers for the DataLoader. Defaults to 0.
    - `seed (Optional[int])`: The random seed for reproducibility. Defaults to 0.
    - `use_cuda (bool)`: Whether to use CUDA for evaluation. Defaults to False.
    - `devices (Optional[Devices])`: The devices to use for evaluation. Defaults to "auto".

    ### Methods
    - `__call__(model, dataset, verbose) -> Dict[float, List[float]]`: Evaluates the model on the given dataset and returns the negative ratios.
    """

    def __init__(
        self,
        batch_size: int = 1,
        num_workers: int = 0,
        seed: Optional[int] = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        Initialize the FeatureSignTest class.

        ### Parameters
        - `batch_size (int)`: The batch size for the DataLoader. Defaults to 1.
        - `num_workers (int)`: The number of workers for the DataLoader. Defaults to 0.
        - `seed (Optional[int])`: The random seed for reproducibility. Defaults to 0.
        - `use_cuda (bool)`: Whether to use CUDA for evaluation. Defaults to False.
        - `devices (Optional[Devices])`: The devices to use for evaluation. Defaults to "auto".
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        """
        Evaluates the model on the given dataset and returns the negative ratios.

        ### Parameters
        - `model (Module)`: The PyTorch model to evaluate.
        - `dataset (Dataset)`: The dataset to evaluate the model on.
        - `verbose (bool)`: Whether to enable verbose output. Defaults to False.

        ### Returns
        - `Dict[float, List[float]]`: The computed negative ratios.
        """
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
                verbose=verbose,
            )

        # negative ratios
        negative_ratios = dict()

        negative_ratios_fnames = ls(
            path=temp_dir,
            fext=".pt",
            sort=True,
        )

        for fname in negative_ratios_fnames:
            negative_ratios_batch: Dict[str, List[float]] = torch.load(
                os.path.join(temp_dir, fname)
            )

            for layer, ratios in negative_ratios_batch.items():
                if layer in negative_ratios.keys():
                    negative_ratios[layer] += ratios
                else:
                    negative_ratios[layer] = ratios

        rmdir(temp_dir)

        dataset.unset_return_data_only()

        return negative_ratios
