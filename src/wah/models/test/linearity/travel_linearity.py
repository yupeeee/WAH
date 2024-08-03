import os

import torch
import tqdm

from ....typing import (
    Dataset,
    DataLoader,
    Device,
    Devices,
    Dict,
    List,
    Literal,
    Module,
    Optional,
    Path,
    Tensor,
    Tuple,
)
from ....utils import dist
from ....utils.dictionary import save_in_csv
from ....utils.path import ls, rmdir
from ....utils.random import seed_everything
from ....utils.time import current_time
from ...load import load_state_dict
from ..travel import DirectionGenerator

__all__ = [
    "TravelLinearityTest",
]

temp_dir = f"wahtmpdir@TravelLinearityTest"


def move_x(
    x: Tensor,
    d: Tensor,
    eps: float,
    bound: bool = True,
) -> Tensor:
    """
    Moves the input data `x` along the direction `d` by `eps`.

    ### Parameters
    - `x (Tensor)`: The input data.
    - `d (Tensor)`: The direction tensor.
    - `eps (float)`: The distance to move along the direction.
    - `bound (bool, optional)`: Whether to clamp the output to [0, 1]. Defaults to True.

    ### Returns
    - `Tensor`: The moved data.
    """
    _x = x + d * eps

    if bound:
        _x = _x.clamp(0, 1)

    return _x


def load_preprocess_and_feature_extractor(
    model: Module,
) -> Tuple[Module, Module]:
    """
    Loads the preprocessing and feature extractor modules from the given model.

    ### Parameters
    - `model (Module)`: The model from which to load the preprocessing and feature extractor modules.

    ### Returns
    - `Tuple[Module, Module]`: A tuple containing the preprocessing module and the feature extractor module.
    """
    # model is timm model
    try:
        if hasattr(model, "preprocess"):
            preprocess = model.preprocess
            feature_extractor = model.model.forward_features
        else:
            preprocess = torch.nn.Identity()
            feature_extractor = model.forward_features

    # TBD
    except AttributeError:
        pass

    return preprocess, feature_extractor


def compute_cossim(
    model: Module,
    data: Tensor,
    directions: Tensor,
    eps: float,
    delta: float,
    device: Device,
    bound: bool = True,
) -> List[float]:
    """
    Computes cosine similarities between feature vectors of moved data.

    ### Parameters
    - `model (Module)`: The model to extract features.
    - `data (Tensor)`: The input data.
    - `directions (Tensor)`: The direction tensor.
    - `eps (float)`: The distance to move along the direction.
    - `delta (float)`: The small perturbation for calculating movement vectors.
    - `device (Device)`: The device to run the model on.
    - `bound (bool, optional)`: Whether to clamp the moved data to [0, 1]. Defaults to True.

    ### Returns
    - `List[float]`: The cosine similarities.
    """
    batch_size = len(data)
    nzd = 1.0e-7  # for nonzero division

    model.to(device)
    data = data.to(device)
    directions = directions.to(device)

    # move data along given directions
    data_eps = move_x(data, directions, eps, bound)
    data_eps_l = move_x(data, directions, eps - delta, bound)
    data_eps_r = move_x(data, directions, eps + delta, bound)

    # compute features
    with torch.no_grad():
        # DDP check & load preprocess/feature_extractor
        if device != "cpu":
            preprocess, feature_extractor = load_preprocess_and_feature_extractor(
                model.module
            )
        else:
            preprocess, feature_extractor = load_preprocess_and_feature_extractor(model)

        out_eps: Tensor = feature_extractor(preprocess(data_eps))
        out_eps_l: Tensor = feature_extractor(preprocess(data_eps_l))
        out_eps_r: Tensor = feature_extractor(preprocess(data_eps_r))

    # compute/normalize movement vectors
    vl = (out_eps_l - out_eps).reshape(batch_size, -1)
    vr = (out_eps_r - out_eps).reshape(batch_size, -1)

    vl = vl / torch.norm(vl, p=2, dim=-1, keepdim=True) + nzd
    vr = vr / torch.norm(vr, p=2, dim=-1, keepdim=True) + nzd

    # compute cosine similiarities
    cossims = torch.sum(-vl * vr, dim=-1)

    return [float(cossim) for cossim in cossims.to(torch.device("cpu"))]


def run(
    rank: int,
    nprocs: int,
    model: Module,
    dataset: Dataset,
    epsilons: List[float],
    delta: float,
    batch_size: int = 1,
    num_workers: int = 0,
    method: str = "fgsm",
    seed: Optional[int] = 0,
    bound: bool = True,
    verbose: bool = False,
) -> None:
    """
    Runs the linearity test for a given model and dataset.

    ### Parameters
    - `rank (int)`: The rank of the current process.
    - `nprocs (int)`: The total number of processes.
    - `model (Module)`: The model to test.
    - `dataset (Dataset)`: The dataset to test on.
    - `epsilons (List[float])`: The list of epsilon values to test.
    - `delta (float)`: The small perturbation for calculating movement vectors.
    - `batch_size (int, optional)`: The batch size for the DataLoader. Defaults to 1.
    - `num_workers (int, optional)`: The number of workers for the DataLoader. Defaults to 0.
    - `method (str, optional)`: The method to use for generating travel directions. Defaults to "fgsm".
    - `bound (bool, optional)`: Whether to clamp the moved data to [0, 1]. Defaults to True.
    - `seed (int, optional)`: The seed for random number generation. Defaults to 0.
    - `verbose (bool, optional)`: Whether to print progress. Defaults to False.

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
        model = model.to(torch.device(rank))

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

    # compute linearity
    direction_generator = DirectionGenerator(
        model=model,
        method=method,
        device=rank,
    )

    if verbose:
        print(f"Linearity@{method}_travel test of {len(dataloader.dataset)} data")

    os.makedirs(temp_dir, exist_ok=True)

    for eps_idx, eps in enumerate(epsilons):
        if method == "signed_rand":
            seed_everything(seed)

        batch_idx = 0

        for data, targets in tqdm.tqdm(
            dataloader,
            desc=f"[{eps_idx + 1}/{len(epsilons)}] eps={eps}",
            disable=not verbose,
        ):
            if method == "same_signed_rand":
                seed_everything(seed)

            directions = direction_generator(data, targets)
            cossim = compute_cossim(
                model=model,
                data=data,
                directions=directions,
                eps=eps,
                delta=delta,
                device=rank,
                bound=bound,
            )

            torch.save(
                cossim,
                os.path.join(
                    temp_dir,
                    f"{method}-{eps_idx}-{batch_idx}-{current_time()}.pt",
                ),
            )
            batch_idx += 1

    # DDP cleanup
    if rank != "cpu":
        dist.cleanup()


class TravelLinearityTest:
    """
    Conducts a linearity test by traveling data along generated directions and measuring cosine similarities.

    ### Attributes
    - `epsilons (List[float])`: The list of epsilon values to test.
    - `method (str)`: The method to use for generating travel directions.
    - `batch_size (int)`: The batch size for the DataLoader.
    - `num_workers (int)`: The number of workers for the DataLoader.
    - `delta (float)`: The small perturbation for calculating movement vectors.
    - `seed (int)`: The seed for random number generation.
    - `bound (bool)`: Whether to clamp the moved data to [0, 1].
    - `use_cuda (bool)`: Whether to use CUDA for computation.
    - `devices (Devices)`: The devices to use for computation.

    ### Methods
    - `__call__(model, dataset, verbose) -> Dict[float, List[float]]`: Conducts the linearity test on the given model and dataset.
    """

    def __init__(
        self,
        min_eps: float,
        max_eps: float,
        num_steps: int,
        method: Literal[
            "fgsm",
            "signed_rand",
        ] = "fgsm",
        batch_size: int = 1,
        num_workers: int = 0,
        delta: float = 1.0e-3,
        seed: Optional[int] = 0,
        bound: bool = True,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        Initialize the TravelLinearityTest class.

        ### Parameters
        - `min_eps (float)`: The minimum epsilon value for traveling.
        - `max_eps (float)`: The maximum epsilon value for traveling.
        - `num_steps (int)`: The number of steps between min_eps and max_eps.
        - `method (str, optional)`: The method to use for generating travel directions. Defaults to "fgsm".
        - `batch_size (int, optional)`: The batch size for the DataLoader. Defaults to 1.
        - `num_workers (int, optional)`: The number of workers for the DataLoader. Defaults to 0.
        - `delta (float, optional)`: The small perturbation for calculating movement vectors. Defaults to 1.0e-3.
        - `seed (int, optional)`: The seed for random number generation. Defaults to 0.
        - `bound (bool, optional)`: Whether to clamp the moved data to [0, 1]. Defaults to True.
        - `use_cuda (bool, optional)`: Whether to use CUDA for computation. Defaults to False.
        - `devices (Devices, optional)`: The devices to use for computation. Defaults to "auto".
        """
        self.epsilons = [
            float(eps) for eps in torch.linspace(min_eps, max_eps, num_steps)
        ]
        self.method = method
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.delta = delta
        self.seed = seed
        self.bound = bound

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
        Conducts the linearity test on the given model and dataset.

        ### Parameters
        - `model (Module)`: The model to test.
        - `dataset (Dataset)`: The dataset to test on.
        - `verbose (bool, optional)`: Whether to print progress. Defaults to False.

        ### Returns
        - `Dict[float, List[float]]`: A dictionary with epsilon values as keys and lists of cosine similarities as values.
        """
        model.eval()

        # GPU
        if self.use_cuda:
            nprocs = len(self.devices)

            dist.run_fn(
                fn=run,
                args=(
                    nprocs,
                    model,
                    dataset,
                    self.epsilons,
                    self.delta,
                    self.batch_size,
                    self.num_workers,
                    self.method,
                    self.seed,
                    self.bound,
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
                model=model,
                dataset=dataset,
                epsilons=self.epsilons,
                delta=self.delta,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                method=self.method,
                seed=self.seed,
                bound=self.bound,
                verbose=verbose,
            )

        # linearity
        linearity = dict()

        cossims_fnames = ls(
            path=temp_dir,
            fext=".pt",
            sort=True,
        )

        for eps_idx, eps in enumerate(self.epsilons):
            fnames = [
                fname
                for fname in cossims_fnames
                if f"{self.method}-{eps_idx}-" in fname
            ]

            for fname in fnames:
                cossims = torch.load(os.path.join(temp_dir, fname))

                if eps in linearity:
                    linearity[eps] += cossims
                else:
                    linearity[eps] = cossims

        rmdir(temp_dir)

        return linearity


class TravelLinearityTests:
    def __init__(
        self,
        min_eps: float,
        max_eps: float,
        num_steps: int,
        method: Literal[
            "fgsm",
            "signed_rand",
        ] = "fgsm",
        batch_size: int = 1,
        num_workers: int = 0,
        delta: float = 1.0e-3,
        seed: Optional[int] = 0,
        bound: bool = True,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        Initializes the TravelLinearityTests class.

        ### Parameters
        - `min_eps (float)`: The minimum epsilon value for traveling.
        - `max_eps (float)`: The maximum epsilon value for traveling.
        - `num_steps (int)`: The number of steps between min_eps and max_eps.
        - `method (str, optional)`: The method to use for generating travel directions. Defaults to "fgsm".
        - `batch_size (int, optional)`: The batch size for the DataLoader. Defaults to 1.
        - `num_workers (int, optional)`: The number of workers for the DataLoader. Defaults to 0.
        - `delta (float, optional)`: The small perturbation for calculating movement vectors. Defaults to 1.0e-3.
        - `seed (int, optional)`: The seed for random number generation. Defaults to 0.
        - `bound (bool, optional)`: Whether to clamp the moved data to [0, 1]. Defaults to True.
        - `use_cuda (bool, optional)`: Whether to use CUDA for computation. Defaults to False.
        - `devices (Devices, optional)`: The devices to use for computation. Defaults to "auto".
        """
        self.test = TravelLinearityTest(
            min_eps=min_eps,
            max_eps=max_eps,
            num_steps=num_steps,
            method=method,
            batch_size=batch_size,
            num_workers=num_workers,
            delta=delta,
            seed=seed,
            bound=bound,
            use_cuda=use_cuda,
            devices=devices,
        )

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
        ckpt_dir: Path,
        save_dir: Path,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> None:
        """
        Conducts the linearity test on the given model and dataset on multiple checkpoints.

        ### Parameters
        - `model (Module)`: The model to evaluate.
        - `dataset (Dataset)`: The dataset to evaluate on.
        - `ckpt_dir (Path)`: The directory containing the checkpoints.
        - `save_dir (Path)`: The directory to save the test results.
        - `verbose (bool)`: Whether to display progress. Defaults to False.
        - `desc (Optional[str])`: Description for the progress bar. Defaults to None.

        ### Returns
        - `None`

        ### Notes
        - This function tests the model on all checkpoints in the specified directory.
        - The results for each checkpoint are saved to the specified directory.
        """
        ckpt_fnames = ls(ckpt_dir, fext=".ckpt", sort=True)
        ckpt_fnames = [fname for fname in ckpt_fnames if "epoch=" in fname]

        for epoch, ckpt_fname in enumerate(ckpt_fnames):
            epoch_id = f"epoch={epoch}"

            load_state_dict(
                model,
                os.path.join(ckpt_dir, ckpt_fname),
                map_location="cpu",
            )

            linearity_data_per_epoch = self.test(
                model=model,
                dataset=dataset,
                verbose=verbose,
                desc=f"TravelLinearityTest ({epoch_id})" if desc is None else desc,
            )

            save_in_csv(
                dictionary=linearity_data_per_epoch,
                save_dir=save_dir,
                save_name=epoch_id,
                index_col="idx",
            )
