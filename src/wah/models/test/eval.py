import os

import torch
import tqdm

from ...typing import (
    DataLoader,
    Dataset,
    Devices,
    Dict,
    List,
    Literal,
    Module,
    Optional,
    Path,
    Tensor,
    Union,
)
from ...datasets.base import ClassificationDataset
from ...utils import dist
from ...utils.dictionary import save_in_csv
from ...utils.path import ls, rmdir
from ...utils.time import current_time
from ..load import load_state_dict

__all__ = [
    "EvalTest",
    "EvalTests",
]

time_id = current_time()
temp_dir = f"wahtmpdir@EvalTest{time_id}"


def set_loss_fn(
    loss: Literal["ce"],
    **kwargs,
) -> Module:
    """
    Sets the loss function for evaluation.

    ### Parameters
    - `loss (Literal["ce"])`: The type of loss function to use. Currently only "ce" (CrossEntropyLoss) is supported.
    - `**kwargs`: Additional keyword arguments for the loss function.

    ### Returns
    - `Module`: The loss function module.

    ### Raises
    - `ValueError`: If an unsupported loss type is provided.
    """
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
    """
    Runs the evaluation process for a single process.

    ### Parameters
    - `rank` (int): The rank of the current process.
    - `nprocs` (int): The total number of processes.
    - `model (Module)`: The model to evaluate.
    - `dataset (Dataset)`: The dataset to evaluate on.
    - `loss_fn (Module)`: The loss function to use.
    - `batch_size (int)`: The batch size to use. Defaults to 1.
    - `num_workers (int)`: The number of workers to use for data loading. Defaults to 0.
    - `verbose (bool)`: Whether to display progress. Defaults to False.
    - `desc (Optional[str])`: Description for the progress bar. Defaults to None.

    ### Returns
    - `None`

    ### Notes
    - This function evaluates the model on the given dataset, computing loss and confidence for each batch.
    - The results are saved temporarily for later aggregation.
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

    # evaluate
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

        confs: Tensor = torch.nn.Softmax(dim=-1)(outputs)
        preds: Tensor = torch.argmax(confs, dim=-1)

        results: Tensor = torch.eq(preds, targets.to(rank))
        signed_confs: Tensor = confs[:, targets].diag() * (results.int() - 0.5).sign()

        eval_data = (
            indices,
            targets.to(torch.device("cpu")),
            preds.to(torch.device("cpu")),
            loss.to(torch.device("cpu")),
            signed_confs.to(torch.device("cpu")),
        )

        torch.save(
            eval_data,
            os.path.join(
                temp_dir,
                f"{batch_idx}-{current_time()}.pt",
            ),
        )

    # DDP cleanup
    if rank != torch.device("cpu"):
        dist.cleanup()


class EvalTest:
    def __init__(
        self,
        loss: Literal["ce"] = "ce",
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        Initializes the EvalTest class.

        - `loss (Literal["ce"])`: The loss function to use. Defaults to "ce".
        - `batch_size (int)`: The batch size to use. Defaults to 1.
        - `num_workers (int)`: The number of workers for data loading. Defaults to 0.
        - `use_cuda (bool)`: Whether to use CUDA for computation. Defaults to False.
        - `devices (Optional[Devices])`: The devices to use for computation. Defaults to "auto".
        """
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
    ) -> Dict[str, Union[int, float]]:
        """
        Evaluates the model on the given dataset.

        ### Parameters
        - `model (Module)`: The model to evaluate.
        - `dataset (Dataset)`: The dataset to evaluate on.
        - `verbose (bool)`: Whether to display progress. Defaults to False.
        - `desc (Optional[str])`: Description for the progress bar. Defaults to None.

        ### Returns
        - `Dict[str, Union[int, float]]`: A dictionary containing evaluation results (indices, ground truths, predictions, losses, confidences).

        ### Notes
        - This function supports both CPU and GPU evaluation.
        - The results are aggregated and returned in a dictionary.
        """
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

        # eval
        indices: List[int] = []
        ground_truths: List[int] = []
        preds: List[int] = []
        losses: List[float] = []
        confs: List[float] = []

        file_names = ls(temp_dir)

        for file_name in file_names:
            i, g, p, l, c = torch.load(os.path.join(temp_dir, file_name))
            indices.append(i)
            ground_truths.append(g)
            preds.append(p)
            losses.append(l)
            confs.append(c)

        indices = torch.cat(indices, dim=0)
        ground_truths = torch.cat(ground_truths, dim=0)
        preds = torch.cat(preds, dim=0)
        losses = torch.cat(losses, dim=0)
        confs = torch.cat(confs, dim=0)

        ground_truths = ground_truths[indices]
        preds = preds[indices]
        losses = losses[indices]
        confs = confs[indices]
        indices, _ = indices.sort()

        rmdir(temp_dir)

        return {
            "idx": [int(i) for i in indices],
            "gt": [int(g) for g in ground_truths],
            "pred": [int(p) for p in preds],
            "loss": [float(l) for l in losses],
            "conf": [float(c) for c in confs],
        }


class EvalTests:
    def __init__(
        self,
        loss: Literal["ce"] = "ce",
        batch_size: int = 1,
        num_workers: int = 0,
        use_cuda: bool = False,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        Initializes the EvalTests class.

        - `loss (Literal["ce"])`: The loss function to use. Defaults to "ce".
        - `batch_size (int)`: The batch size to use. Defaults to 1.
        - `num_workers (int)`: The number of workers for data loading. Defaults to 0.
        - `use_cuda (bool)`: Whether to use CUDA for computation. Defaults to False.
        - `devices (Optional[Devices])`: The devices to use for computation. Defaults to "auto".
        """
        self.test = EvalTest(
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
        save_dir: Path,
        verbose: bool = False,
        desc: Optional[str] = None,
    ) -> None:
        """
        Evaluates the model on multiple checkpoints.

        ### Parameters
        - `model (Module)`: The model to evaluate.
        - `dataset (Dataset)`: The dataset to evaluate on.
        - `ckpt_dir (Path)`: The directory containing the checkpoints.
        - `save_dir (Path)`: The directory to save the evaluation results.
        - `verbose (bool)`: Whether to display progress. Defaults to False.
        - `desc (Optional[str])`: Description for the progress bar. Defaults to None.

        ### Returns
        - `None`

        ### Notes
        - This function evaluates the model on all checkpoints in the specified directory.
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

            eval_data_per_epoch = self.test(
                model=model,
                dataset=dataset,
                verbose=verbose,
                desc=f"EvalTest ({epoch_id})" if desc is None else desc,
            )

            save_in_csv(
                dictionary=eval_data_per_epoch,
                save_dir=save_dir,
                save_name=epoch_id,
                index_col="idx",
            )
