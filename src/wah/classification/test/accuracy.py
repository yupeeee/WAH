import lightning as L
import torch

from ... import utils
from ...typing import Config, Dataset, Devices, Dict, List, Module, Optional, Tensor
from ..datasets import to_dataloader
from ..train.utils import load_accelerator_and_devices

__all__ = [
    "AccuracyTest",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        config: Config,
        res_dict: Dict[str, List],
    ) -> None:
        super().__init__()

        self.model = model
        self.config = config
        self.res_dict = res_dict

        self.corrects = []

    def test_step(self, batch, batch_idx):
        data: Tensor
        targets: Tensor

        data, targets = batch

        outputs: Tensor = self.model(data)
        _, preds = outputs.topk(k=self.config["top_k"], dim=-1)

        for k in range(self.config["top_k"]):
            self.corrects.append(preds[:, k].eq(targets).sum().cpu())

    def on_test_epoch_end(self) -> None:
        corrects: List[Tensor] = self.all_gather(self.corrects)
        if corrects[0].dim() == 0:
            corrects = [c.unsqueeze(0) for c in corrects]
        corrects = torch.cat(corrects, dim=-1).flatten().sum()

        self.res_dict["corrects"] = float(corrects)


class AccuracyTest:
    """
    A class for evaluating the top-k accuracy of a model on a given dataset using PyTorch Lightning.

    ### Attributes
    - `batch_size` (int): The batch size for the test.
    - `num_workers` (int): The number of workers for the DataLoader.
    - `top_k` (Optional[int]): The value of `k` for top-k accuracy. Defaults to `1`.
    - `mixup_alpha` (Optional[float]): Alpha value for Mixup data augmentation. Defaults to `0.0`.
    - `cutmix_alpha` (Optional[float]): Alpha value for CutMix data augmentation. Defaults to `0.0`.
    - `seed` (Optional[int]): Random seed for deterministic behavior. Defaults to `None`.
    - `devices` (Optional[Devices]): The devices to run the test on. Defaults to `"auto"`.
    - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.

    ### Methods
    - `__call__(model: Module, dataset: Dataset) -> float`: Runs the accuracy test and returns the accuracy.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        top_k: Optional[int] = 1,
        mixup_alpha: Optional[float] = 0.0,
        cutmix_alpha: Optional[float] = 0.0,
        seed: Optional[int] = None,
        devices: Optional[Devices] = "auto",
        verbose: Optional[bool] = True,
    ) -> None:
        """
        - `batch_size` (int): The batch size for the test.
        - `num_workers` (int): The number of workers for the DataLoader.
        - `top_k` (Optional[int], optional): The value of `k` for top-k accuracy. Defaults to `1`.
        - `mixup_alpha` (Optional[float], optional): Alpha value for Mixup data augmentation. Defaults to `0.0`.
        - `cutmix_alpha` (Optional[float], optional): Alpha value for CutMix data augmentation. Defaults to `0.0`.
        - `seed` (Optional[int], optional): Random seed for deterministic behavior. Defaults to `None`.
        - `devices` (Optional[Devices], optional): The devices to run the test on. Defaults to `"auto"`.
        - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.top_k = top_k
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.seed = seed
        self.devices = devices
        self.verbose = verbose

        utils.seed(self.seed)
        accelerator, devices = load_accelerator_and_devices(self.devices)

        if not verbose:
            utils.disable_lightning_logging()

        self.runner = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=False,
            max_epochs=1,
            log_every_n_steps=None,
            deterministic="warn" if self.seed is not None else False,
            enable_progress_bar=verbose,
            enable_model_summary=verbose,
        )
        self.config = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "top_k": self.top_k,
            "mixup_alpha": self.mixup_alpha,
            "cutmix_alpha": self.cutmix_alpha,
        }

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
    ) -> float:
        """
        Runs the accuracy test for the given model and dataset.

        ### Parameters
        - `model` (Module): The model to be evaluated.
        - `dataset` (Dataset): The dataset to evaluate the model on.

        ### Returns
        - `float`: The computed accuracy of the model on the dataset.
        """
        res_dict = {}
        model = Wrapper(model, self.config, res_dict)
        dataloader = to_dataloader(
            dataset=dataset,
            train=False,
            **self.config,
        )

        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )

        acc = res_dict["corrects"] / len(dataset)

        return acc
