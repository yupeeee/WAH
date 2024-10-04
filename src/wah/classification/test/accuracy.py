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
        corrects = torch.cat(corrects, dim=-1).flatten().sum()

        self.res_dict["corrects"] = float(corrects)


class AccuracyTest:
    """
    A class for evaluating the top-k accuracy of a neural network model on a given dataset.

    ### Attributes
    - `batch_size` (int): The batch size to use during evaluation.
    - `num_workers` (int): The number of worker threads for loading data.
    - `top_k` (int, optional): The value of k for top-k accuracy. Defaults to 1.
    - `mixup_alpha` (float, optional): The alpha value for mixup augmentation. Defaults to 0.0.
    - `cutmix_alpha` (float, optional): The alpha value for cutmix augmentation. Defaults to 0.0.
    - `seed` (int, optional): The random seed for reproducibility. Defaults to None.
    - `devices` (Devices, optional): The devices to use for computation. Defaults to "auto".

    ### Methods
    - `__call__(model, dataset) -> float`: Evaluates the accuracy of the model on the given dataset.
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
    ) -> None:
        """
        - `batch_size` (int): The batch size to use during evaluation.
        - `num_workers` (int): The number of worker threads for loading data.
        - `top_k` (int, optional): The value of k for top-k accuracy. Defaults to 1.
        - `mixup_alpha` (float, optional): The alpha value for mixup augmentation. Defaults to 0.0.
        - `cutmix_alpha` (float, optional): The alpha value for cutmix augmentation. Defaults to 0.0.
        - `seed` (int, optional): The random seed for reproducibility. Defaults to None.
        - `devices` (Devices, optional): The devices to use for computation. Defaults to "auto".
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.top_k = top_k
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.seed = seed
        self.devices = devices

        utils.random.seed(self.seed)
        accelerator, devices = load_accelerator_and_devices(self.devices)

        self.runner = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            logger=False,
            max_epochs=1,
            log_every_n_steps=None,
            deterministic="warn" if self.seed is not None else False,
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
        Evaluates the accuracy of the model on the given dataset.

        ### Parameters
        - `model (Module)`: The neural network model to evaluate.
        - `dataset (Dataset)`: The dataset to evaluate the model on.

        ### Returns
        - `float`: The top-k accuracy of the model on the dataset.
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
