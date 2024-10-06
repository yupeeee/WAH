import lightning as L
import torch
from torch.nn import Softmax

from ... import utils
from ...typing import Dataset, Devices, Dict, List, Module, Optional, Tensor
from ..datasets import to_dataloader
from ..train.utils import load_accelerator_and_devices

__all__ = [
    "PredTest",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        res_dict: Dict[str, List],
    ) -> None:
        super().__init__()

        self.model = model
        self.res_dict = res_dict

        self.softmax = Softmax(dim=-1)

        self.idx = []
        self.pred = []

    def test_step(self, batch, batch_idx):
        indices: Tensor
        data: Tensor

        indices, data = batch

        outputs: Tensor = self.model(data)

        confs: Tensor = self.softmax(outputs)
        preds: Tensor = torch.argmax(confs, dim=-1)

        self.idx.append(indices.cpu())
        self.pred.append(preds.cpu())

    def on_test_epoch_end(self) -> None:
        idx: List[Tensor] = self.all_gather(self.idx)
        pred: List[Tensor] = self.all_gather(self.pred)

        idx = torch.cat(idx, dim=-1).flatten()
        idx, indices = torch.sort(idx)
        pred = torch.cat(pred, dim=-1).flatten()[indices]

        self.res_dict["pred"] = [int(p) for p in pred]


class PredTest:
    """
    A class for evaluating a model and collecting predictions for a dataset.

    ### Attributes
    - `batch_size` (int): The batch size for the test.
    - `num_workers` (int): The number of workers for the DataLoader.
    - `mixup_alpha` (Optional[float]): Alpha value for Mixup data augmentation. Defaults to `0.0`.
    - `cutmix_alpha` (Optional[float]): Alpha value for CutMix data augmentation. Defaults to `0.0`.
    - `seed` (Optional[int]): Random seed for deterministic behavior. Defaults to `None`.
    - `devices` (Optional[Devices]): The devices to run the test on. Defaults to `"auto"`.

    ### Methods
    - `__call__(model: Module, dataset: Dataset) -> List[int]`: Runs the test and returns a list of predictions.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        mixup_alpha: Optional[float] = 0.0,
        cutmix_alpha: Optional[float] = 0.0,
        seed: Optional[int] = None,
        devices: Optional[Devices] = "auto",
    ) -> None:
        """
        - `batch_size` (int): The batch size for the test.
        - `num_workers` (int): The number of workers for the DataLoader.
        - `mixup_alpha` (Optional[float], optional): Alpha value for Mixup data augmentation. Defaults to `0.0`.
        - `cutmix_alpha` (Optional[float], optional): Alpha value for CutMix data augmentation. Defaults to `0.0`.
        - `seed` (Optional[int], optional): Random seed for deterministic behavior. Defaults to `None`.
        - `devices` (Optional[Devices], optional): The devices to run the test on. Defaults to `"auto"`.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
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
            "mixup_alpha": self.mixup_alpha,
            "cutmix_alpha": self.cutmix_alpha,
        }

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
    ) -> List[int]:
        """
        Runs the test for the given model and dataset.

        ### Parameters
        - `model` (Module): The model to be evaluated.
        - `dataset` (Dataset): The dataset to evaluate the model on.

        ### Returns
        - `List[int]`: A list of predicted labels for the dataset.
        """
        res_dict = {}
        dataset.set_return_data_only()
        dataset.set_return_w_index()
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

        return res_dict["pred"]
