import lightning as L
from torch.nn import CrossEntropyLoss, Softmax

from ... import utils
from ...typing import Config, Dataset, Devices, Dict, List, Module, Optional, Tensor
from ..datasets import to_dataloader
from ..train.utils import load_accelerator_and_devices
from .utils import process_gathered_data

__all__ = [
    "LossTest",
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

        self.criterion = CrossEntropyLoss(
            label_smoothing=self.config["label_smoothing"],
        )
        self.softmax = Softmax(dim=-1)

        self.loss = []

    def test_step(self, batch, batch_idx):
        data: Tensor
        targets: Tensor

        data, targets = batch

        outputs: Tensor = self.model(data)

        losses: Tensor = self.criterion(outputs, targets)

        self.loss.append(losses.cpu())

    def on_test_epoch_end(self) -> None:
        loss: List[Tensor] = self.all_gather(self.loss)
        loss = process_gathered_data(loss, 1, -1, None)

        self.res_dict["loss"] = float(loss)


class LossTest:
    """
    A class for evaluating the average loss of a model on a given dataset using PyTorch Lightning.

    ### Attributes
    - `batch_size` (int): The batch size for the test.
    - `num_workers` (int): The number of workers for the DataLoader.
    - `mixup_alpha` (Optional[float]): Alpha value for Mixup data augmentation. Defaults to `0.0`.
    - `cutmix_alpha` (Optional[float]): Alpha value for CutMix data augmentation. Defaults to `0.0`.
    - `label_smoothing` (Optional[float]): The amount of label smoothing to apply to the loss. Defaults to `0.0`.
    - `seed` (Optional[int]): Random seed for deterministic behavior. Defaults to `None`.
    - `devices` (Optional[Devices]): The devices to run the test on. Defaults to `"auto"`.
    - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.

    ### Methods
    - `__call__(model: Module, dataset: Dataset) -> float`: Runs the loss evaluation and returns the average loss.
    """

    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        mixup_alpha: Optional[float] = 0.0,
        cutmix_alpha: Optional[float] = 0.0,
        label_smoothing: Optional[float] = 0.0,
        seed: Optional[int] = None,
        devices: Optional[Devices] = "auto",
        verbose: Optional[bool] = True,
    ) -> None:
        """
        - `batch_size` (int): The batch size for the test.
        - `num_workers` (int): The number of workers for the DataLoader.
        - `mixup_alpha` (Optional[float], optional): Alpha value for Mixup data augmentation. Defaults to `0.0`.
        - `cutmix_alpha` (Optional[float], optional): Alpha value for CutMix data augmentation. Defaults to `0.0`.
        - `label_smoothing` (Optional[float], optional): The amount of label smoothing to apply to the loss. Defaults to `0.0`.
        - `seed` (Optional[int], optional): Random seed for deterministic behavior. Defaults to `None`.
        - `devices` (Optional[Devices], optional): The devices to run the test on. Defaults to `"auto"`.
        - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
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
            "mixup_alpha": self.mixup_alpha,
            "cutmix_alpha": self.cutmix_alpha,
            "label_smoothing": self.label_smoothing,
        }

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
    ) -> float:
        """
        Runs the loss evaluation for the given model and dataset.

        ### Parameters
        - `model` (Module): The model to be evaluated.
        - `dataset` (Dataset): The dataset to evaluate the model on.

        ### Returns
        - `float`: The computed mean loss of the model on the dataset.
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

        return res_dict["loss"]
