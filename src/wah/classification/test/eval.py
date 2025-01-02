import lightning as L
import torch
from torch.nn import Softmax

from ... import utils
from ...typing import Dataset, Devices, Dict, List, Module, Optional, Tensor
from ..datasets import load_dataloader
from ..train.criterion import load_criterion
from ..train.utils import load_accelerator_and_devices
from .utils import process_gathered_data

__all__ = [
    "EvalTest",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        res_dict: Dict[str, List],
        **kwargs,
    ) -> None:
        super().__init__()

        self.model = model
        self.res_dict = res_dict
        self.config = kwargs

        self.criterion = load_criterion(train=False, reduction="none", **self.config)
        self.softmax = Softmax(dim=-1)

        self.idx = []
        self.gt = []
        self.pred = []
        self.loss = []
        self.conf = []
        self.gt_conf = []

    def test_step(self, batch, batch_idx):
        indices: Tensor
        data: Tensor
        targets: Tensor

        indices, (data, targets) = batch

        outputs: Tensor = self.model(data)

        losses: Tensor = self.criterion(outputs, targets)

        confs: Tensor = self.softmax(outputs)
        preds: Tensor = torch.argmax(confs, dim=-1)

        results: Tensor = torch.eq(preds, targets)
        signs: Tensor = (results.int() - 0.5).sign()
        signed_confs: Tensor = confs[:, preds].diag() * signs
        signed_gt_confs: Tensor = confs[:, targets].diag() * signs

        self.idx.append(indices.cpu())
        self.gt.append(targets.cpu())
        self.pred.append(preds.cpu())
        self.loss.append(losses.cpu())
        self.conf.append(signed_confs.cpu())
        self.gt_conf.append(signed_gt_confs.cpu())

    def on_test_epoch_end(self) -> None:
        idx: List[Tensor] = self.all_gather(self.idx)
        gt: List[Tensor] = self.all_gather(self.gt)
        pred: List[Tensor] = self.all_gather(self.pred)
        loss: List[Tensor] = self.all_gather(self.loss)
        conf: List[Tensor] = self.all_gather(self.conf)
        gt_conf: List[Tensor] = self.all_gather(self.gt_conf)

        idx = process_gathered_data(idx, 2, 1, (1, 0))
        gt = process_gathered_data(gt, 2, 1, (1, 0))
        pred = process_gathered_data(pred, 2, 1, (1, 0))
        loss = process_gathered_data(loss, 2, 1, (1, 0))
        conf = process_gathered_data(conf, 2, 1, (1, 0))
        gt_conf = process_gathered_data(gt_conf, 2, 1, (1, 0))

        self.res_dict["idx"] = [int(i) for i in idx]
        self.res_dict["gt"] = [int(g) for g in gt]
        self.res_dict["pred"] = [int(p) for p in pred]
        self.res_dict["loss"] = [float(l) for l in loss]
        self.res_dict["conf"] = [float(c) for c in conf]
        self.res_dict["gt_conf"] = [float(gc) for gc in gt_conf]


class EvalTest:
    """
    A class for evaluating a model on a dataset and collecting various metrics
    such as predictions, losses, and confidence scores.

    ### Attributes
    - `batch_size` (int): The batch size for the test.
    - `num_workers` (int): The number of workers for the DataLoader.
    - `mixup_alpha` (Optional[float]): Alpha value for Mixup data augmentation. Defaults to `0.0`.
    - `cutmix_alpha` (Optional[float]): Alpha value for CutMix data augmentation. Defaults to `0.0`.
    - `label_smoothing` (Optional[float]): The amount of label smoothing to apply to the loss. Defaults to `0.0`.
    - `seed` (Optional[int]): Random seed for deterministic behavior. Defaults to `None`.
    - `devices` (Optional[Devices]): The devices to run the test on. Defaults to `"auto"`.
    - `amp` (Optional[bool]): Whether to use automatic mixed precision (AMP) for model evaluation. Defaults to `False`.
    - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.

    ### Methods
    - `__call__(model: Module, dataset: Dataset) -> Dict[str, List[float]]`:
    Runs the evaluation and returns a dictionary of results.

    ### Resulting Dictionary
    The returned dictionary contains the following keys:
    - `idx` (List[int]): The indices of the samples in the dataset.
    - `gt` (List[int]): The ground truth labels for each sample.
    - `pred` (List[int]): The predicted labels for each sample.
    - `loss` (List[float]): The loss values for each sample.
    - `conf` (List[float]): The confidence scores for the predicted labels,
    with the sign indicating correctness (positive for correct predictions, negative for incorrect).
    - `gt_conf` (List[float]): The confidence scores for the ground truth labels,
    with the sign indicating correctness (positive for correct predictions, negative for incorrect).
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
        amp: Optional[bool] = False,
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
        - `amp` (Optional[bool]): Whether to use automatic mixed precision (AMP) for model evaluation. Defaults to `False`.
        - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.devices = devices
        self.amp = amp
        self.verbose = verbose

        utils.seed(self.seed)
        accelerator, devices = load_accelerator_and_devices(self.devices)

        if not verbose:
            utils.disable_lightning_logging()

        self.runner = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision="16-mixed" if self.amp else "32-true",
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
    ) -> Dict[str, List[float]]:
        """
        Runs the evaluation for the given model and dataset.

        ### Parameters
        - `model` (Module): The model to be evaluated.
        - `dataset` (Dataset): The dataset to evaluate the model on.

        ### Returns
        - `Dict[str, List[float]]`: A dictionary containing evaluation results
        including predictions, losses, and confidence scores.
        """
        res_dict = {}
        dataset.set_return_w_index()
        model = Wrapper(model, res_dict, **self.config)
        dataloader = load_dataloader(
            dataset=dataset,
            train=False,
            **self.config,
        )

        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )

        return res_dict
