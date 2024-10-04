import lightning as L
import torch
from torch.nn import CrossEntropyLoss, Softmax

from ... import utils
from ...typing import Config, Dataset, Devices, Dict, List, Module, Optional, Tensor
from ..datasets import to_dataloader
from ..train.utils import load_accelerator_and_devices

__all__ = [
    "EvalTest",
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
            reduction="none",
            label_smoothing=self.config["label_smoothing"],
        )
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
        signed_target_confs: Tensor = confs[:, targets].diag() * signs

        self.idx.append(indices.cpu())
        self.gt.append(targets.cpu())
        self.pred.append(preds.cpu())
        self.loss.append(losses.cpu())
        self.conf.append(signed_confs.cpu())
        self.gt_conf.append(signed_target_confs.cpu())

    def on_test_epoch_end(self) -> None:
        idx: List[Tensor] = self.all_gather(self.idx)
        gt: List[Tensor] = self.all_gather(self.gt)
        pred: List[Tensor] = self.all_gather(self.pred)
        loss: List[Tensor] = self.all_gather(self.loss)
        conf: List[Tensor] = self.all_gather(self.conf)
        gt_conf: List[Tensor] = self.all_gather(self.gt_conf)

        idx = torch.cat(idx, dim=-1).flatten()
        idx, indices = torch.sort(idx)
        gt = torch.cat(gt, dim=-1).flatten()[indices]
        pred = torch.cat(pred, dim=-1).flatten()[indices]
        loss = torch.cat(loss, dim=-1).flatten()[indices]
        conf = torch.cat(conf, dim=-1).flatten()[indices]
        gt_conf = torch.cat(gt_conf, dim=-1).flatten()[indices]

        self.res_dict["idx"] = [int(i) for i in idx]
        self.res_dict["gt"] = [int(g) for g in gt]
        self.res_dict["pred"] = [int(p) for p in pred]
        self.res_dict["loss"] = [float(l) for l in loss]
        self.res_dict["conf"] = [float(c) for c in conf]
        self.res_dict["gt_conf"] = [float(gc) for gc in gt_conf]


class EvalTest:
    """
    Performs evaluation of a model on a given dataset with support for techniques like Mixup, CutMix, and label smoothing.

    ### Attributes
    - `batch_size (int)`: The number of samples per batch.
    - `num_workers (int)`: The number of worker threads for loading data.
    - `mixup_alpha (Optional[float])`: The alpha value for Mixup augmentation. Defaults to 0.0.
    - `cutmix_alpha (Optional[float])`: The alpha value for CutMix augmentation. Defaults to 0.0.
    - `label_smoothing (Optional[float])`: The label smoothing factor. Defaults to 0.0.
    - `seed (Optional[int])`: The seed for random number generators to ensure reproducibility. Defaults to None.
    - `devices (Optional[Devices])`: The devices to run the evaluation on (e.g., CPU, GPU). Defaults to "auto".
    - `runner (L.Trainer)`: The Lightning Trainer used to run the evaluation.
    - `config (Dict[str, Any])`: The configuration dictionary storing the evaluation settings.

    ### Methods
    - `__init__(...)`: Initializes the EvalTest class with the specified configuration.
    - `__call__(model, dataset) -> Dict[str, List[float]]`: Evaluates the model on the provided dataset and returns the evaluation metrics.
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
    ) -> None:
        """
        - `batch_size (int)`: The number of samples per batch.
        - `num_workers (int)`: The number of worker threads for loading data.
        - `mixup_alpha (Optional[float])`: The alpha value for Mixup augmentation. Defaults to 0.0.
        - `cutmix_alpha (Optional[float])`: The alpha value for CutMix augmentation. Defaults to 0.0.
        - `label_smoothing (Optional[float])`: The label smoothing factor. Defaults to 0.0.
        - `seed (Optional[int])`: The seed for random number generators to ensure reproducibility. Defaults to None.
        - `devices (Optional[Devices])`: The devices to run the evaluation on (e.g., CPU, GPU). Defaults to "auto".
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
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
            "label_smoothing": self.label_smoothing,
        }

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
    ) -> Dict[str, List[float]]:
        """
        Evaluates the model on the provided dataset and returns the evaluation metrics.

        ### Parameters
        - `model (Module)`: The neural network model to evaluate.
        - `dataset (Dataset)`: The dataset on which the evaluation is performed.

        ### Returns
        - `Dict[str, List[float]]`: A dictionary containing the evaluation metrics:
            - `gt`: Ground truth labels.
            - `pred`: Predicted labels.
            - `loss`: Loss values for each sample.
            - `conf`: Confidence scores for each prediction.

        ### Notes
        - This method wraps the model in a `Wrapper` class, converts the dataset into a DataLoader, and runs the evaluation using the Lightning Trainer.
        """
        res_dict = {}
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

        return res_dict
