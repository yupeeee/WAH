"""
Implementation of TID (Training Inference Discrepancy) in
"Understanding the Failure of Batch Normalization for Transformers in NLP"
by Jiaxi Wang, Ji Wu, and Lei Huang (NeurIPS 2022)

https://arxiv.org/abs/2210.05153
"""

import lightning as L
import torch
from torch.nn import CrossEntropyLoss, Softmax

from ... import utils
from ...module import _getattr, get_attrs, get_module_name
from ...typing import Config, Dataset, Devices, Dict, List, Module, Optional, Tensor
from ..datasets import to_dataloader
from ..models.feature_extraction import FeatureExtractor
from ..train.utils import load_accelerator_and_devices

__all__ = [
    "TIDTest",
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

        self.attrs = get_attrs(model, specify="BatchNorm2d")
        assert len(self.attrs) > 0, f"Model does not have BatchNorm2d."

        self.input_extractor = FeatureExtractor(
            model=model,
            attrs=self.attrs,
            return_inputs=True,
        )

        for attr in self.attrs:
            setattr(self, f"{attr}_mean", [])
            setattr(self, f"{attr}_var", [])

    def test_step(self, batch, batch_idx):
        data: Tensor

        data = batch

        inputs: Tensor = self.input_extractor(data)

        for attr in self.attrs:
            attr_input = inputs[attr]
            mean = torch.mean(attr_input, dim=(0, 2, 3))
            var = torch.var(attr_input, dim=(0, 2, 3), unbiased=False)

            getattr(self, f"{attr}_mean").append(mean.cpu())
            getattr(self, f"{attr}_var").append(var.cpu())

    def on_test_epoch_end(self) -> None:
        for attr in self.attrs:
            means: List[Tensor] = self.all_gather(getattr(self, f"{attr}_mean"))
            vars: List[Tensor] = self.all_gather(getattr(self, f"{attr}_var"))

        idx: List[Tensor] = self.all_gather(self.idx)
        gt: List[Tensor] = self.all_gather(self.gt)
        pred: List[Tensor] = self.all_gather(self.pred)
        loss: List[Tensor] = self.all_gather(self.loss)
        conf: List[Tensor] = self.all_gather(self.conf)
        gt_conf: List[Tensor] = self.all_gather(self.gt_conf)

        idx = torch.cat(idx, dim=1).permute(1, 0).flatten()
        gt = torch.cat(gt, dim=1).permute(1, 0).flatten()
        pred = torch.cat(pred, dim=1).permute(1, 0).flatten()
        loss = torch.cat(loss, dim=1).permute(1, 0).flatten()
        conf = torch.cat(conf, dim=1).permute(1, 0).flatten()
        gt_conf = torch.cat(gt_conf, dim=1).permute(1, 0).flatten()

        self.res_dict["idx"] = [int(i) for i in idx]
        self.res_dict["gt"] = [int(g) for g in gt]
        self.res_dict["pred"] = [int(p) for p in pred]
        self.res_dict["loss"] = [float(l) for l in loss]
        self.res_dict["conf"] = [float(c) for c in conf]
        self.res_dict["gt_conf"] = [float(gc) for gc in gt_conf]


class TIDTest:
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        seed: Optional[int] = None,
        devices: Optional[Devices] = "auto",
        verbose: Optional[bool] = True,
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        }

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
    ) -> Dict[str, List[float]]:
        res_dict = {}
        dataset.set_return_data_only()
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
