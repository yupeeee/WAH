"""
Implementation of TID (Training Inference Discrepancy) in
"Understanding the Failure of Batch Normalization for Transformers in NLP"
by Jiaxi Wang, Ji Wu, and Lei Huang (NeurIPS 2022)

https://arxiv.org/abs/2210.05153
"""

import lightning as L
import torch

from ... import utils
from ...module import _getattr, get_attrs
from ...typing import Dataset, Devices, Dict, List, Module, Optional, Tensor
from ..datasets import to_dataloader
from ..models.feature_extraction import FeatureExtractor
from ..train.utils import load_accelerator_and_devices
from .utils import process_gathered_data

__all__ = [
    "TIDTest",
]


def get_running_stats(
    model: Module,
    attrs: List[str],
) -> Dict[str, Tensor]:
    running_stats: Dict[str, Tensor] = {}

    for attr in attrs:
        module = _getattr(model, attr)
        running_stats[f"{attr}.running_mean"] = module.running_mean
        running_stats[f"{attr}.running_var"] = module.running_var

    return running_stats


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        res_dict: Dict[str, List],
        eps: Optional[float] = 1e-8,
    ) -> None:
        super().__init__()

        self.model = model
        self.res_dict = res_dict
        self.eps = eps

        self.attrs = get_attrs(model, specify="BatchNorm2d")
        assert len(self.attrs) > 0, f"Model does not have BatchNorm2d."

        # check if weight contains running_mean and running_var
        weight_attrs = [
            attr for attr in model.state_dict().keys() if ".running_" in attr
        ]
        assert len(self.attrs) * 2 == len(  # mean, var per attr
            weight_attrs
        ), f"There seems to be an issue with your model. Please verify if the BatchNorm2d layers have track_running_stats set to False."
        # weight_attrs = [attr.replace(".running_mean", "") for attr in weight_attrs[::2]]

        # store running stats
        self.running_stats = get_running_stats(model, self.attrs)

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
            if isinstance(attr_input, tuple):
                if len(attr_input) == 1:
                    attr_input = attr_input[0]
                else:
                    attr_input = torch.cat(list(attr_input), dim=0)

            mean = torch.mean(attr_input, dim=(0, 2, 3))
            var = torch.var(attr_input, dim=(0, 2, 3), unbiased=False)

            getattr(self, f"{attr}_mean").append(
                torch.norm(
                    mean.to(self.device)
                    - self.running_stats[f"{attr}.running_mean"].to(self.device)
                ).cpu()
            )
            getattr(self, f"{attr}_var").append(
                torch.norm(
                    var.to(self.device)
                    - self.running_stats[f"{attr}.running_var"].to(self.device)
                ).cpu()
            )

    def on_test_epoch_end(self) -> None:
        for attr in self.attrs:
            mean_tid: List[Tensor] = self.all_gather(getattr(self, f"{attr}_mean"))
            var_tid: List[Tensor] = self.all_gather(getattr(self, f"{attr}_var"))

            mean_tid = process_gathered_data(mean_tid, 1, -1, None).mean()
            var_tid = process_gathered_data(var_tid, 1, -1, None).mean()

            running_var_l2 = torch.norm(self.running_stats[f"{attr}.running_var"]).cpu() + self.eps
            mean_tid = mean_tid / running_var_l2
            var_tid = var_tid / running_var_l2

            self.res_dict[f"{attr}_mean"] = float(mean_tid)
            self.res_dict[f"{attr}_var"] = float(var_tid)


class TIDTest:
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        seed: Optional[int] = None,
        devices: Optional[Devices] = "auto",
        amp: Optional[bool] = False,
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
        eps: Optional[float] = 1e-8,
    ) -> Dict[str, float]:
        res_dict = {}
        dataset.set_return_data_only()
        model = Wrapper(model, res_dict, eps)
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
