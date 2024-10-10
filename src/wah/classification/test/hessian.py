import lightning as L
import torch
from torch.func import hessian

from ... import utils
from ...classification.datasets import to_dataloader
from ...classification.train.utils import load_accelerator_and_devices
from ...typing import Dataset, Devices, Dict, List, Module, Optional, Tensor

__all__ = [
    "HessianSigVals",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        res_dict: Dict[str, List],
        criterion: str = "CrossEntropyLoss",
    ) -> None:
        super().__init__()

        self.model = model
        self.res_dict = res_dict
        self.criterion = getattr(torch.nn, criterion)()

        self.num_data = []
        self.sigval = []

    def test_step(self, batch: Tensor, batch_idx):
        data, targets = batch

        batch_size = len(data)
        input_dim = batch[0].numel()

        # enable gradients
        with torch.inference_mode(False):
            data = data.clone().detach().requires_grad_(True)
            H: Tensor = hessian(lambda x: self.criterion(self.model(x), targets))(data)

        assert H.requires_grad
        H = H.detach().reshape(batch_size, input_dim, input_dim)

        sigvals: Tensor = torch.linalg.svdvals(H)

        self.num_data.append(batch_size)
        self.sigval.append(sigvals.cpu())

    def on_test_epoch_end(self) -> None:
        num_data: List[Tensor] = self.all_gather(self.num_data)
        sigval: List[Tensor] = self.all_gather(self.sigval)

        num_data = torch.cat(num_data, dim=0).flatten().sum()
        sigval: Tensor = torch.cat(sigval, dim=1).permute(1, 0, 2).reshape(num_data, -1)

        self.res_dict["sigval"] = sigval


class HessianSigVals:
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        seed: Optional[int] = None,
        devices: Optional[Devices] = "auto",
    ) -> None:
        self.batch_size = batch_size
        self.num_workers = num_workers
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
        }

    def __call__(
        self,
        model: Module,
        dataset: Dataset,
        criterion: str = "CrossEntropyLoss",
    ) -> Tensor:
        res_dict = {}
        model = Wrapper(model, res_dict, criterion)
        dataloader = to_dataloader(
            dataset=dataset,
            train=False,
            **self.config,
        )

        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )

        return res_dict["sigval"].to(torch.device("cpu"))
