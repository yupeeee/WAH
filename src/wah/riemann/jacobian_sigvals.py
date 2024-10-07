import lightning as L
import torch
from torch.func import jacrev

from .. import utils
from ..classification.datasets import to_dataloader
from ..classification.train.utils import load_accelerator_and_devices
from ..typing import Dataset, Devices, Dict, List, Module, Optional, Tensor

__all__ = [
    "JacobianSigVals",
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

        self.num_data = []
        self.sigval = []

    def test_step(self, batch: Tensor, batch_idx):
        batch_size = len(batch)
        input_dim = batch[0].numel()

        # enable gradients
        with torch.inference_mode(False):
            batch = batch.clone().detach().requires_grad_(True)
            J: Tensor = jacrev(self.model)(batch)

        assert J.requires_grad
        J = J.detach().reshape(batch_size, -1, input_dim)

        sigvals: Tensor = torch.linalg.svdvals(J)

        self.num_data.append(batch_size)
        self.sigval.append(sigvals.cpu())

    def on_test_epoch_end(self) -> None:
        num_data: List[int] = self.all_gather(self.num_data)
        sigval: List[Tensor] = self.all_gather(self.sigval)

        num_data = sum(num_data)
        sigval = torch.cat(sigval, dim=1).permute(1, 0, -1)
        print(num_data)
        print(sigval.shape)

        # idx = torch.cat(idx, dim=-1).flatten()
        # idx, indices = torch.sort(idx)
        # gt = torch.cat(gt, dim=-1).flatten()[indices]
        # pred = torch.cat(pred, dim=-1).flatten()[indices]
        # loss = torch.cat(loss, dim=-1).flatten()[indices]
        # conf = torch.cat(conf, dim=-1).flatten()[indices]
        # gt_conf = torch.cat(gt_conf, dim=-1).flatten()[indices]

        # self.res_dict["idx"] = [int(i) for i in idx]


class JacobianSigVals:
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
    ) -> Tensor:
        res_dict = {}
        model = Wrapper(model, res_dict)
        dataloader = to_dataloader(
            dataset=dataset,
            train=False,
            **self.config,
        )

        self.runner.test(
            model=model,
            dataloaders=dataloader,
        )

        return torch.cat(res_dict["sigval"], dim=0)
