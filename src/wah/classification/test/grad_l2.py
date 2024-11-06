import lightning as L
import torch

from ...module import _getattr, get_attrs
from .utils import process_gathered_data
from ...typing import (
    Config,
    Dict,
    List,
    Module,
    Optional,
    Path,
    SummaryWriter,
    Tensor,
    Trainer,
    Tuple,
)

__all__ = [
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

        self.criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=config["label_smoothing"],
        )

        self.attrs = [attr for attr, _ in self.model.named_parameters()]

        self.idx = []
        for attr in self.attrs:
            setattr(self, attr, [])
            setattr(self, attr, [])

    def configure_optimizers(self):
        return None

    def training_step(self, batch, batch_idx):
        indices: Tensor
        data: Tensor
        targets: Tensor

        indices, (data, targets) = batch

        outputs: Tensor = self.model(data)

        loss: Tensor = self.criterion(outputs, targets)

        self.idx.append(indices.cpu())

        return loss

    def on_after_backward(self) -> None:
        for layer, param in self.model.named_parameters():
            grad_l2 = torch.norm(param.grad.flatten(), p=2)
            getattr(self, layer).append(grad_l2.view(1))

    def on_train_epoch_end(self) -> None:
        idx: List[Tensor] = self.all_gather(self.idx)
        for attr in self.attrs:
            setattr(self, attr, [])
            setattr(self, attr, [])
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
