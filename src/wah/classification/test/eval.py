import lightning as L
import torch

from ...misc.typing import Dict, List, Module, Tensor, Tuple
from ..train.criterion import load_criterion


class EvalTest(L.LightningModule):
    def __init__(self, model: Module, **kwargs):
        super().__init__()
        self.model = model
        self.config = kwargs
        self.criterion = load_criterion(train=False, reduction="none", **self.config)
        # Store results
        self.gt: List[Tensor] = []
        self.pred: List[Tensor] = []
        self.conf: List[Tensor] = []
        self.gt_conf: List[Tensor] = []
        self.loss: List[Tensor] = []
        self.l2: List[Tensor] = []

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        # Get predictions and confidences
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        # Compute confidence scores (positive if correct, negative if wrong)
        confs = probs[torch.arange(len(y)), preds]
        confs = torch.where(preds == y, confs, -confs)
        gt_confs = probs[torch.arange(len(y)), y]
        gt_confs = torch.where(preds == y, gt_confs, -gt_confs)
        # Compute loss and L2 norm
        loss = self.criterion(logits, y)
        l2_norm = torch.norm(logits, p=2, dim=1)
        # Store results
        self.gt.append(y.cpu())
        self.pred.append(preds.cpu())
        self.conf.append(confs.cpu())
        self.gt_conf.append(gt_confs.cpu())
        self.loss.append(loss.cpu())
        self.l2.append(l2_norm.cpu())

    def on_test_epoch_end(self) -> Dict[str, Tensor]:
        results = {
            "gt": torch.cat(self.gt).tolist(),
            "pred": torch.cat(self.pred).tolist(),
            "conf": torch.cat(self.conf).tolist(),
            "gt_conf": torch.cat(self.gt_conf).tolist(),
            "loss": torch.cat(self.loss).tolist(),
            "l2": torch.cat(self.l2).tolist(),
        }
        return results
