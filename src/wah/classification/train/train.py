import lightning as L
import torch
from torchmetrics import Accuracy, MeanMetric

from ... import path as _path
from ...typing import Dict, List, Module, Optional, Path, SummaryWriter, Tensor, Trainer
from ...utils import save_dict_to_csv
from .criterion import load_criterion
from .optimizer import load_optimizer
from .scheduler import load_scheduler
from .utils import (
    check_config,
    get_lr,
    load_accelerator_and_devices,
    load_checkpoint_callback,
    load_tensorboard_logger,
    process_gathered_data,
)

__all__ = [
    "Wrapper",
    "load_trainer",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        **kwargs,
    ) -> None:
        super().__init__()

        check_config(**kwargs)

        self.model = model
        self.config = kwargs
        self.save_hyperparameters(self.config)

        seed = self.config.get("seed", -1)
        if seed is None:
            seed = -1
        if seed >= 0:
            L.seed_everything(seed)

        self.sync_dist: bool = False
        if (
            "gpu" in self.config["devices"]
            and len(self.config["devices"].split("gpu:")[-1].split(",")) > 1
        ):
            self.sync_dist = True

        self.train_criterion = load_criterion(train=True, **self.config)
        self.val_criterion = load_criterion(train=False, **self.config)

        # metrics
        self.lr = MeanMetric()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_acc = Accuracy(
            task="multiclass",
            num_classes=self.config["num_classes"],
        )
        self.val_acc = Accuracy(
            task="multiclass",
            num_classes=self.config["num_classes"],
        )

        # loss/confidence
        self.softmax = torch.nn.Softmax(dim=-1)
        # train
        self._train_idx: List[int] = []
        self._train_gt: List[int] = []
        self._train_pred: List[int] = []
        self._train_loss: List[float] = []
        self._train_conf: List[float] = []
        self._train_gt_conf: List[float] = []
        # val
        self._val_idx: List[int] = []
        self._val_gt: List[int] = []
        self._val_pred: List[int] = []
        self._val_loss: List[float] = []
        self._val_conf: List[float] = []
        self._val_gt_conf: List[float] = []

        # grad_l2
        self.grad_l2_dict: Dict[str, List[Tensor]] = {}
        for i, (layer, _) in enumerate(self.model.named_parameters()):
            self.grad_l2_dict[f"{i}_{layer}"] = []

    def configure_optimizers(self):
        optimizer = load_optimizer(self.model, **self.config)
        scheduler_config = {k: v for k, v in self.config.items() if k != "optimizer"}
        lr_scheduler = load_scheduler(optimizer, **scheduler_config)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
            },
        }

    def training_step(self, batch, batch_idx):
        idxs: Tensor
        data: Tensor
        targets: Tensor

        idxs, (data, targets) = batch

        outputs: Tensor = self.model(data)
        losses: Tensor = self.train_criterion(outputs, targets)

        confs: Tensor = self.softmax(outputs)
        preds: Tensor = torch.argmax(confs, dim=-1)
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)

        results: Tensor = torch.eq(preds, targets)
        signs: Tensor = (results.int() - 0.5).sign()
        signed_confs: Tensor = confs[:, preds].diag() * signs
        signed_gt_confs: Tensor = confs[:, targets].diag() * signs

        self._train_idx.append(idxs.cpu())
        self._train_gt.append(targets.cpu())
        self._train_pred.append(preds.cpu())
        self._train_loss.append(losses.cpu())
        self._train_conf.append(signed_confs.cpu())
        self._train_gt_conf.append(signed_gt_confs.cpu())

        loss = losses.mean()

        self.train_loss(loss)
        self.train_acc(outputs, targets)

        return loss

    def on_after_backward(self):
        # track: grad_l2
        for i, (layer, param) in enumerate(self.model.named_parameters()):
            grad_l2 = torch.norm(param.grad.flatten(), p=2)
            self.grad_l2_dict[f"{i}_{layer}"].append(grad_l2.view(1))

    def on_train_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1

        # epoch as global_step
        self.log("step", current_epoch, sync_dist=self.sync_dist)

        # log: lr
        self.lr(get_lr(self.trainer))
        self.log(f"lr-{self.config['optimizer']}", self.lr, sync_dist=self.sync_dist)

        # log: loss, acc@1
        self.log("train/avg_loss", self.train_loss, sync_dist=self.sync_dist)
        self.log("train/acc@1", self.train_acc, sync_dist=self.sync_dist)

        # log: eval
        idx = process_gathered_data(self.all_gather(self._train_idx), 2, 1, (1, 0))
        gt = process_gathered_data(self.all_gather(self._train_gt), 2, 1, (1, 0))
        pred = process_gathered_data(self.all_gather(self._train_pred), 2, 1, (1, 0))
        loss = process_gathered_data(self.all_gather(self._train_loss), 2, 1, (1, 0))
        conf = process_gathered_data(self.all_gather(self._train_conf), 2, 1, (1, 0))
        gt_conf = process_gathered_data(
            self.all_gather(self._train_gt_conf), 2, 1, (1, 0)
        )
        save_dict_to_csv(
            dictionary={
                "idx": [int(i) for i in idx],
                "gt": [int(g) for g in gt],
                "pred": [int(p) for p in pred],
                "loss": [float(l) for l in loss],
                "conf": [float(c) for c in conf],
                "gt_conf": [float(gc) for gc in gt_conf],
            },
            save_dir=_path.join(self.trainer._log_dir, "eval/train"),
            save_name=f"epoch={current_epoch}",
            index_col="idx",
        )
        self._train_idx.clear()
        self._train_gt.clear()
        self._train_pred.clear()
        self._train_loss.clear()
        self._train_conf.clear()
        self._train_gt_conf.clear()

        # log: grad_l2
        tensorboard: SummaryWriter = self.logger.experiment
        tag = self.trainer.logger.name
        for layer, grad_l2 in self.grad_l2_dict.items():
            grad_l2 = torch.cat(grad_l2)
            tensorboard.add_histogram(
                tag=f"grad_l2/{tag}/{layer}",
                values=grad_l2,
                global_step=current_epoch,
            )
            self.grad_l2_dict[layer].clear()  # reset

        # log: params
        for i, (layer, param) in enumerate(self.model.named_parameters()):
            tensorboard.add_histogram(
                tag=f"params/{tag}/{i}_{layer}",
                values=param.flatten(),
                global_step=current_epoch,
            )

        # save checkpoint
        if "save_per_epoch" in self.config.keys():
            if (current_epoch + 1) % self.config["save_per_epoch"] == 0:
                ckpt_dir = _path.join(self.trainer._log_dir, "checkpoints")
                _path.mkdir(ckpt_dir)
                torch.save(
                    self.model.state_dict(),
                    _path.join(ckpt_dir, f"epoch={current_epoch}.ckpt"),
                )

    def validation_step(self, batch, batch_idx):
        idxs: Tensor
        data: Tensor
        targets: Tensor

        idxs, (data, targets) = batch

        outputs: Tensor = self.model(data)
        losses: Tensor = self.val_criterion(outputs, targets)

        confs: Tensor = self.softmax(outputs)
        preds: Tensor = torch.argmax(confs, dim=-1)
        if targets.dim() == 2:
            targets = targets.argmax(dim=1)

        results: Tensor = torch.eq(preds, targets)
        signs: Tensor = (results.int() - 0.5).sign()
        signed_confs: Tensor = confs[:, preds].diag() * signs
        signed_gt_confs: Tensor = confs[:, targets].diag() * signs

        self._val_idx.append(idxs.cpu())
        self._val_gt.append(targets.cpu())
        self._val_pred.append(preds.cpu())
        self._val_loss.append(losses.cpu())
        self._val_conf.append(signed_confs.cpu())
        self._val_gt_conf.append(signed_gt_confs.cpu())

        loss = losses.mean()

        self.val_loss(loss)
        self.val_acc(outputs, targets)

    def on_validation_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1

        # epoch as global_step
        self.log("step", current_epoch, sync_dist=self.sync_dist)

        # log: loss, acc@1
        self.log("val/avg_loss", self.val_loss, sync_dist=self.sync_dist)
        self.log("val/acc@1", self.val_acc, sync_dist=self.sync_dist)

        # log: eval
        idx = process_gathered_data(self.all_gather(self._val_idx), 2, 1, (1, 0))
        gt = process_gathered_data(self.all_gather(self._val_gt), 2, 1, (1, 0))
        pred = process_gathered_data(self.all_gather(self._val_pred), 2, 1, (1, 0))
        loss = process_gathered_data(self.all_gather(self._val_loss), 2, 1, (1, 0))
        conf = process_gathered_data(self.all_gather(self._val_conf), 2, 1, (1, 0))
        gt_conf = process_gathered_data(
            self.all_gather(self._val_gt_conf), 2, 1, (1, 0)
        )
        save_dict_to_csv(
            dictionary={
                "idx": [int(i) for i in idx],
                "gt": [int(g) for g in gt],
                "pred": [int(p) for p in pred],
                "loss": [float(l) for l in loss],
                "conf": [float(c) for c in conf],
                "gt_conf": [float(gc) for gc in gt_conf],
            },
            save_dir=_path.join(self.trainer._log_dir, "eval/train"),
            save_name=f"epoch={current_epoch}",
            index_col="idx",
        )
        self._val_idx.clear()
        self._val_gt.clear()
        self._val_pred.clear()
        self._val_loss.clear()
        self._val_conf.clear()
        self._val_gt_conf.clear()


def load_trainer(
    save_dir: Path,
    name: str,
    version: Optional[str] = None,
    **kwargs,
) -> Trainer:
    config = kwargs

    accelerator, devices = load_accelerator_and_devices(config["devices"])

    tensorboard_logger = load_tensorboard_logger(
        save_dir=save_dir,
        name=name,
        version=version,
    )
    _log_dir = tensorboard_logger.log_dir
    checkpoint_callback = load_checkpoint_callback()

    if config["amp"] is True:
        torch.set_float32_matmul_precision("medium")

    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        precision="16-mixed" if config["amp"] is True else "32-true",
        logger=[
            tensorboard_logger,
        ],
        callbacks=[
            checkpoint_callback,
        ],
        max_epochs=config["epochs"],
        log_every_n_steps=None,
        gradient_clip_val=(
            config["gradient_clip_val"]
            if "gradient_clip_val" in config.keys()
            else None
        ),
        deterministic="warn" if config["seed"] is not None else False,
    )
    setattr(trainer, "_log_dir", _log_dir)

    return trainer
