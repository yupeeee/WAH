import lightning as L
import torch
from torchmetrics import Accuracy, MeanMetric

from ... import path as _path
from ...typing import (
    Dict,
    List,
    Module,
    Optional,
    Path,
    SummaryWriter,
    Tensor,
    Trainer,
    Union,
)
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
        self.train_eval_dict: Dict[str, List[Union[int, float]]] = {
            "idx": [],
            "gt": [],
            "pred": [],
            "loss": [],
            "conf": [],
            "gt_conf": [],
        }
        self.val_eval_dict: Dict[str, List[Union[int, float]]] = {
            "idx": [],
            "gt": [],
            "pred": [],
            "loss": [],
            "conf": [],
            "gt_conf": [],
        }
        _path.mkdir(_path.join(self.trainer._log_dir, "eval/train"))
        _path.mkdir(_path.join(self.trainer._log_dir, "eval/val"))

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

        self.train_eval_dict["idx"].append(idxs.cpu())
        self.train_eval_dict["gt"].append(targets.cpu())
        self.train_eval_dict["pred"].append(preds.cpu())
        self.train_eval_dict["loss"].append(losses.cpu())
        self.train_eval_dict["conf"].append(signed_confs.cpu())
        self.train_eval_dict["gt_conf"].append(signed_gt_confs.cpu())

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
        save_dict_to_csv(
            self.train_eval_dict,
            save_dir=_path.join(self.trainer._log_dir, "eval/train"),
            save_name=f"epoch={current_epoch}",
            index_col="idx",
        )

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

        self.val_eval_dict["idx"].append(idxs.cpu())
        self.val_eval_dict["gt"].append(targets.cpu())
        self.val_eval_dict["pred"].append(preds.cpu())
        self.val_eval_dict["loss"].append(losses.cpu())
        self.val_eval_dict["conf"].append(signed_confs.cpu())
        self.val_eval_dict["gt_conf"].append(signed_gt_confs.cpu())

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
        save_dict_to_csv(
            self.val_eval_dict,
            save_dir=_path.join(self.trainer._log_dir, "eval/val"),
            save_name=f"epoch={current_epoch}",
            index_col="idx",
        )


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
