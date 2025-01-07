import lightning as L
import torch
from torchmetrics import Accuracy, MeanMetric

from ... import path as _path
from ...typing import Dict, List, Module, Optional, Path, Tensor, Trainer, Union
from ...utils.dictionary import save_dict_to_csv
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

        self.train_criterion = load_criterion(
            train=True,
            reduction="none",
            **self.config,
        )
        self.val_criterion = load_criterion(
            train=False,
            reduction="none",
            **self.config,
        )
        self.softmax = torch.nn.Softmax(dim=-1)

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

        # # eval
        # self.train_res_dict: Dict[str, Union[int, float]] = {
        #     "gt": [],
        #     "pred": [],
        #     "loss": [],
        #     "conf": [],
        #     "gt_conf": [],
        #     "l2": [],
        # }
        # self.val_res_dict: Dict[str, Union[int, float]] = {
        #     "gt": [],
        #     "pred": [],
        #     "loss": [],
        #     "conf": [],
        #     "gt_conf": [],
        #     "l2": [],
        # }

        # grad_l2
        self.grad_l2_dict: Dict[str, List[float]] = {}
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

    def on_train_epoch_start(self):
        if self.trainer.is_global_zero:
            # eval
            # self.train_res_dict: Dict[str, Union[int, float]] = {
            #     "gt": [],
            #     "pred": [],
            #     "loss": [],
            #     "conf": [],
            #     "gt_conf": [],
            #     "l2": [],
            # }
            save_dict_to_csv(
                dictionary={
                    key: []
                    for key in [
                        # "idx",
                        "gt",
                        "pred",
                        "loss",
                        "conf",
                        "gt_conf",
                        "l2",
                    ]
                },
                save_dir=_path.join(self.trainer._log_dir, "eval/train"),
                save_name=f"epoch={self.current_epoch + 1}",
                # index_col="idx",
                mode="w",
                write_header=True,
                filelock=False,
            )

            # grad_l2
            for i, (layer, _) in enumerate(self.model.named_parameters()):
                self.grad_l2_dict[f"{i}_{layer}"] = []

    def training_step(self, batch, batch_idx):
        # idxs: Tensor
        data: Tensor
        targets: Tensor

        # idxs, (data, targets) = batch
        data, targets = batch

        outputs: Tensor = self.model(data)
        losses: Tensor = self.train_criterion(outputs, targets)

        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
        self.train_loss(losses)
        self.train_acc(outputs, targets)

        confs: Tensor = self.softmax(outputs)
        preds: Tensor = torch.argmax(confs, dim=-1)
        results: Tensor = torch.eq(preds, targets)
        signs: Tensor = (results.int() - 0.5).sign()
        signed_confs: Tensor = confs[:, preds].diag() * signs
        signed_gt_confs: Tensor = confs[:, targets].diag() * signs
        l2_norms: Tensor = outputs.norm(p=2, dim=-1)

        if self.sync_dist:
            # idxs_g = self.all_gather(idxs)
            losses_g = self.all_gather(losses)
            preds_g = self.all_gather(preds)
            targets_g = self.all_gather(targets)
            signed_confs_g = self.all_gather(signed_confs)
            signed_gt_confs_g = self.all_gather(signed_gt_confs)
            l2_norms_g = self.all_gather(l2_norms)
        else:
            # idxs_g = idxs
            losses_g = losses
            preds_g = preds
            targets_g = targets
            signed_confs_g = signed_confs
            signed_gt_confs_g = signed_gt_confs
            l2_norms_g = l2_norms

        if self.trainer.is_global_zero:
            # idxs_g = idxs_g.view(-1)
            losses_g = losses_g.view(-1)
            preds_g = preds_g.view(-1)
            targets_g = targets_g.view(-1)
            signed_confs_g = signed_confs_g.view(-1)
            signed_gt_confs_g = signed_gt_confs_g.view(-1)
            l2_norms_g = l2_norms_g.view(-1)

            res_dict = {
                # "idx": [int(x) for x in idxs_g],
                "gt": [int(x) for x in targets_g],
                "pred": [int(x) for x in preds_g],
                "loss": [float(x) for x in losses_g],
                "conf": [float(x) for x in signed_confs_g],
                "gt_conf": [float(x) for x in signed_gt_confs_g],
                "l2": [float(x) for x in l2_norms_g],
            }
            # self.train_res_dict = {
            #     key: self.train_res_dict.get(key, []) + res_dict.get(key, [])
            #     for key in self.train_res_dict.keys() | res_dict.keys()
            # }
            save_dict_to_csv(
                dictionary=res_dict,
                save_dir=_path.join(self.trainer._log_dir, "eval/train"),
                save_name=f"epoch={self.current_epoch + 1}",
                # index_col="idx",
                mode="a",
                write_header=False,
                filelock=False,
            )

        return losses.mean()

    def on_after_backward(self):
        # track: grad_l2
        for i, (layer, param) in enumerate(self.model.named_parameters()):
            grad_l2 = torch.norm(param.grad.flatten(), p=2)
            self.grad_l2_dict[f"{i}_{layer}"] += [float(l) for l in grad_l2.view(1)]

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

        if self.trainer.is_global_zero:
            # # log: eval
            # save_dict_to_csv(
            #     dictionary=self.train_res_dict,
            #     save_dir=_path.join(self.trainer._log_dir, "eval/train"),
            #     save_name=f"epoch={self.current_epoch + 1}",
            #     # index_col="idx",
            #     mode="w",
            #     write_header=True,
            #     filelock=False,
            # )

            # log: grad_l2
            save_dict_to_csv(
                dictionary=self.grad_l2_dict,
                save_dir=_path.join(self.trainer._log_dir, "grad_l2"),
                save_name=f"epoch={self.current_epoch + 1}",
                mode="w",
                write_header=True,
                filelock=False,
            )
            self.grad_l2_dict = {}

            # save checkpoint
            if "save_per_epoch" in self.config.keys():
                if (current_epoch + 1) % self.config["save_per_epoch"] == 0:
                    ckpt_dir = _path.join(self.trainer._log_dir, "checkpoints")
                    _path.mkdir(ckpt_dir)
                    torch.save(
                        self.model.state_dict(),
                        _path.join(ckpt_dir, f"epoch={current_epoch}.ckpt"),
                    )

    def on_validation_epoch_start(self):
        if self.trainer.is_global_zero:
            # eval
            # self.val_res_dict: Dict[str, Union[int, float]] = {
            #     "gt": [],
            #     "pred": [],
            #     "loss": [],
            #     "conf": [],
            #     "gt_conf": [],
            #     "l2": [],
            # }
            save_dict_to_csv(
                dictionary={
                    key: []
                    for key in [
                        # "idx",
                        "gt",
                        "pred",
                        "loss",
                        "conf",
                        "gt_conf",
                        "l2",
                    ]
                },
                save_dir=_path.join(self.trainer._log_dir, "eval/val"),
                save_name=f"epoch={self.current_epoch + 1}",
                # index_col="idx",
                mode="w",
                write_header=True,
                filelock=False,
            )

    def validation_step(self, batch, batch_idx):
        # idxs: Tensor
        data: Tensor
        targets: Tensor

        # idxs, (data, targets) = batch
        data, targets = batch

        outputs: Tensor = self.model(data)
        losses: Tensor = self.val_criterion(outputs, targets)

        if targets.dim() == 2:
            targets = targets.argmax(dim=1)
        self.val_loss(losses)
        self.val_acc(outputs, targets)

        confs: Tensor = self.softmax(outputs)
        preds: Tensor = torch.argmax(confs, dim=-1)
        results: Tensor = torch.eq(preds, targets)
        signs: Tensor = (results.int() - 0.5).sign()
        signed_confs: Tensor = confs[:, preds].diag() * signs
        signed_gt_confs: Tensor = confs[:, targets].diag() * signs
        l2_norms: Tensor = outputs.norm(p=2, dim=-1)

        if self.sync_dist:
            # idxs_g = self.all_gather(idxs)
            losses_g = self.all_gather(losses)
            preds_g = self.all_gather(preds)
            targets_g = self.all_gather(targets)
            signed_confs_g = self.all_gather(signed_confs)
            signed_gt_confs_g = self.all_gather(signed_gt_confs)
            l2_norms_g = self.all_gather(l2_norms)
        else:
            # idxs_g = idxs
            losses_g = losses
            preds_g = preds
            targets_g = targets
            signed_confs_g = signed_confs
            signed_gt_confs_g = signed_gt_confs
            l2_norms_g = l2_norms

        if self.trainer.is_global_zero:
            # idxs_g = idxs_g.view(-1)
            losses_g = losses_g.view(-1)
            preds_g = preds_g.view(-1)
            targets_g = targets_g.view(-1)
            signed_confs_g = signed_confs_g.view(-1)
            signed_gt_confs_g = signed_gt_confs_g.view(-1)
            l2_norms_g = l2_norms_g.view(-1)

            res_dict = {
                # "idx": [int(x) for x in idxs_g],
                "gt": [int(x) for x in targets_g],
                "pred": [int(x) for x in preds_g],
                "loss": [float(x) for x in losses_g],
                "conf": [float(x) for x in signed_confs_g],
                "gt_conf": [float(x) for x in signed_gt_confs_g],
                "l2": [float(x) for x in l2_norms_g],
            }
            # self.val_res_dict = {
            #     key: self.val_res_dict.get(key, []) + res_dict.get(key, [])
            #     for key in self.val_res_dict.keys() | res_dict.keys()
            # }
            save_dict_to_csv(
                dictionary=res_dict,
                save_dir=_path.join(self.trainer._log_dir, "eval/val"),
                save_name=f"epoch={self.current_epoch + 1}",
                # index_col="idx",
                mode="a",
                write_header=False,
                filelock=False,
            )

    def on_validation_epoch_end(self) -> None:
        current_epoch = self.current_epoch + 1

        # epoch as global_step
        self.log("step", current_epoch, sync_dist=self.sync_dist)

        # log: loss, acc@1
        self.log("val/avg_loss", self.val_loss, sync_dist=self.sync_dist)
        self.log("val/acc@1", self.val_acc, sync_dist=self.sync_dist)

        # if self.trainer.is_global_zero:
        #     # log: eval
        #     save_dict_to_csv(
        #         dictionary=self.val_res_dict,
        #         save_dir=_path.join(self.trainer._log_dir, "eval/val"),
        #         save_name=f"epoch={self.current_epoch + 1}",
        #         # index_col="idx",
        #         mode="w",
        #         write_header=True,
        #         filelock=False,
        #     )


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
