from ...typing import (
    Config,
    DataLoader,
    Dataset,
    LightningModule,
    Module,
    Optional,
    Path,
)
from ...typing import Trainer as _Trainer
from ..datasets import load_dataloader
from .train import Wrapper, load_trainer

__all__ = [
    "Trainer",
]


class Trainer:
    def __init__(
        self,
        log_root: Path,
        name: str,
        version: Optional[str] = None,
        **kwargs,
    ) -> None:
        self.trainer: _Trainer = load_trainer(log_root, name, version, **kwargs)
        self.log_dir = self.trainer._log_dir
        self.config: Config = kwargs

    def run(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        model: Module,
        resume: bool = False,
    ) -> None:
        # train_dataset.set_return_w_index()
        # val_dataset.set_return_w_index()
        train_dataloader: DataLoader = load_dataloader(
            train_dataset, train=True, **self.config
        )
        val_dataloader: DataLoader = load_dataloader(
            val_dataset, train=False, **self.config
        )
        model: LightningModule = Wrapper(model, **self.config)

        self.trainer.fit(
            model,
            train_dataloader,
            val_dataloader,
            ckpt_path="last" if resume else None,
        )
