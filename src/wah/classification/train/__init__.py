from ...misc.typing import (
    Config,
    DataLoader,
    Dataset,
    LightningModule,
    Module,
    Optional,
    Path,
)
from ...misc.typing import Trainer as _Trainer
from ..datasets import load_dataloader
from .train import Wrapper, load_trainer

__all__ = [
    "Trainer",
]


class Trainer:
    """Trainer class for classification models.

    ### Args
        - `log_root` (Path): Root directory for saving logs
        - `name` (str): Name of the experiment
        - `version` (Optional[str]): Version of the experiment
        - `**kwargs` (Config): Configuration parameters

    ### Attributes
        - `trainer` (Trainer): Lightning trainer instance
        - `log_dir` (Path): Directory where logs are saved
        - `config` (Config): Configuration parameters

    ### Example
    ```python
    >>> trainer = Trainer("logs", "experiment1", batch_size=32, epochs=100, ...)
    >>> trainer.run(train_dataset, val_dataset, model)
    ```
    """

    def __init__(
        self,
        log_root: Path,
        name: str,
        version: Optional[str] = None,
        **kwargs,
    ) -> None:
        """
        - `log_root` (Path): Root directory for saving logs
        - `name` (str): Name of the experiment
        - `version` (Optional[str]): Version of the experiment
        - `**kwargs` (Config): Configuration parameters
        """
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
        """Run training.

        ### Args
            - `train_dataset` (Dataset): Training dataset
            - `val_dataset` (Dataset): Validation dataset
            - `model` (Module): Model to train
            - `resume` (bool): Whether to resume training from last checkpoint

        ### Example
        ```python
        >>> trainer.run(train_dataset, val_dataset, model)
        ```
        """
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
