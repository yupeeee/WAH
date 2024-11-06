import lightning as L
import torch

from ... import utils
from ...module import get_attrs
from ...typing import (
    Config,
    Dataset,
    Devices,
    Dict,
    List,
    Module,
    Optional,
    Sequence,
    Tensor,
)
from ..datasets import to_dataloader
from ..models.feature_extraction import FeatureExtractor
from ..train.utils import load_accelerator_and_devices
from .utils import process_gathered_data

__all__ = [
    "MeanVarTest",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        model: Module,
        config: Config,
        res_dict: Dict[str, List],
        bn_dim: Sequence[int],
        ln_dim: Sequence[int],
    ) -> None:
        super().__init__()

        self.model = model
        self.config = config
        self.res_dict = res_dict
        self.bn_dim = bn_dim
        self.ln_dim = ln_dim

        self.attrs = self.model.attrs

        for attr in self.model.attrs:
            setattr(self, f"{attr}_bn_mean", [])
            setattr(self, f"{attr}_bn_var", [])
            setattr(self, f"{attr}_ln_mean", [])
            setattr(self, f"{attr}_ln_var", [])

    def test_step(self, batch, batch_idx):
        data: Tensor

        data = batch

        inputs: Dict[str, Tensor] = self.model(data)

        for attr, v in inputs.keys():
            if isinstance(v, tuple):
                assert len(v) == 1
                v = v[0]

            getattr(self, f"{attr}_bn_mean").append(
                torch.mean(v, dim=self.bn_dim).cpu()
            )
            getattr(self, f"{attr}_bn_var").append(
                torch.var(v, dim=self.bn_dim, unbiased=False).cpu()
            )
            getattr(self, f"{attr}_ln_mean").append(
                torch.mean(v, dim=self.ln_dim).cpu()
            )
            getattr(self, f"{attr}_ln_var").append(
                torch.var(v, dim=self.ln_dim, unbiased=False).cpu()
            )

    def on_test_epoch_end(self) -> None:
        for attr in self.attrs:
            self.res_dict[f"{attr}_bn_mean"] = [
                float(x) # TODO: it is not float! it is a vector!
                for x in process_gathered_data(
                    data=self.all_gather(getattr(self, f"{attr}_bn_mean")),
                    unsqueeze_until=2,
                    cat_along=1,
                    permute_dims=(1, 0),
                )
            ]
            self.res_dict[f"{attr}_bn_var"] = [
                float(x)
                for x in process_gathered_data(
                    data=self.all_gather(getattr(self, f"{attr}_bn_var")),
                    unsqueeze_until=2,
                    cat_along=1,
                    permute_dims=(1, 0),
                )
            ]
            self.res_dict[f"{attr}_ln_mean"] = [
                float(x)
                for x in process_gathered_data(
                    data=self.all_gather(getattr(self, f"{attr}_ln_mean")),
                    unsqueeze_until=2,
                    cat_along=1,
                    permute_dims=(1, 0),
                )
            ]
            self.res_dict[f"{attr}_ln_var"] = [
                float(x)
                for x in process_gathered_data(
                    data=self.all_gather(getattr(self, f"{attr}_ln_var")),
                    unsqueeze_until=2,
                    cat_along=1,
                    permute_dims=(1, 0),
                )
            ]


class MeanVarTest:
    def __init__(
        self,
        batch_size: int,
        num_workers: int,
        mixup_alpha: Optional[float] = 0.0,
        cutmix_alpha: Optional[float] = 0.0,
        label_smoothing: Optional[float] = 0.0,
        seed: Optional[int] = None,
        devices: Optional[Devices] = "auto",
        amp: Optional[bool] = False,
        verbose: Optional[bool] = True,
    ) -> None:
        """
        - `batch_size` (int): The batch size for the test.
        - `num_workers` (int): The number of workers for the DataLoader.
        - `mixup_alpha` (Optional[float], optional): Alpha value for Mixup data augmentation. Defaults to `0.0`.
        - `cutmix_alpha` (Optional[float], optional): Alpha value for CutMix data augmentation. Defaults to `0.0`.
        - `label_smoothing` (Optional[float], optional): The amount of label smoothing to apply to the loss. Defaults to `0.0`.
        - `seed` (Optional[int], optional): Random seed for deterministic behavior. Defaults to `None`.
        - `devices` (Optional[Devices], optional): The devices to run the test on. Defaults to `"auto"`.
        - `amp` (Optional[bool]): Whether to use automatic mixed precision (AMP) for model evaluation. Defaults to `False`.
        - `verbose` (Optional[bool], optional): Whether to show progress bar and model summary during testing. Defaults to `True`.
        """
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
        self.label_smoothing = label_smoothing
        self.seed = seed
        self.devices = devices
        self.amp = amp
        self.verbose = verbose

        utils.seed(self.seed)
        accelerator, devices = load_accelerator_and_devices(self.devices)

        if not verbose:
            utils.disable_lightning_logging()

        self.runner = L.Trainer(
            accelerator=accelerator,
            devices=devices,
            precision="16-mixed" if self.amp else "32-true",
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
            "mixup_alpha": self.mixup_alpha,
            "cutmix_alpha": self.cutmix_alpha,
            "label_smoothing": self.label_smoothing,
        }

    def __call__(
        self,
        model: Module,
        norm: str,
        dataset: Dataset,
        bn_dim: Sequence[int],
        ln_dim: Sequence[int],
    ) -> Dict[str, List[float]]:
        res_dict = {}
        dataset.set_return_data_only()
        input_extractor = FeatureExtractor(
            model=model,
            attrs=get_attrs(model, specify=norm),
            return_inputs=True,
        )
        model = Wrapper(input_extractor, self.config, res_dict, bn_dim, ln_dim)
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
