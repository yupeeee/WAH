import lightning as L
import torch
import tqdm
from torch.utils.data import DataLoader, Dataset

from ...misc import lists as _lists
from ...misc import path as _path
from ...misc.lightning import load_accelerator_and_devices
from ...misc.typing import Devices, List, Optional, Path, Tensor, Trainer, Union
from .v1 import SDv1
from .v2 import SDv2

__all__ = [
    "Logger",
]


class Wrapper(L.LightningModule):
    def __init__(
        self,
        pipe: Union[SDv1, SDv2],
        seed: Optional[int] = None,
        t: Optional[int] = 0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pipe = pipe
        self.seed = seed

        self.pipe.init(**kwargs)
        self.pipe.seed(self.seed)
        self.t = t

        self.log_dir: Path

    def on_test_epoch_start(self) -> None:
        self.log_dir = _path.join(
            self.trainer._log_dir,
            self.pipe.config.pipe_id,
        )
        if self.trainer.is_global_zero:
            if (
                _path.exists(_path.join(self.log_dir, "prompts", "run"))
                or _path.exists(_path.join(self.log_dir, "latents", "run"))
                or _path.exists(_path.join(self.log_dir, "noise_preds", "run"))
            ):
                print("Previous run found. Deleting...")
                _path.rmdir(_path.join(self.log_dir, "prompts", "run"))
                _path.rmdir(_path.join(self.log_dir, "latents", "run"))
                _path.rmdir(_path.join(self.log_dir, "noise_preds", "run"))
            _path.mkdir(_path.join(self.log_dir, "prompts", "run"))
            _path.mkdir(_path.join(self.log_dir, "latents", "run"))
            _path.mkdir(_path.join(self.log_dir, "noise_preds", "run"))

    def test_step(self, batch, batch_idx) -> None:
        self.pipe.to(self.local_rank)
        prompt = batch
        latents = self.pipe.make_latents(prompt=prompt)
        with torch.no_grad():
            _ = self.pipe(latents, t=self.t)

        torch.save(
            prompt,
            _path.join(
                self.log_dir,
                "prompts",
                "run",
                f"batch{batch_idx}@{self.local_rank}_{self.seed}.pt",
            ),
        )
        torch.save(
            self.pipe._latents,
            _path.join(
                self.log_dir,
                "latents",
                "run",
                f"batch{batch_idx}@{self.local_rank}_{self.seed}.pt",
            ),
        )
        torch.save(
            self.pipe._noise_preds,
            _path.join(
                self.log_dir,
                "noise_preds",
                "run",
                f"batch{batch_idx}@{self.local_rank}_{self.seed}.pt",
            ),
        )

    def on_test_epoch_end(self) -> None:
        # Wait for all devices to finish test_step
        self.trainer.strategy.barrier()

        if self.trainer.is_global_zero:
            prompt_batch_fpaths = _path.ls(
                _path.join(self.log_dir, "prompts", "run"),
                fext="pt",
                sort=True,
                absolute=True,
            )
            latents_batch_fpaths = _path.ls(
                _path.join(self.log_dir, "latents", "run"),
                fext="pt",
                sort=True,
                absolute=True,
            )
            noise_preds_batch_fpaths = _path.ls(
                _path.join(self.log_dir, "noise_preds", "run"),
                fext="pt",
                sort=True,
                absolute=True,
            )

            prompts = []
            i = 0
            for (
                prompt_batch_fpath,
                latents_batch_fpath,
                noise_preds_batch_fpath,
            ) in tqdm.tqdm(
                zip(
                    prompt_batch_fpaths, latents_batch_fpaths, noise_preds_batch_fpaths
                ),
                total=len(prompt_batch_fpaths),
                desc="Saving latents and noise preds",
            ):
                prompt_batch = torch.load(prompt_batch_fpath)
                latents_batch = torch.load(latents_batch_fpath)
                noise_preds_batch = torch.load(noise_preds_batch_fpath)

                prompts.extend(prompt_batch)
                latents_batch = torch.stack(latents_batch, dim=1)
                noise_preds_batch = torch.stack(noise_preds_batch, dim=1)

                for latents, noise_preds in zip(latents_batch, noise_preds_batch):
                    torch.save(
                        latents,
                        _path.join(self.log_dir, "latents", f"{i}@{self.seed}.pt"),
                    )
                    torch.save(
                        noise_preds,
                        _path.join(self.log_dir, "noise_preds", f"{i}@{self.seed}.pt"),
                    )
                    i += 1

                _path.rmfile(prompt_batch_fpath)
                _path.rmfile(latents_batch_fpath)
                _path.rmfile(noise_preds_batch_fpath)

            _lists.save(prompts, _path.join(self.log_dir, "prompts.txt"))

            assert len(_path.ls(_path.join(self.log_dir, "prompts", "run"))) == 0
            _path.rmdir(_path.join(self.log_dir, "prompts"))
            assert len(_path.ls(_path.join(self.log_dir, "latents", "run"))) == 0
            _path.rmdir(_path.join(self.log_dir, "latents", "run"))
            assert len(_path.ls(_path.join(self.log_dir, "noise_preds", "run"))) == 0
            _path.rmdir(_path.join(self.log_dir, "noise_preds", "run"))


def load_logger(
    log_dir: Path,
    devices: Optional[Devices] = "auto",
) -> Trainer:
    # Load accelerator and devices
    accelerator, devices = load_accelerator_and_devices(devices)
    torch.set_float32_matmul_precision("medium")
    # Load trainer
    trainer = L.Trainer(
        accelerator=accelerator,
        devices=devices,
        logger=False,
        max_epochs=1,
        log_every_n_steps=None,
        deterministic="warn",
    )
    # Set log directory
    setattr(trainer, "_log_dir", log_dir)
    # Return trainer
    return trainer


class PromptDataset(Dataset):
    def __init__(self, prompts: List[str]) -> None:
        self.prompts = prompts

    def __getitem__(self, idx: int) -> str:
        return self.prompts[idx]

    def __len__(self) -> int:
        return len(self.prompts)

    def dataloader(self, **kwargs) -> DataLoader:
        return DataLoader(self, **kwargs)


class Logger:
    def __init__(
        self,
        log_dir: Path,
        pipe: Union[SDv1, SDv2],
        seeds: Optional[List[int]] = None,
        t: Optional[int] = 0,
        devices: Optional[Devices] = "auto",
        **kwargs,
    ) -> None:
        self.log_dir = log_dir
        self.pipe = pipe
        self.seeds = seeds
        self.t = t
        self.devices = devices
        self.config = kwargs

    def __call__(self, prompts: List[str], **kwargs) -> None:
        dataloader = PromptDataset(prompts).dataloader(**kwargs)
        for seed in self.seeds:
            print(f"Logging {self.pipe.config.pipe_id}@SEED={seed}")
            logger = load_logger(
                self.log_dir,
                self.devices,
            )
            logger.test(
                model=Wrapper(self.pipe, seed, self.t, **self.config),
                dataloaders=dataloader,
            )
