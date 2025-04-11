import lightning as L
import torch
import tqdm
from lightning.pytorch.strategies import DDPStrategy
from torch.utils.data import DataLoader, Dataset

from ...misc import lists as _lists
from ...misc import path as _path
from ...misc.lightning import load_accelerator_and_devices
from ...misc.typing import Devices, List, Optional, Path, Tensor, Trainer, Union
from .pipe import StableDiffusion

__all__ = [
    "Logger",
]


def rearrange(
    logs: List[Union[List[str], List[Tensor]]],
) -> List[Union[List[str], List[Tensor]]]:
    logs = list(map(list, zip(*logs)))
    logs = [log for batch in logs for log in batch]
    return logs


class Wrapper(L.LightningModule):
    def __init__(
        self,
        pipe: StableDiffusion,
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.pipe = pipe
        self.seed = seed

        self.pipe.init(**kwargs)
        self.pipe.seed(self.seed)

        self.log_dir: Path

    def on_test_epoch_start(self) -> None:
        self.log_dir = _path.join(
            self.trainer._log_dir,
            self.pipe.config.pipe_id,
        )
        torch.save(self.pipe.config, _path.join(self.log_dir, "config.pt"))
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
            # Group files by batch index
            batch_indices = _lists.sort(
                list(
                    set(
                        [
                            _path.basename(fpath).split("@")[0].replace("batch", "")
                            for fpath in prompt_batch_fpaths
                        ]
                    )
                )
            )
            prompt_batch_fpaths = [
                [fpath for fpath in prompt_batch_fpaths if f"batch{batch_idx}" in fpath]
                for batch_idx in batch_indices
            ]
            latents_batch_fpaths = [
                [
                    fpath
                    for fpath in latents_batch_fpaths
                    if f"batch{batch_idx}" in fpath
                ]
                for batch_idx in batch_indices
            ]
            noise_preds_batch_fpaths = [
                [
                    fpath
                    for fpath in noise_preds_batch_fpaths
                    if f"batch{batch_idx}" in fpath
                ]
                for batch_idx in batch_indices
            ]

            prompts = []
            i = 0
            for (
                prompt_fpaths,
                latents_fpaths,
                noise_preds_fpaths,
            ) in tqdm.tqdm(
                zip(
                    prompt_batch_fpaths,
                    latents_batch_fpaths,
                    noise_preds_batch_fpaths,
                ),
                total=len(prompt_batch_fpaths),
                desc="Saving latents and noise preds",
            ):
                prompts_batch = [
                    torch.load(fpath, weights_only=True) for fpath in prompt_fpaths
                ]
                prompts.extend(rearrange(prompts_batch))

                # (world_size, batch_size, T, *feature_dim)
                latents_batch = [
                    torch.stack(torch.load(fpath, weights_only=True), dim=1).unbind(
                        dim=0
                    )
                    for fpath in latents_fpaths
                ]
                noise_preds_batch = [
                    torch.stack(torch.load(fpath, weights_only=True), dim=1).unbind(
                        dim=0
                    )
                    for fpath in noise_preds_fpaths
                ]
                # rearrange to original order
                latents_batch = rearrange(latents_batch)
                noise_preds_batch = rearrange(noise_preds_batch)[
                    1::2
                ]  # remove noise_pred_uncond (use noise_pred_text only)

                for latent, noise_pred in zip(latents_batch, noise_preds_batch):
                    torch.save(
                        latent,
                        _path.join(self.log_dir, "latents", f"{i}_{self.seed}.pt"),
                    )
                    torch.save(
                        noise_pred,
                        _path.join(self.log_dir, "noise_preds", f"{i}_{self.seed}.pt"),
                    )
                    i += 1

                for (
                    prompt_fpath,
                    latents_fpath,
                    noise_preds_fpath,
                ) in zip(
                    prompt_fpaths,
                    latents_fpaths,
                    noise_preds_fpaths,
                ):
                    _path.rmfile(prompt_fpath)
                    _path.rmfile(latents_fpath)
                    _path.rmfile(noise_preds_fpath)

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
        strategy=DDPStrategy(replace_sampler_ddp=False),
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
        return DataLoader(
            self,
            shuffle=False,
            **kwargs,
        )


class Logger:
    def __init__(
        self,
        log_dir: Path,
        pipe: StableDiffusion,
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
