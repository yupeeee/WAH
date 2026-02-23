import os
import socket
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, Dataset, DistributedSampler

__all__ = [
    "Env",
    "DDP",
    "make_dataloader",
]


def _parse_devices(devices: Union[str, List[int], int] = "auto") -> List[int]:
    """Parse a device specification into a list of GPU indices.

    ### Args
        - `devices` (str, int, or list of ints): Device specification.
            * Examples:
                - "cpu", "cuda", "gpu", "auto"
                - "cuda:0,1", "gpu:2,3,4"
                - "0,1,2"
                - int (e.g., 0)
                - List[int] (e.g., [0, 1, 2])

    ### Returns
        - `List[int]`: List of GPU indices (empty if using CPU).

    ### Example
    ```python
    >>> parse_devices("cpu")
    []
    >>> parse_devices("cuda:0,1")
    [0, 1]
    >>> parse_devices([0, 1, 2])
    [0, 1, 2]
    >>> parse_devices("auto")
    # Returns all available GPU indices or [] if no GPU present
    ```
    """
    if devices == "auto":
        return (
            list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        )
    if isinstance(devices, str):
        s = devices.lower().replace(" ", "")
        if s == "cpu":
            return []
        if s == "cuda" or s == "gpu":
            assert torch.cuda.is_available(), "CUDA is not available"
            # Use only the first visible gpu
            return [0]
        # Parse things like "cuda:0,1,2" or "gpu:1,2"
        if s.startswith("cuda:") or s.startswith("gpu:"):
            idxs = s.split(":")[1]
            return [int(i) for i in idxs.split(",") if i.strip() != ""]
        # Parse comma separated digits, like "0,1,2"
        return [int(i) for i in s.split(",") if i.strip() != ""]
    if isinstance(devices, int):
        return [devices]
    if isinstance(devices, (list, tuple)):
        return [int(i) for i in devices]
    raise ValueError(f"Invalid device specification: {devices}")


def _find_free_port(master_addr: str) -> int:
    """Find a free TCP port on `master_addr`."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((master_addr, 0))
        return int(s.getsockname()[1])


@dataclass(frozen=True)
class Env:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    device_idx: Optional[int]  # None for CPU


class DDP:
    """Distributed/single-process runner that invokes a worker with an `Env`.

    Configures devices (GPUs or CPU), backend (e.g. NCCL/Gloo), and either
    spawns one process per device or runs the worker once. Use `run()` to
    execute a callable that receives the process `Env` and your args/kwargs.

    ### Example
    ```python
    >>> def worker(rank: int, world_size: int, lr: float):
    ...     model = Model().to(torch.device(f"cuda:{rank}"))
    ...     ...
    >>> ddp = DDP("cuda:0,1")
    >>> ddp.run(worker, lr=1e-3)
    ```
    """

    def __init__(
        self,
        devices: Union[str, List[int], int] = "auto",
        backend: Optional[str] = None,
        master_addr: Optional[str] = "localhost",
        master_port: Optional[int] = None,
        verbose: bool = True,
        force_spawn: bool = False,
    ) -> None:
        self.devices: List[int] = _parse_devices(devices)
        self.world_size: int = len(self.devices) if len(self.devices) > 0 else 1

        self.master_addr: str = master_addr
        self.master_port: int = (
            master_port
            if master_port is not None
            else _find_free_port(self.master_addr)
        )
        self.verbose: bool = verbose
        self.force_spawn: bool = force_spawn

        # backend choice:
        # - if GPUs selected: default to nccl
        # - else: default to gloo
        if backend is None:
            backend = "nccl" if len(self.devices) > 0 else "gloo"
        self.backend: str = backend
        if self.backend == "nccl" and not torch.cuda.is_available():
            raise RuntimeError(
                "backend 'nccl' requires CUDA, but torch.cuda.is_available() is False"
            )

    def run(
        self,
        worker_fn: Callable[[Env, Any], Any],
        *args,
        **kwargs,
    ) -> Any:
        """Run a worker function under this DDP setup.

        ### Args
            - `worker_fn` (Callable[[Env, Any], Any]): Function to run. Receives
                the process `Env` plus `*args` and `**kwargs`.
            - `*args`: Positional arguments passed to `worker_fn` after `Env`.
            - `**kwargs`: Keyword arguments passed to `worker_fn`.

        ### Returns
            - Return value of `worker_fn` (or from the rank-0 process when using
                multiple processes).

        ### Example
        ```python
        >>> def train(env: Env, lr: float):
        ...     model = Model().to(env.device)
        ...     ...
        >>> ddp = DDP("cuda:0,1")
        >>> ddp.run(train, lr=1e-3)
        ```
        """
        multi_proc = (len(self.devices) > 1) or self.force_spawn

        if multi_proc:
            return self._spawn(worker_fn, args, kwargs)
        else:
            env = self._single_proc_env()
            if self.verbose:
                self._log_env(env)
            return worker_fn(env, *args, **kwargs)

    def _single_proc_env(self) -> Env:
        # CPU-only
        if len(self.devices) == 0:
            return Env(
                rank=0,
                local_rank=0,
                world_size=1,
                device=torch.device("cpu"),
                device_idx=None,
            )

        # Single GPU
        idx = self.devices[0]
        torch.cuda.set_device(idx)
        return Env(
            rank=0,
            local_rank=0,
            world_size=1,
            device=torch.device(f"cuda:{idx}"),
            device_idx=idx,
        )

    def _spawn(
        self,
        worker_fn: Callable[[Env, Any], Any],
        args: Tuple[Any, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        os.environ["MASTER_ADDR"] = self.master_addr
        os.environ["MASTER_PORT"] = str(self.master_port)

        if self.verbose:
            device_str = (
                ",".join(str(d) for d in self.devices) if self.devices else "cpu"
            )
            print(
                f"\033[1m[wah.ddp]\033[0m "
                f"Launching {self.world_size} process{'es' if self.world_size > 1 else ''} "
                f"(backend='{self.backend}', master={self.master_addr}:{self.master_port}, devices={device_str})"
            )

        mp.spawn(
            self._worker_entry,
            args=(worker_fn, args, kwargs),
            nprocs=self.world_size,
            join=True,
        )

    def _worker_entry(
        self,
        local_rank: int,
        worker_fn: Callable[[Env, Any], Any],
        args: Tuple[Any, ...],
        kwargs: dict,
    ) -> None:
        # Map process local_rank -> actual CUDA device index
        if len(self.devices) == 0:
            device = torch.device("cpu")
            idx = None
        else:
            idx = self.devices[local_rank]
            torch.cuda.set_device(idx)
            device = torch.device(f"cuda:{idx}")

        # For single-node, rank == local_rank
        rank = local_rank
        world_size = self.world_size

        dist.init_process_group(
            backend=self.backend,
            init_method="env://",
            rank=rank,
            world_size=world_size,
        )

        env = Env(
            rank=rank,
            local_rank=local_rank,
            world_size=world_size,
            device=device,
            device_idx=idx,
        )

        if self.verbose:
            self._log_env(env)

        try:
            worker_fn(env, *args, **kwargs)
        finally:
            if dist.is_initialized():
                dist.destroy_process_group()

    def _log_env(self, env: Env) -> None:
        msg = (
            f"\033[1m[wah.ddp]\033[0m "
            f"Initialized rank {env.rank}/{env.world_size} "
            f"(device={env.device})"
        )
        print(msg)


def make_dataloader(
    *,
    rank: int,
    world_size: int,
    dataset: Dataset,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    collate_fn: Optional[Callable[..., Any]] = None,
    pin_memory: Optional[bool] = None,
    drop_last: bool = False,
    persistent_workers: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
    generator: Optional[torch.Generator] = None,
    sampler_seed: int = 0,
) -> Tuple[DataLoader, Optional[DistributedSampler]]:
    """Create a DataLoader suitable for (single-node) DDP.

    When world_size > 1, uses DistributedSampler and disables DataLoader shuffle.
    When world_size == 1, no sampler is used and DataLoader shuffle is as requested.
    If a sampler is returned, call sampler.set_epoch(epoch) each epoch.

    ### Args
        - `rank` (int): Global rank of this process.
        - `world_size` (int): Total number of processes.
        - `dataset` (Dataset): Any torch.utils.data.Dataset.
        - `batch_size` (int): Per-process batch size.
        - `shuffle` (bool): Shuffle the data. In DDP this is implemented by DistributedSampler.
        - `num_workers` (int): DataLoader workers per process.
        - `collate_fn` (callable, optional): Collate function for batching samples.
        - `pin_memory` (bool, optional): Defaults to True if CUDA is available, else False.
        - `drop_last` (bool): Drop last incomplete batch.
        - `persistent_workers` (bool, optional): Defaults to True when num_workers > 0, else False.
        - `prefetch_factor` (int, optional): Only valid when num_workers > 0.
        - `generator` (torch.Generator, optional): Used only in single-process mode (sampler is None).
        - `sampler_seed` (int): Base seed for DistributedSampler; vary per epoch via set_epoch().

    ### Returns
        - `Tuple[DataLoader, Optional[DistributedSampler]]`: (dataloader, sampler). Call
            sampler.set_epoch(epoch) each epoch when sampler is not None.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = num_workers > 0

    if world_size > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=shuffle,
            seed=sampler_seed,
            drop_last=drop_last,
        )
        dl_shuffle = False
    else:
        sampler = None
        dl_shuffle = shuffle

    dl_kwargs = dict(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=dl_shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        sampler=sampler,
        drop_last=drop_last,
        persistent_workers=persistent_workers,
    )

    # Only pass prefetch_factor if num_workers > 0 (PyTorch will error otherwise)
    if num_workers > 0 and prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    # generator is only meaningful when sampler is None (single-process shuffle)
    if sampler is None and generator is not None:
        dl_kwargs["generator"] = generator

    loader = DataLoader(**dl_kwargs)
    return loader, sampler
