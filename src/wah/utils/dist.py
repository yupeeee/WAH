import os
import platform

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from ..typing import (
    Any,
    DataLoader,
    Dataset,
    Devices,
    List,
    Module,
    Union,
)

__all__ = [
    "parse_devices",
    "init_os_env",
    "init_dist",
    "cleanup",
    "Queue",
    "run_fn",
    "load_dataloader",
    "load_model",
]


def parse_devices(
    devices: Devices,
) -> Union[List[int], str]:
    assert torch.cuda.device_count() > 0

    msg = (
        f"devices must be 'auto', "
        + f"or a type of int, str, list[int|str] that specify device IDs, "
        + f"but got {devices} (type: {type(devices)})"
    )

    # single device in int
    if isinstance(devices, int):
        devices = [devices]

    elif isinstance(devices, str):
        # remove spaces
        devices = devices.replace(" ", "")

        # auto
        if devices == "auto":
            if torch.cuda.device_count() > 0:
                devices = [
                    i
                    for i in range(torch.cuda.device_count())
                    if torch.cuda.get_device_properties(i).name
                ]

        # single device in str
        elif devices.isnumeric():
            devices = [int(devices)]

        # multiple devices in str
        elif "," in devices:
            devices = [int(d) for d in devices.split(",")]

        else:
            raise ValueError(msg)

    # single/multiple device(s) in list
    elif isinstance(devices, list):
        devices = [int(device) for device in devices]

    else:
        raise ValueError(msg)

    devices.sort()

    return devices


def set_backend() -> str:
    current_os = platform.system()

    if current_os == "Linux":
        return "nccl"

    elif current_os == "Windows":
        return "gloo"

    else:
        raise SystemError(
            f"backend auto init is supported only for Linux and Windows (current os: {current_os})"
        )


def init_os_env(
    devices: List[int],
    master_addr: str = "localhost",
    master_port: int = 12345,
) -> None:
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in devices])


def init_dist(
    rank: int,
    nprocs: int,
    # backend: str = "nccl",
    init_method: str = "env://",
) -> None:
    backend = set_backend()

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=nprocs,
        rank=rank,
    )

    torch.cuda.set_device(rank)


def cleanup() -> None:
    dist.destroy_process_group()


Queue = mp.Queue


def run_fn(
    fn: Any,
    args: Any,
    nprocs: int,
    **kwargs,
) -> None:
    mp.spawn(fn, args, nprocs, **kwargs)


def load_dataloader(
    rank: int,
    nprocs: int,
    dataset: Dataset,
    **kwargs,
) -> DataLoader:
    sampler = DistributedSampler(dataset, num_replicas=nprocs, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, **kwargs)

    return dataloader


def load_model(
    rank: int,
    model: Module,
) -> Module:
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    return model
