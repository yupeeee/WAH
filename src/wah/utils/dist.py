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
    Process,
    Union,
)

__all__ = [
    "parse_devices",
    "init_os_env",
    "init_dist",
    "cleanup",
    "Queue",
    "set_start_method",
    "run_fn",
    "load_dataloader",
    "load_model",
]


def parse_devices(
    devices: Devices,
) -> Union[List[int], str]:
    """
    Parses and validates the specified devices for CUDA computation.

    ### Parameters
    - `devices` (Union[int, str, List[Union[int, str]]]):
      The devices to be used.
        - Can be "auto" to automatically use all available CUDA devices.
        - Can be a single int or str representing a device ID.
        - Can be a list of ints or strs representing multiple device IDs.

    ### Returns
    - `Union[List[int], str]`:
      A sorted list of device IDs if valid, or raises a ValueError if the input is invalid.

    ### Raises
    - `AssertionError`:
      If no CUDA devices are available.
    - `ValueError`:
      If the input devices are not in a valid format.

    ### Notes
    - The function assumes that at least one CUDA device is available.
    - The "auto" option automatically detects and uses all available CUDA devices.
    """
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
    """
    Automatically sets the backend for distributed computing based on the current operating system.

    ### Returns
    - `str`:
      The backend to be used for distributed computing.
        - "nccl" for Linux.
        - "gloo" for Windows.

    ### Raises
    - `SystemError`:
      If the function is run on an unsupported operating system.

    ### Notes
    - This function supports automatic backend initialization only for Linux and Windows.
    """
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
    """
    Initializes the environment variables for distributed computing.

    ### Parameters
    - `devices` (List[int]):
      A list of device IDs to be made visible to CUDA.
    - `master_addr` (str):
      The address of the master node.
      Defaults to "localhost".
    - `master_port` (int):
      The port of the master node.
      Defaults to 12345.

    ### Returns
    - `None`
    """
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(d) for d in devices])


def init_dist(
    rank: int,
    nprocs: int,
    # backend: str = "nccl",
    init_method: str = "env://",
) -> None:
    """
    Initializes the distributed computing environment.

    ### Parameters
    - `rank` (int):
      The rank of the current process.
    - `nprocs` (int):
      The total number of processes.
    - `init_method` (str):
      The initialization method for the process group.
      Defaults to "env://".

    ### Returns
    - `None`

    ### Raises
    - `SystemError`:
      If the backend is not supported on the current operating system.
    """
    backend = set_backend()

    dist.init_process_group(
        backend=backend,
        init_method=init_method,
        world_size=nprocs,
        rank=rank,
    )

    torch.cuda.set_device(rank)


def cleanup() -> None:
    """
    Cleans up the distributed computing environment by destroying the process group.

    ### Parameters
    - `None`

    ### Returns
    - `None`

    ### Notes
    - This function is used to properly shut down the distributed process group initialized by `torch.distributed`.
    - It should be called after all distributed operations are complete to ensure that resources are properly released.
    """
    dist.destroy_process_group()


Queue = mp.Queue


def set_start_method() -> None:
    """
    Sets the multiprocessing start method to 'spawn' to avoid CUDA initialization issues.

    ### Parameters
    - `None`

    ### Returns
    - `None`

    ### Notes
    - This function sets the start method for multiprocessing to "spawn".
    - It addresses the `RuntimeError` related to re-initializing CUDA in forked subprocesses.
    - To use CUDA with multiprocessing, the "spawn" start method must be used.
    """
    # [FIX2] Add set_start_method("spawn") due to RuntimeError:
    # Cannot re-initialize CUDA in forked subprocess.
    # To use CUDA with multiprocessing, you must use the 'spawn' start method
    torch.multiprocessing.set_start_method("spawn")


def run_fn(
    fn: Any,
    args: Any,
    nprocs: int,
    **kwargs,
) -> None:
    """
    Runs a function across multiple processes using multiprocessing.

    ### Parameters
    - `fn` (Any):
      The function to be executed in multiple processes.
    - `args` (Any):
      The arguments to pass to the function.
    - `nprocs` (int):
      The number of processes to spawn.
    - `**kwargs`:
      Additional keyword arguments to pass to `mp.spawn`.

    ### Returns
    - `None`

    ### Notes
    - This function tries to use `mp.spawn` to run the function in multiple processes.
    - If an exception occurs (e.g., due to a SIGSEGV error), it falls back to manually starting and joining processes.
    """
    try:
        mp.spawn(fn, args, nprocs, **kwargs)

    # [FIX1] Replace mp.spawn to start/join due to SIGSEGV error
    except Exception:
        children: List[Process] = []

        for i in range(nprocs):
            subargs = tuple([i] + list(args))
            subproc = mp.Process(target=fn, args=subargs)
            children.append(subproc)
            subproc.start()

        for i in range(nprocs):
            children[i].join()


def load_dataloader(
    rank: int,
    nprocs: int,
    dataset: Dataset,
    **kwargs,
) -> DataLoader:
    """
    Loads a DataLoader for the given dataset with distributed sampling.

    ### Parameters
    - `rank` (int):
      The rank of the current process.
    - `nprocs` (int):
      The total number of processes.
    - `dataset` (Dataset):
      The dataset to load.
    - `**kwargs`:
      Additional keyword arguments to pass to the DataLoader.

    ### Returns
    - `DataLoader`:
      A DataLoader configured with distributed sampling.

    ### Notes
    - This function creates a `DistributedSampler` for the dataset to ensure that each process gets a unique subset of the data.
    - The `DataLoader` is configured with this sampler to handle distributed data loading.
    """
    sampler = DistributedSampler(dataset, num_replicas=nprocs, rank=rank)
    dataloader = DataLoader(dataset, sampler=sampler, **kwargs)

    return dataloader


def load_model(
    rank: int,
    model: Module,
) -> Module:
    """
    Loads the model onto the specified device and wraps it with DistributedDataParallel (DDP).

    ### Parameters
    - `rank` (int):
      The rank of the current process, which determines the device ID.
    - `model` (Module):
      The PyTorch model to be loaded and wrapped with DDP.

    ### Returns
    - `Module`:
      The model loaded onto the specified device and wrapped with DDP.

    ### Notes
    - This function moves the model to the device corresponding to the given rank.
    - The model is then wrapped with `DistributedDataParallel` to enable distributed training.
    """
    model.to(rank)
    model = DDP(model, device_ids=[rank])

    return model
