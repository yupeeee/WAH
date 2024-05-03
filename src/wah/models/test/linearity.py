import torch
import tqdm

from ...typing import (
    DataLoader,
    Dataset,
    Devices,
    Module,
    Optional,
    ResQueue,
    Tensor,
)
from ...utils import dist

__all__ = [
    "LinearityTest",
]


def cpu_run(
    model: Module,
    dataset: Dataset,
    batch_size: int = 1,
    num_workers: int = 0,
    verbose: bool = False,
    desc: Optional[str] = None,
) -> float:
    device = torch.device("cpu")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    model = model.to(device)

    for data, targets in tqdm.tqdm(
        dataloader,
        desc=f"linearity test" if desc is None else desc,
        disable=not verbose,
    ):
        data: Tensor = data.to(device)
        targets: Tensor = targets.to(device)

        # TODO
