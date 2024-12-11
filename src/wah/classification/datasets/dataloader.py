from torch.utils.data import DataLoader, default_collate

from ...typing import Dataset
from .transforms import get_mixup_cutmix

__all__ = [
    "load_dataloader",
]


def load_dataloader(
    dataset: Dataset,
    train: bool,
    **kwargs,
) -> DataLoader:
    batch_size = kwargs.get("batch_size", 1)
    shuffle = True if train else False
    num_workers = kwargs.get("num_workers", 0)
    pin_memory = kwargs.get("pin_memory", False)
    mixup_alpha = kwargs.get("mixup_alpha", 0.0)
    cutmix_alpha = kwargs.get("cutmix_alpha", 0.0)
    num_classes = kwargs.get("num_classes", 1000)

    mixup_cutmix = get_mixup_cutmix(
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        num_classes=num_classes,
    )
    if mixup_cutmix is not None:

        def collate_fn(batch):
            return mixup_cutmix(*default_collate(batch))

    else:
        collate_fn = default_collate

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
