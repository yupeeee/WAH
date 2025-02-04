from torch.utils.data import DataLoader, default_collate

from ...misc.typing import Dataset
from .transforms import get_mixup_cutmix

__all__ = [
    "load_dataloader",
]


def load_dataloader(
    dataset: Dataset,
    train: bool,
    **kwargs,
) -> DataLoader:
    """Load a DataLoader for a dataset.

    ### Args
        - `dataset` (Dataset): Dataset to load
        - `train` (bool): Whether the dataset is for training
        - `**kwargs`: Additional arguments to pass to the DataLoader
            - `batch_size` (int): Batch size. Defaults to 1.
            - `num_workers` (int): Number of workers. Defaults to 0.
            - `pin_memory` (bool): Whether to pin memory. Defaults to False.
            - `mixup_alpha` (float): Mixup alpha value. Defaults to 0.0.
            - `cutmix_alpha` (float): CutMix alpha value. Defaults to 0.0.
            - `num_classes` (int): Number of classes. Defaults to 1000.

    ### Returns
        - `DataLoader`: DataLoader for the dataset

    ### Example
    ```python
    >>> from wah.classification.datasets import CIFAR10
    >>> train_dataset = CIFAR10(root="/raid/datasets/cifar10", split="train", transform="auto")
    >>> # Create training dataloader
    >>> train_loader = load_dataloader(
    ...     dataset=train_dataset,
    ...     train=True,
    ...     batch_size=256,
    ...     num_workers=4,
    ...     pin_memory=True,
    ...     mixup_alpha=0.2,
    ...     cutmix_alpha=1.0,
    ...     num_classes=10,
    ... )
    >>> test_dataset = CIFAR10(root="/raid/datasets/cifar10", split="test", transform="auto")
    >>> # Create test dataloader
    >>> test_loader = load_dataloader(
    ...     dataset=test_dataset,
    ...     train=False,
    ...     batch_size=256,
    ...     num_workers=4,
    ...     pin_memory=True,
    ...     num_classes=10,
    ... )
    ```
    """
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
    if train and mixup_cutmix is not None:

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
