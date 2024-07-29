from ...typing import (
    DataLoader,
    Dataset,
    Optional,
)
from .transforms import CollateFunction

__all__ = [
    "to_dataloader",
    "CollateFunction",
]


def to_dataloader(
    dataset: Dataset,
    train: Optional[bool] = False,
    batch_size: int = 1,
    num_workers: int = 0,
    mixup_alpha: float = 0.0,
    cutmix_alpha: float = 0.0,
    **kwargs,
) -> DataLoader:
    """
    Loads a DataLoader for the given dataset with various configurable options.

    ### Parameters
    - `dataset` (Dataset): The dataset to load.
    - `train` (Optional[bool]): Indicates if the DataLoader is for training. Defaults to False.
        - If True, data is shuffled and mixup/cutmix is applied if specified.
        - If False, data is not shuffled and mixup/cutmix is not applied.
    - `batch_size` (int): The number of samples per batch. Defaults to 1.
    - `num_workers` (int): The number of subprocesses to use for data loading. Defaults to 0.
    - `mixup_alpha` (float): The alpha parameter for mixup. Defaults to 0.0, meaning no mixup is applied.
    - `cutmix_alpha` (float): The alpha parameter for cutmix. Defaults to 0.0, meaning no cutmix is applied.
    - `**kwargs`: Additional keyword arguments for further customization.
        - Possible keys include:
            - "batch_size": Override the default batch size.
            - "num_workers": Override the default number of workers.
            - "mixup_alpha": Override the default mixup alpha.
            - "cutmix_alpha": Override the default cutmix alpha.
            - "num_classes": The number of classes in the dataset.

    ### Returns
    - `DataLoader`: A DataLoader configured with the specified options.

    ### Notes
    - If `train` is True, a custom collate function `CollateFunction` is used for mixup/cutmix augmentation.
    - The `persistent_workers` flag is set to True if `num_workers` is greater than 0.
    - The number of classes (`num_classes`) is inferred from the dataset's targets if not provided.

    ### Example
    ```python
    import wah

    dataset = Dataset(...)
    dataloader = wah.dataloader.load(
        dataset=dataset,
        train=False,
        batch_size=64,
        num_workers=...,
        mixup_alpha=...,
        cutmix_alpha=...,
    )
    for batch in dataloader:
        print(len(batch))   # 64
        break
    ```
    """
    batch_size = kwargs["batch_size"] if "batch_size" in kwargs.keys() else batch_size
    shuffle = True if train else False
    num_workers = (
        kwargs["num_workers"] if "num_workers" in kwargs.keys() else num_workers
    )
    persistent_workers = True if num_workers > 0 else False
    mixup_alpha = (
        kwargs["mixup_alpha"] if "mixup_alpha" in kwargs.keys() else mixup_alpha
    )
    cutmix_alpha = (
        kwargs["cutmix_alpha"] if "cutmix_alpha" in kwargs.keys() else cutmix_alpha
    )
    num_classes = (
        kwargs["num_classes"]
        if "num_classes" in kwargs.keys()
        else (
            len(list(set(dataset.targets)))
            if hasattr(dataset, "targets")
            else len(list(set(dataset.dataset.targets)))
        )
    )
    collate_fn = (
        CollateFunction(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            num_classes=num_classes,
        )
        if train
        else None
    )

    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        collate_fn=collate_fn,
    )
