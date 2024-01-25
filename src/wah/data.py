from .typing import Config, DataLoader, Dataset, Optional

__all__ = [
    "load_dataloader",
]


def load_dataloader(
        dataset: Dataset,
        config: Config,
        shuffle: Optional[bool] = None,
) -> DataLoader:
    return DataLoader(
        dataset=dataset,
        batch_size=config["batch_size"],
        shuffle=shuffle,
        num_workers=config["num_workers"],
        persistent_workers=True if config["num_workers"] > 0 else False,
    )
