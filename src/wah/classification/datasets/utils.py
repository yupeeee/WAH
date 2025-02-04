import torch
from torch.utils.data import Subset

from ...misc.typing import Dataset, Literal, Optional, Tensor, Tuple

__all__ = [
    "compute_mean_and_std",
    "portion_dataset",
]


def compute_mean_and_std(
    dataset: Dataset,
    data_shape: Literal[
        "CHW",
        "HWC",
        "HW",
    ] = "CHW",
) -> Tuple[Tensor, Tensor]:
    """Compute the mean and standard deviation of a dataset.

    ### Args
        - `dataset` (Dataset): Dataset to compute statistics for
        - `data_shape` (Literal["CHW", "HWC", "HW"]): Shape of the data. Defaults to "CHW".
            - "CHW": Channel x Height x Width (PyTorch format)
            - "HWC": Height x Width x Channel (NumPy format)
            - "HW": Height x Width (Grayscale)

    ### Returns
        - `Tuple[Tensor, Tensor]`: Mean and standard deviation tensors

    ### Example
    ```python
    >>> from torchvision.datasets import CIFAR10
    >>> dataset = CIFAR10(root="data", train=True, download=True)
    >>> mean, std = compute_mean_and_std(dataset)
    >>> print(mean)
    tensor([0.4914, 0.4822, 0.4465])
    >>> print(std)
    tensor([0.2470, 0.2435, 0.2616])
    ```
    """
    # Convert dataset to tensor
    data = torch.stack([x[0] if isinstance(x, tuple) else x for x in dataset]).to(
        torch.float32
    )
    # Define reduction dimensions based on data shape
    dim_to_reduce = {"CHW": (0, 2, 3), "HWC": (0, 1, 2), "HW": (0, 1, 2)}.get(
        data_shape
    )
    if dim_to_reduce is None:
        raise ValueError(f"Unsupported data_shape: {data_shape}")
    # Compute statistics in one pass
    mean = data.mean(dim=dim_to_reduce)
    std = data.std(dim=dim_to_reduce)
    return mean, std


def portion_dataset(
    dataset: Dataset,
    portion: float,
    balanced: Optional[bool] = True,
    random_sample: Optional[bool] = False,
) -> Dataset:
    """Create a portion of a dataset.

    ### Args
        - `dataset` (Dataset): Dataset to create a portion from
        - `portion` (float): Portion of the dataset to use (0 < portion <= 1)
        - `balanced` (Optional[bool]): Whether to create a balanced dataset. Defaults to True.
          If True, the dataset must have a `targets` attribute.
        - `random_sample` (Optional[bool]): Whether to randomly sample the data. Defaults to False.
          If True, the data will be randomly sampled. Otherwise, the first `portion` of the data will be used.

    ### Returns
        - `Dataset`: A subset of the dataset containing the specified portion of data

    ### Example
    ```python
    >>> from wah.classification.datasets import CIFAR10
    >>> dataset = CIFAR10(...)
    >>> print(f"Original dataset size: {len(dataset)}")  # 50000
    >>> # Create a balanced dataset with 10% of the data
    >>> subset = portion_dataset(dataset, portion=0.1)
    >>> print(f"Subset size: {len(subset)}")  # 5000
    >>> # Create an unbalanced dataset with 10% of the data
    >>> subset = portion_dataset(dataset, portion=0.1, balanced=False)
    >>> print(f"Subset size: {len(subset)}")  # 5000
    >>> # Create a balanced dataset with 10% of randomly sampled data
    >>> subset = portion_dataset(dataset, portion=0.1, random_sample=True)
    >>> print(f"Subset size: {len(subset)}")  # 5000
    ```
    """
    assert 0 < portion <= 1, f"Expected 0 < portion <= 1, got {portion}"
    if balanced:
        assert hasattr(
            dataset, "targets"
        ), f"Unable to create a balanced dataset as there are no targets in the dataset."
        targets = torch.tensor(dataset.targets)
        classes = torch.unique(targets).sort()[0]
        indices = []
        for c in classes:
            c_indices = torch.where(targets == c)[0]
            num_c = int(len(c_indices) * portion)
            if random_sample:
                perm = torch.randperm(len(c_indices))[:num_c]
                indices.extend(c_indices[perm].tolist())
            else:
                indices.extend(c_indices[:num_c].tolist())
    else:
        num_data = int(len(dataset) * portion)
        if random_sample:
            indices = torch.randperm(len(dataset))[:num_data].tolist()
        else:
            indices = list(range(num_data))
    return Subset(dataset, indices)
