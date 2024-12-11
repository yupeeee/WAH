import random

import torch
from torch.utils.data import Subset

from ...typing import Dataset, List, Literal, Optional, Tensor, Tuple

__all__ = [
    "compute_mean_and_std",
    "portion_dataset",
    "tensor_to_dataset",
]


def compute_mean_and_std(
    dataset: Dataset,
    data_shape: Literal[
        "CHW",
        "HWC",
        "HW",
    ] = "CHW",
) -> Tuple[Tensor, Tensor]:
    """
    Computes the mean and standard deviation of the dataset.

    ### Parameters
    - `dataset` (Dataset): The dataset for which to compute the mean and standard deviation. Note that the dataset must contain data of type `torch.Tensor` for computation.
    - `data_shape` (Literal["CHW", "HWC", "HW"]): The shape format of the data in the dataset. Defaults to "CHW".
        - "CHW": Channels-Height-Width
        - "HWC": Height-Width-Channels
        - "HW": Height-Width (grayscale images)

    ### Returns
    - `Tuple[Tensor, Tensor]`: The mean and standard deviation of the dataset.

    ### Raises
    - `ValueError`: If an unsupported `data_shape` is provided.

    ### Notes
    - The dataset must contain data of type `torch.Tensor` for computation. For instance, use `torchvision.transforms.ToTensor()` to transform PIL Image data into tensor format.
    - The data is converted to float32 before computing mean and standard deviation.

    ### Example
    ```python
    import wah

    dataset = Dataset(...)    # data is RGB image
    mean, std = wah.datasets.mean_and_std(
        dataset=dataset,
        data_shape="CHW",
    )
    # mean.shape: torch.Size([3])
    # std.shape: torch.Size([3])
    ```
    """
    data = []

    for x in dataset:
        if isinstance(x, tuple):
            x: Tensor = x[0]

        data.append(x.unsqueeze(dim=0))

    data = torch.cat(data, dim=0).to(torch.float32)

    if data_shape == "CHW":
        dim_to_reduce = (0, 2, 3)
    elif data_shape == "HWC":
        dim_to_reduce = (0, 1, 2)
    elif data_shape == "HW":
        dim_to_reduce = (0, 1, 2)
    else:
        raise ValueError(f"Unsupported data_shape: {data_shape}")

    mean = data.mean(dim=dim_to_reduce)
    std = data.std(dim=dim_to_reduce)

    return mean, std


def portion_dataset(
    dataset: Dataset,
    portion: float,
    balanced: Optional[bool] = True,
    random_sample: Optional[bool] = False,
) -> Dataset:
    """
    Creates a subset of the given dataset based on the specified portion.

    ### Parameters
    - `dataset` (Dataset): The dataset from which to create the subset.
    - `portion` (float): The portion of the dataset to include in the subset. Must be in range (0, 1].
    - `balanced` (Optional[bool]): Whether to create a balanced subset. If True, the subset will have a balanced number of samples from each class. Defaults to True.
    - `random_sample` (Optional[bool]): Whether to randomly sample the indices. If False, the subset will include the first `portion` of the dataset. Defaults to False.

    ### Returns
    - `Dataset`: A subset of the original dataset based on the specified portion.

    ### Raises
    - `AssertionError`: If `portion` is not in range (0, 1].
    - `AssertionError`: If `balanced` is True and the dataset does not have targets.

    ### Notes
    - When `balanced` is True, the dataset must have a 'targets' attribute containing the labels for balancing.
    - When `random_sample` is True, the function will randomly select samples. This can lead to different subsets on different runs.

    ### Example
    ```python
    import wah

    dataset = Dataset(...)
    subset = wah.datasets.portion_dataset(
        dataset=dataset,
        portion=0.8,
        balanced=True,
        random_sample=False,
    )
    # len(subset) == len(dataset) * 0.8
    ```
    """
    assert 0 < portion <= 1, f"Expected 0 < portion <= 1, got {portion}"

    if balanced:
        assert hasattr(
            dataset, "targets"
        ), f"Unable to create a balanced dataset as there are no targets in the dataset."

        targets = dataset.targets
        classes = list(set(targets))

        indices = []

        for c in classes:
            c_indices = [i for i, target in enumerate(targets) if target == c]
            num_c = int(len(c_indices) * portion)

            if random_sample:
                c_indices = random.sample(c_indices, num_c)
            else:
                c_indices = c_indices[:num_c]

            indices += c_indices

    else:
        num_data = int(len(dataset) * portion)

        if random_sample:
            indices = random.sample([i for i in range(len(dataset))], num_data)
        else:
            indices = [i for i in range(len(dataset))][:num_data]

    return Subset(dataset, indices)


class TensorDataset(Dataset):
    """
    A dataset class for wrapping a tensor and optionally providing target labels.

    ### Attributes
    - `data` (Tensor): The tensor that stores the dataset.
    - `targets` (Optional[List[int]]): The list of target labels, if provided.

    ### Methods
    - `__getitem__(index) -> Union[Tensor, Tuple[Tensor, int]]`: Retrieves the tensor (and optionally the target) at the specified index.
    - `__len__() -> int`: Returns the length of the dataset (i.e., the number of elements in the tensor).
    - `add_targets(targets) -> None`: Adds target labels to the dataset.
    - `add_dummy_targets() -> None`: Adds dummy target labels to the dataset.
    """

    def __init__(
        self,
        tensor: Tensor,
        targets: Optional[List[int]] = None,
    ) -> None:
        """
        - `tensor` (Tensor): The tensor that stores the dataset.
        - `targets` (Optional[List[int]]): The list of target labels. Defaults to None.
        """
        super().__init__()

        self.data = tensor
        self.targets = targets

    def __getitem__(
        self,
        index: int,
    ) -> Tensor:
        if self.targets is None:
            return self.data[index]
        else:
            return self.data[index], self.targets[index]

    def __len__(
        self,
    ) -> int:
        return len(self.data)

    def add_targets(
        self,
        targets: List[int],
    ) -> None:
        """
        Adds target labels to the dataset.

        ### Parameters
        - `targets (List[int])`: The list of target labels to add.

        ### Returns
        - `None`
        """
        self.targets = targets

    def add_dummy_targets(
        self,
    ) -> None:
        """
        Adds dummy target labels to the dataset.

        ### Parameters
        - `None`

        ### Returns
        - `None`
        """
        self.targets = torch.zeros(size=(len(self),), dtype=torch.int64)


def tensor_to_dataset(
    tensor: Tensor,
    create_dummy_targets: Optional[bool] = False,
) -> TensorDataset:
    """
    Converts a tensor into a dataset by wrapping it with the `TensorDataset` class.

    ### Parameters
    - `tensor (Tensor)`: The tensor to convert into a dataset.
    - `create_dummy_targets (Optional[bool])`: Whether to add dummy target labels to the dataset. Defaults to False.

    ### Returns
    - `Dataset`: A dataset object wrapping the input tensor.
    """
    dataset = TensorDataset(tensor)

    if create_dummy_targets:
        dataset.add_dummy_targets()

    return dataset
