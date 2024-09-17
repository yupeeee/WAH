import torch
from torchmetrics.classification import (
    MulticlassCalibrationError as _MultiClassCalibrationError,
)

from ...typing import Any, List, Literal, Optional, Tensor, Tuple, Union

__all__ = [
    "MultiClassCalibrationError",
]


def _binning_bucketize(
    confidences: Tensor,
    accuracies: Tensor,
    bin_boundaries: Tensor,
) -> Tuple[Tensor, Tensor, Tensor]:
    """
    Compute calibration bins using ``torch.bucketize``. Use for ``pytorch >=1.6``.

    ### Parameters
    - `confidences (Tensor)`: The confidence (i.e. predicted prob) of the top1 prediction.
    - `accuracies (Tensor)`: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
    - `bin_boundaries (Tensor)`: Bin boundaries separating the ``linspace`` from 0 to 1.

    ### Returns
    - `Tuple[Tensor, Tensor, Tensor]`: A tuple with binned accuracy, binned confidence and binned probabilities.
    """
    accuracies = accuracies.to(dtype=confidences.dtype)
    acc_bin = torch.zeros(
        len(bin_boundaries), device=confidences.device, dtype=confidences.dtype
    )
    conf_bin = torch.zeros(
        len(bin_boundaries), device=confidences.device, dtype=confidences.dtype
    )
    count_bin = torch.zeros(
        len(bin_boundaries), device=confidences.device, dtype=confidences.dtype
    )

    indices = torch.bucketize(confidences, bin_boundaries, right=True) - 1

    count_bin.scatter_add_(dim=0, index=indices, src=torch.ones_like(confidences))

    conf_bin.scatter_add_(dim=0, index=indices, src=confidences)
    conf_bin = torch.nan_to_num(conf_bin / count_bin)

    acc_bin.scatter_add_(dim=0, index=indices, src=accuracies)
    acc_bin = torch.nan_to_num(acc_bin / count_bin)

    prop_bin = count_bin / count_bin.sum()

    return acc_bin, conf_bin, prop_bin


def _ce_compute(
    confidences: Tensor,
    accuracies: Tensor,
    bin_boundaries: Union[Tensor, int],
    norm: str = "l1",
    debias: bool = False,
) -> Tensor:
    """
    Compute the calibration error given the provided bin boundaries and norm.

    ### Parameters
    - `confidences (Tensor)`: The confidence (i.e. predicted prob) of the top1 prediction.
    - `accuracies (Tensor)`: 1.0 if the top-1 prediction was correct, 0.0 otherwise.
    - `bin_boundaries (Union[Tensor, int])`: Bin boundaries separating the ``linspace`` from 0 to 1.
    - `norm (str)`: Norm function to use when computing calibration error. Defaults to "l1".
    - `debias (bool)`: Apply debiasing to L2 norm computation as in `Verified Uncertainty Calibration`. Defaults to False.

    ### Returns
    - `Tensor`: Calibration error scalar.

    ### Raises
    - `ValueError`: If an unsupported norm function is provided.
    """
    if isinstance(bin_boundaries, int):
        bin_boundaries = torch.linspace(
            0, 1, bin_boundaries + 1, dtype=confidences.dtype, device=confidences.device
        )

    if norm not in ["l1", "sign", "l2", "max"]:
        raise ValueError(
            f"Argument `norm` is expected to be one of `l1`, `sign`, `l2`, `max` but got {norm}"
        )

    with torch.no_grad():
        acc_bin, conf_bin, prop_bin = _binning_bucketize(
            confidences, accuracies, bin_boundaries
        )

    if norm == "l1":
        return torch.sum(torch.abs(acc_bin - conf_bin) * prop_bin)
    if norm == "sign":
        return torch.sum((acc_bin - conf_bin) * prop_bin)
    if norm == "max":
        ce = torch.max(torch.abs(acc_bin - conf_bin))
    if norm == "l2":
        ce = torch.sum(torch.pow(acc_bin - conf_bin, 2) * prop_bin)
        if debias:
            debias_bins = (acc_bin * (acc_bin - 1) * prop_bin) / (
                prop_bin * accuracies.size()[0] - 1
            )
            ce += torch.sum(torch.nan_to_num(debias_bins))
        return torch.sqrt(ce) if ce > 0 else torch.tensor(0)

    return ce


def dim_zero_cat(x: Union[Tensor, List[Tensor]]) -> Tensor:
    """
    Concatenation along the zero dimension.

    ### Parameters
    - `x (Union[Tensor, List[Tensor]])`: The input tensor or list of tensors to concatenate.

    ### Returns
    - `Tensor`: The concatenated tensor.

    ### Raises
    - `ValueError`: If the input list is empty.
    """
    if isinstance(x, torch.Tensor):
        return x

    x = [y.unsqueeze(0) if y.numel() == 1 and y.ndim == 0 else y for y in x]

    if not x:
        raise ValueError("No samples to concatenate")

    return torch.cat(x, dim=0)


class MulticlassCalibrationError(_MultiClassCalibrationError):
    def __init__(
        self,
        num_classes: int,
        n_bins: int = 10,
        norm: Literal[
            "l1",
            "sign",
            "l2",
            "max",
        ] = "l1",
        ignore_index: Optional[int] = None,
        validate_args: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initializes the MulticlassCalibrationError class.

        ### Parameters
        - `num_classes (int)`: Number of classes.
        - `n_bins (int)`: Number of bins to use. Defaults to 10.
        - `norm (Literal["l1", "sign", "l2", "max"])`: Norm function to use when computing calibration error. Defaults to "l1".
        - `ignore_index (Optional[int])`: Index to ignore when computing. Defaults to None.
        - `validate_args (bool)`: If True, validates the arguments. Defaults to False.
        - `**kwargs (Any)`: Additional keyword arguments.
        """
        super().__init__(
            num_classes, n_bins, norm, ignore_index, validate_args, **kwargs
        )

    def compute(self) -> Tensor:
        """
        Compute the metric.

        ### Returns
        - `Tensor`: Computed calibration error.
        """
        confidences = dim_zero_cat(self.confidences)
        accuracies = dim_zero_cat(self.accuracies)

        return _ce_compute(confidences, accuracies, self.n_bins, norm=self.norm)
