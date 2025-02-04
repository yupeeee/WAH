import torch

from ...misc.typing import Tensor

__all__ = [
    "brier_score",
    "ece",
]


def brier_score(
    signed_confs: Tensor,
) -> Tensor:
    """Compute the Brier score for predicted probabilities.

    The Brier score measures the mean squared error between predicted probabilities
    and actual outcomes. Lower values indicate better calibration.

    ### Args
        - `signed_confs` (Tensor): 2D tensor of shape (num_epochs, num_data). Each element:
            - > 0 => correct prediction
            - < 0 => incorrect prediction;
            abs(value) => confidence (interpreted as probability in [0,1] if clamp_to_unit=True)

    ### Returns
        - `Tensor`: 1D tensor of shape (num_epochs,), where each element is the Brier score for that epoch

    ### Example
    ```python
    >>> confs = torch.tensor([[0.8, -0.3, 0.6], [-0.9, 0.7, -0.2]])  # 2 epochs, 3 samples each
    >>> brier_scores = brier_score(confs)
    >>> print(brier_scores)  # Brier score for each epoch
    tensor([0.0967, 0.3133])
    ```
    """
    if signed_confs.dim() == 1:
        signed_confs = signed_confs.unsqueeze(dim=0)
    assert signed_confs.dim() == 2

    # 1) Clamp to [-1,1], then take abs() for predicted probability
    p = signed_confs.clamp(-1.0, 1.0).abs()

    # 2) Correctness indicator: 1 if > 0, else 0
    y = (signed_confs > 0).float()

    # 3) Compute Brier score for all epochs: mean((p - y)^2)
    brier_scores = ((p - y) ** 2).mean(dim=1)

    return brier_scores


def ece(
    signed_confs: Tensor,
    n_bins: int = 10,
) -> Tensor:
    """Compute the Expected Calibration Error (ECE) for predicted probabilities.

    ### Args
        - `signed_confs` (Tensor): 2D tensor of shape (num_epochs, num_data). Each element:
            - > 0 => correct prediction
            - < 0 => incorrect prediction;
            abs(value) => confidence (interpreted as probability in [0,1] if clamp_to_unit=True)
        - `n_bins` (int): Number of bins to use when computing ECE. Defaults to 10.

    ### Returns
        - `Tensor`: 1D tensor of shape (num_epochs,), where each element is the ECE for that epoch

    ### Example
    ```python
    >>> confs = torch.tensor([[0.8, -0.3, 0.6], [-0.9, 0.7, -0.2]])  # 2 epochs, 3 samples each
    >>> eces = ece(confs, n_bins=5)
    >>> print(eces)  # ECE value for each epoch
    tensor([0.3000, 0.4667])
    ```
    """
    device = signed_confs.device
    if signed_confs.dim() == 1:
        signed_confs = signed_confs.unsqueeze(dim=0)
    assert signed_confs.dim() == 2

    # Pre-compute the bin boundaries (shared across epochs)
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=device)

    # 1) Clamp to [-1,1], then take abs() for predicted probability
    p = signed_confs.clamp(-1.0, 1.0).abs()

    # 2) Correctness indicator: 1 if > 0, else 0
    y = (signed_confs > 0).float()

    # 3) Assign each sample to a bin
    bin_indices = torch.bucketize(p, bin_boundaries[1:], right=False)

    # 4) Compute bin statistics for all epochs at once
    # Create one-hot encoding for bin assignments
    bin_one_hot = torch.nn.functional.one_hot(
        bin_indices, num_classes=n_bins + 1
    ).float()

    # Sum across samples for each bin and epoch
    counts = bin_one_hot.sum(dim=1)  # shape: (num_epochs, n_bins+1)
    conf_sums = torch.bmm(p.unsqueeze(1), bin_one_hot).squeeze(1)
    acc_sums = torch.bmm(y.unsqueeze(1), bin_one_hot).squeeze(1)

    # Only use first n_bins indices
    counts = counts[:, :n_bins]
    conf_sums = conf_sums[:, :n_bins]
    acc_sums = acc_sums[:, :n_bins]

    # 5) Calculate fractions and averages
    N = p.size(1)
    fraction_in_bin = counts / N
    acc_in_bin = acc_sums / counts.clamp(min=1e-12)
    conf_in_bin = conf_sums / counts.clamp(min=1e-12)

    # 6) Compute ECE for all epochs
    eces = (fraction_in_bin * (acc_in_bin - conf_in_bin).abs()).sum(dim=1)

    return eces
