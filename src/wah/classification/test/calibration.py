import torch

from ...typing import Tensor

__all__ = [
    "ece",
]


def ece(
    signed_confs: Tensor,
    n_bins: int = 10,
) -> Tensor:
    """
    Compute the Expected Calibration Error (ECE) for each row (epoch) of a 2D tensor
    of 'signed confidences'.

    Args:
      signed_confs (Tensor):
          2D tensor of shape (num_epochs, num_data)
          Each element:
            - > 0 => correct prediction
            - < 0 => incorrect prediction
          abs(value) => confidence (interpreted as probability in [0,1] if clamp_to_unit=True)
      n_bins (int):
          Number of bins to use when computing ECE. Default = 10.

    Returns:
      Tensor:
          1D tensor of shape (num_epochs,), where each element is the ECE for that epoch.
    """
    device = signed_confs.device
    if signed_confs.dim() == 1:
        signed_confs = signed_confs.unsqueeze(dim=0)
    assert signed_confs.dim() == 2
    num_epochs, _ = signed_confs.shape

    # Pre-compute the bin boundaries (shared across epochs)
    bin_boundaries = torch.linspace(0, 1, steps=n_bins + 1, device=device)

    ece_values = []
    for epoch_idx in range(num_epochs):
        # Extract the row for this epoch
        row = signed_confs[epoch_idx]

        # 1) Clamp to [-1,1], then take abs() for predicted probability
        p = row.clamp(-1.0, 1.0).abs()

        # 2) Correctness indicator: 1 if > 0, else 0
        y = (row > 0).float()

        # 3) Assign each sample to a bin
        #    bucketize(...) returns an integer in [0..n_bins]
        #    We treat p == 1.0 as the last bin (n_bins index).
        bin_indices = torch.bucketize(p, bin_boundaries[1:], right=False)

        # 4) Compute counts, sums of confidences, sums of correctness in each bin
        counts = torch.bincount(bin_indices, minlength=n_bins + 1).float()
        conf_sums = torch.bincount(bin_indices, weights=p, minlength=n_bins + 1)
        acc_sums = torch.bincount(bin_indices, weights=y, minlength=n_bins + 1)

        # Only use first n_bins indices (bucketize can produce index == n_bins if p==1.0)
        counts = counts[:n_bins]
        conf_sums = conf_sums[:n_bins]
        acc_sums = acc_sums[:n_bins]

        # 5) Fraction of samples in each bin, plus average accuracy/confidence
        N = p.numel()
        fraction_in_bin = counts / N
        acc_in_bin = acc_sums / counts.clamp(min=1e-12)  # avoid divide-by-zero
        conf_in_bin = conf_sums / counts.clamp(min=1e-12)

        # 6) Summation of fraction_in_bin * |acc - conf|
        ece_val = (fraction_in_bin * (acc_in_bin - conf_in_bin).abs()).sum()
        ece_values.append(ece_val)

    return torch.stack(ece_values)
