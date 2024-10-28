import torch

from ...typing import List, Tensor, Tuple, Union


def compute_ece(
    signed_confs: Union[Tensor, List[float]],
    num_bins: int = 10,
) -> Tuple[float, List[float], List[float]]:
    if isinstance(signed_confs, list):
        signed_confs = torch.Tensor(signed_confs)

    is_correct: Tensor = (signed_confs > 0).int()
    confs = torch.abs(signed_confs)
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    accuracy_per_bin: List[float] = []
    confidence_per_bin: List[float] = []

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confs > bin_lower) & (confs <= bin_upper)
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0:
            accuracy_in_bin = is_correct[in_bin].float().mean().item()
            avg_confidence_in_bin = confs[in_bin].mean().item()
            ece += abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin.item()

            accuracy_per_bin.append(accuracy_in_bin)
            confidence_per_bin.append(avg_confidence_in_bin)

        else:
            accuracy_per_bin.append(0.0)
            confidence_per_bin.append(0.0)

    return ece, accuracy_per_bin, confidence_per_bin
