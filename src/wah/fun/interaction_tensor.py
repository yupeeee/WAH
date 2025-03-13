# Yiding Jiang, Christina Baek, J Kolter
# On the Joint Interaction of Models, Data, and Features
# ICLR 2024 (Oral)
# https://iclr.cc/virtual/2024/oral/19712

# TODO: implement the interaction tensor

import torch

from ..misc.typing import Tensor


def pca_proj(
    features: Tensor,
    n_components: int,
) -> Tensor:
    """Project features to PCA subspace."""
    # Center the data
    features_centered = features - torch.mean(features, dim=0)
    # Compute SVD
    U, S, Vh = torch.linalg.svd(features_centered, full_matrices=False)
    # Select top n_components right singular vectors
    components = Vh.T[:, :n_components]
    # Project the data
    return torch.mm(features_centered, components)
