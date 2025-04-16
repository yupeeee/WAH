import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from scipy import linalg
from torchvision.models import Inception_V3_Weights, inception_v3

from ..misc.typing import Device, Image, List, Tensor

__all__ = [
    "FID",
]


class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device: Device) -> None:
        super().__init__()
        self.device = device
        self.model = (
            inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
            .to(self.device)
            .eval()
        )
        self.acts: Tensor = None

        # Register hook
        self.model.avgpool.register_forward_hook(self._hook_fn)

    def _hook_fn(self, module, input, output):
        self.acts = output.detach()

    def forward(self, x: Tensor) -> Tensor:
        self.acts = None  # reset
        with torch.no_grad():
            _ = self.model(x.to(self.device))
        return self.acts.view(x.size(0), -1)  # (N, 2048)


def _calculate_fid(
    f1: Tensor,
    f2: Tensor,
    eps: float = 1e-6,
) -> float:
    # https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py
    f1 = f1.cpu().numpy()  # (N, 2048)
    f2 = f2.cpu().numpy()  # (N, 2048)

    mu1 = np.mean(f1, axis=0)
    mu2 = np.mean(f2, axis=0)

    sigma1 = np.cov(f1, rowvar=False)
    sigma2 = np.cov(f2, rowvar=False)

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, (
        "Training and test mean vectors have different lengths"
    )
    assert sigma1.shape == sigma2.shape, (
        "Training and test covariances have different dimensions"
    )

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


class FID:
    """[Fréchet Inception Distance (FID)](https://arxiv.org/abs/1706.08500) metric.

    ### Args
        - `device` (Device): Device to run the model on.

    ### Attributes
        - `device` (Device): Device to run the model on.
        - `model` (InceptionFeatureExtractor): Inception model for feature extraction.
        - `transform` (Compose): Image preprocessing transforms.

    ### Example
    ```python
    >>> fid = FID(device="cuda")
    >>> score = fid(images1, images2)  # Calculate FID between two sets of images
    ```
    """

    def __init__(self, device: Device) -> None:
        self.device = device
        self.model = InceptionFeatureExtractor(self.device)
        self.transform = T.Compose(
            [
                T.Resize((299, 299)),
                T.ToTensor(),
                T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
            ]
        )

    def __call__(self, images1: List[Image], images2: List[Image]) -> float:
        f1 = self.model(torch.stack([self.transform(img) for img in images1]))
        f2 = self.model(torch.stack([self.transform(img) for img in images2]))
        return _calculate_fid(f1, f2)
