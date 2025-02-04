# https://github.com/pytorch/vision/blob/main/references/classification/presets.py
# https://github.com/pytorch/vision/blob/main/references/classification/transforms.py

import torch
import torchvision.transforms.v2 as T
from torchvision.transforms.functional import InterpolationMode

from ...misc.typing import Sequence

__all__ = [
    "ClassificationPresetTrain",
    "ClassificationPresetEval",
    "DeNormalize",
]


def get_mixup_cutmix(
    *,
    mixup_alpha,
    cutmix_alpha,
    num_classes,
):
    mixup_cutmix = []
    if mixup_alpha > 0:
        mixup_cutmix.append(T.MixUp(alpha=mixup_alpha, num_classes=num_classes))
    if cutmix_alpha > 0:
        mixup_cutmix.append(T.CutMix(alpha=cutmix_alpha, num_classes=num_classes))
    if not mixup_cutmix:
        return None

    return T.RandomChoice(mixup_cutmix)


class ClassificationPresetTrain:
    # Note: this transform assumes that the input to forward() are always PIL
    # images, regardless of the backend parameter. We may change that in the
    # future though, if we change the output type from the dataset.
    def __init__(self, **kwargs):
        crop_size: int = kwargs.get("crop_size", 224)
        mean: Sequence[float] = kwargs.get("mean", None)
        std: Sequence[float] = kwargs.get("std", None)
        interpolation: str = kwargs.get("interpolation", "bilinear")
        hflip_prob: float = kwargs.get("hflip_prob", 0.5)
        auto_augment_policy: str = kwargs.get("auto_augment", None)
        ra_magnitude: int = kwargs.get("ra_magnitude", 9)
        augmix_severity: int = kwargs.get("augmix_severity", 3)
        random_erase_prob: float = kwargs.get("random_erase", 0.0)
        backend: str = kwargs.get("backend", "pil")

        interpolation = getattr(InterpolationMode, interpolation.upper())

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms.append(
            T.RandomResizedCrop(crop_size, interpolation=interpolation, antialias=True)
        )
        if hflip_prob > 0:
            transforms.append(T.RandomHorizontalFlip(hflip_prob))
        if auto_augment_policy is not None:
            if auto_augment_policy == "ra":
                transforms.append(
                    T.RandAugment(interpolation=interpolation, magnitude=ra_magnitude)
                )
            elif auto_augment_policy == "ta_wide":
                transforms.append(T.TrivialAugmentWide(interpolation=interpolation))
            elif auto_augment_policy == "augmix":
                transforms.append(
                    T.AugMix(interpolation=interpolation, severity=augmix_severity)
                )
            else:
                aa_policy = T.AutoAugmentPolicy(auto_augment_policy)
                transforms.append(
                    T.AutoAugment(policy=aa_policy, interpolation=interpolation)
                )

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms.append(T.ToDtype(torch.float, scale=True))

        if mean is not None and std is not None:
            transforms.append(T.Normalize(mean=mean, std=std))

        if random_erase_prob > 0:
            transforms.append(T.RandomErasing(p=random_erase_prob))

        transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class ClassificationPresetEval:
    def __init__(self, **kwargs):
        crop_size: int = kwargs.get("crop_size", 224)
        resize_size: int = kwargs.get("resize_size", 256)
        mean: Sequence[float] = kwargs.get("mean", None)
        std: Sequence[float] = kwargs.get("std", None)
        interpolation: str = kwargs.get("interpolation", "bilinear")
        backend: str = kwargs.get("backend", "pil")

        interpolation = getattr(InterpolationMode, interpolation.upper())

        transforms = []
        backend = backend.lower()
        if backend == "tensor":
            transforms.append(T.PILToTensor())
        elif backend != "pil":
            raise ValueError(f"backend can be 'tensor' or 'pil', but got {backend}")

        transforms += [
            T.Resize(resize_size, interpolation=interpolation, antialias=True),
            T.CenterCrop(crop_size),
        ]

        if backend == "pil":
            transforms.append(T.PILToTensor())

        transforms.append(T.ToDtype(torch.float, scale=True))

        if mean is not None and std is not None:
            transforms.append(T.Normalize(mean=mean, std=std))

        transforms.append(T.ToPureTensor())

        self.transforms = T.Compose(transforms)

    def __call__(self, img):
        return self.transforms(img)


class DeNormalize(T.Normalize):
    """Reverse the normalization transform.

    ### Args
        - `mean` (Sequence[float]): Mean values used in original normalization
        - `std` (Sequence[float]): Standard deviation values used in original normalization

    ### Example
    ```python
    >>> denormalize = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    >>> normalized_img = transforms.Normalize(...)(img)
    >>> original_img = denormalize(normalized_img)  # Reverses normalization
    ```
    """

    def __init__(
        self,
        mean: Sequence[float],
        std: Sequence[float],
    ) -> None:
        """
        - `mean` (Sequence[float]): Mean values used in original normalization
        - `std` (Sequence[float]): Standard deviation values used in original normalization
        """
        self._mean = mean
        self._std = std
        # Reverse the operation of normalization: de-normalization
        reversed_mean = [-m / s for m, s in zip(mean, std)]
        reversed_std = [1 / s for s in std]
        super().__init__(mean=reversed_mean, std=reversed_std)

    def __repr__(self) -> str:
        return f"DeNormalize(mean={tuple(self._mean)}, std={tuple(self._std)})"
