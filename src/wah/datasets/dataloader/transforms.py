from torch.utils.data.dataloader import default_collate
from torchvision.transforms import v2

from ...typing import (
    Transform,
)

__all__ = [
    "CollateFunction",
]


def get_mixup_cutmix(
    mixup_alpha: float,
    cutmix_alpha: float,
    num_classes: int,
) -> Transform:
    """
    Creates a transform for mixup and cutmix data augmentation based on the provided parameters.

    ### Parameters
    - `mixup_alpha` (float):
      The alpha parameter for mixup augmentation.
      A value greater than 0.0 enables mixup.
    - `cutmix_alpha` (float):
      The alpha parameter for cutmix augmentation.
      A value greater than 0.0 enables cutmix.
    - `num_classes` (int):
      The number of classes in the dataset.

    ### Returns
    - `Transform`:
      A composite transform that applies mixup and/or cutmix augmentations based on the specified alphas.
        - If both mixup_alpha and cutmix_alpha are 0.0, returns None.
        - If only one of mixup_alpha or cutmix_alpha is greater than 0.0, returns the corresponding transform.
        - If both mixup_alpha and cutmix_alpha are greater than 0.0, returns a random choice between mixup and cutmix transforms.
    """
    mixup_cutmix = []

    if mixup_alpha > 0.0:
        mixup = v2.MixUp(
            alpha=mixup_alpha,
            num_classes=num_classes,
        )
        mixup_cutmix.append(mixup)

    if cutmix_alpha > 0.0:
        cutmix = v2.CutMix(
            alpha=cutmix_alpha,
            num_classes=num_classes,
        )
        mixup_cutmix.append(cutmix)

    if not mixup_cutmix:
        return None

    return v2.RandomChoice(mixup_cutmix)


class CollateFunction:
    """
    A collate function class for data augmentation using MixUp and CutMix techniques.

    ### Attributes
    - `collate_fn` (callable):
      Default collate function used to combine a list of samples into a mini-batch.
    - `mixup_cutmix` (callable or None):
      Function to apply MixUp or CutMix augmentations, or None if neither is to be applied.

    ### Methods
    - `__call__`:
      Calls the collate function on the batch, applying MixUp or CutMix if applicable.
    """

    def __init__(
        self,
        mixup_alpha: float,
        cutmix_alpha: float,
        num_classes: int,
    ) -> None:
        """
        - `mixup_alpha` (float):
          Parameter for MixUp alpha value.
        - `cutmix_alpha` (float):
          Parameter for CutMix alpha value.
        - `num_classes` (int):
          Number of classes in the dataset.
        """
        self.collate_fn = default_collate

        self.mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            num_classes=num_classes,
        )

        if self.mixup_cutmix is not None:
            self.collate_fn = self._collate_fn

    def _collate_fn(self, batch):
        return self.mixup_cutmix(*default_collate(batch))

    def __call__(self, batch):
        return self.collate_fn(batch)
