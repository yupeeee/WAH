from torch.utils.data.dataloader import default_collate
from torchvision.transforms import v2

from ..typing import Transform

__all__ = [
    "CollateFunction",
]


def get_mixup_cutmix(
    mixup_alpha: float,
    cutmix_alpha: float,
    num_classes: int,
) -> Transform:
    mixup_cutmix = []

    if mixup_alpha > 0.:
        mixup = v2.MixUp(
            alpha=mixup_alpha,
            num_classes=num_classes,
        )
        mixup_cutmix.append(mixup)

    if cutmix_alpha > 0.:
        cutmix = v2.CutMix(
            alpha=cutmix_alpha,
            num_classes=num_classes,
        )
        mixup_cutmix.append(cutmix)

    if not mixup_cutmix:
        return None

    return v2.RandomChoice(mixup_cutmix)


class CollateFunction:
    def __init__(
        self,
        mixup_alpha: float,
        cutmix_alpha: float,
        num_classes: int,
    ) -> None:
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
