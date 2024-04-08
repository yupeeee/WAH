import math

import torch
from torch.utils.data.dataloader import default_collate
from torchvision.transforms import functional as F

from ..typing import Tensor, Tuple

__all__ = [
    "CollateFunction",
]


def get_module(use_v2: bool, ):
    # Protected import to avoid the V2 warning
    # in case just V1 is used
    if use_v2:
        import torchvision.transforms.v2

        return torchvision.transforms.v2

    else:
        import torchvision.transforms

        return torchvision.transforms


class RandomMixUp(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Expected a positive integer for num_classes, got {num_classes}"
            )

        if alpha <= 0:
            raise ValueError(
                f"Expected alpha > 0, got {alpha}"
            )

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(
        self,
        batch: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if batch.ndim != 4:
            raise ValueError(
                f"Expected dim(batch) == 4, got {batch.ndim}"
            )

        if target.ndim != 1:
            raise ValueError(
                f"Expected dim(target) == 1, got {target.ndim}"
            )

        if not batch.is_floating_point():
            raise TypeError(
                f"Expected dtype(batch) as a float tensor, got {batch.dtype}"
            )

        if target.dtype != torch.int64:
            raise TypeError(
                f"Expected dtype(target) == torch.int64, got {target.dtype}"
            )

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, self.num_classes).to(
                dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one
        # instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on mixup paper, page 3
        lambda_param = float(torch._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha]))[0])

        batch_rolled.mul_(1.0 - lambda_param)
        batch.mul_(lambda_param).add_(batch_rolled)

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"p={self.p}, "
            f"alpha={self.alpha}, "
            f"inplace={self.inplace}"
            f")"
        )

        return s


class RandomCutMix(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        p: float = 0.5,
        alpha: float = 1.0,
        inplace: bool = False,
    ) -> None:
        super().__init__()

        if num_classes < 1:
            raise ValueError(
                f"Expected a positive integer for num_classes, got {num_classes}"
            )

        if alpha <= 0:
            raise ValueError(
                f"Expected alpha > 0, got {alpha}"
            )

        self.num_classes = num_classes
        self.p = p
        self.alpha = alpha
        self.inplace = inplace

    def forward(
        self,
        batch: Tensor,
        target: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        if batch.ndim != 4:
            raise ValueError(
                f"Expected dim(batch) == 4, got {batch.ndim}"
            )

        if target.ndim != 1:
            raise ValueError(
                f"Expected dim(target) == 1, got {target.ndim}"
            )

        if not batch.is_floating_point():
            raise TypeError(
                f"Expected dtype(batch) as a float tensor, got {batch.dtype}"
            )

        if target.dtype != torch.int64:
            raise TypeError(
                f"Expected dtype(target) == torch.int64, got {target.dtype}"
            )

        if not self.inplace:
            batch = batch.clone()
            target = target.clone()

        if target.ndim == 1:
            target = torch.nn.functional.one_hot(
                target, self.num_classes).to(
                dtype=batch.dtype)

        if torch.rand(1).item() >= self.p:
            return batch, target

        # It's faster to roll the batch by one
        # instead of shuffling it to create image pairs
        batch_rolled = batch.roll(1, 0)
        target_rolled = target.roll(1, 0)

        # Implemented as on cutmix paper, page 12
        # (with minor corrections on typos)
        lambda_param = float(torch._sample_dirichlet(
            torch.tensor([self.alpha, self.alpha]))[0])
        _, H, W = F.get_dimensions(batch)

        r_x = torch.randint(W, (1,))
        r_y = torch.randint(H, (1,))

        r = 0.5 * math.sqrt(1.0 - lambda_param)
        r_w_half = int(r * W)
        r_h_half = int(r * H)

        x1 = int(torch.clamp(r_x - r_w_half, min=0))
        y1 = int(torch.clamp(r_y - r_h_half, min=0))
        x2 = int(torch.clamp(r_x + r_w_half, max=W))
        y2 = int(torch.clamp(r_y + r_h_half, max=H))

        batch[:, :, y1:y2, x1:x2] = batch_rolled[:, :, y1:y2, x1:x2]
        lambda_param = float(1.0 - (x2 - x1) * (y2 - y1) / (W * H))

        target_rolled.mul_(1.0 - lambda_param)
        target.mul_(lambda_param).add_(target_rolled)

        return batch, target

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}("
            f"num_classes={self.num_classes}, "
            f"p={self.p}, "
            f"alpha={self.alpha}, "
            f"inplace={self.inplace}"
            f")"
        )

        return s


def get_mixup_cutmix(
    *,
    mixup_alpha: float,
    cutmix_alpha: float,
    num_classes: int,
    use_v2: bool,
):
    transforms_module = get_module(use_v2)

    mixup_cutmix = []

    if mixup_alpha > 0:
        mixup_cutmix.append(
            transforms_module.MixUp(
                alpha=mixup_alpha,
                num_classes=num_classes,
            ) if use_v2
            else RandomMixUp(
                num_classes=num_classes,
                p=1.0,
                alpha=mixup_alpha,
            )
        )

    if cutmix_alpha > 0:
        mixup_cutmix.append(
            transforms_module.CutMix(
                alpha=mixup_alpha,
                num_classes=num_classes,
            ) if use_v2
            else RandomCutMix(
                num_classes=num_classes,
                p=1.0,
                alpha=mixup_alpha,
            )
        )

    if not mixup_cutmix:
        return None

    return transforms_module.RandomChoice(mixup_cutmix)


class CollateFunction:
    def __init__(
        self,
        *,
        mixup_alpha: float,
        cutmix_alpha: float,
        num_classes: int,
        use_v2: bool,
    ) -> None:
        self.collate_fn = default_collate

        # mixup_cutmix
        mixup_cutmix = get_mixup_cutmix(
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            num_classes=num_classes,
            use_v2=use_v2,
        )

        if mixup_cutmix is not None:
            def collate_fn(batch):
                return mixup_cutmix(*default_collate(batch))

            self.collate_fn = collate_fn

    def __call__(self, batch):
        return self.collate_fn(batch)
