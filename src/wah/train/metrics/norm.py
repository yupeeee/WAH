import math

import torch
from torchmetrics.aggregation import MeanMetric

from ...typing import (
    Tensor,
    Union,
)

__all__ = [
    "L2Avg",
    "L2Std",
    "RMSAvg",
    "RMSStd",
]


def l2(x: Tensor, ) -> Tensor:
    x = x.reshape(len(x), -1)

    return torch.norm(x, p=2, dim=-1)


def rms(x: Tensor, ) -> Tensor:
    x = x.reshape(len(x), -1)

    return torch.norm(x, p=2, dim=-1) / math.sqrt(x.size(-1))


class L2Avg(MeanMetric):
    label = "l2avg"

    def __init__(self, ) -> None:
        super().__init__()

    def update(
        self,
        value: Union[float, Tensor],
        weight: Union[float, Tensor] = 1.0,
    ) -> None:
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = torch.as_tensor(
                value, dtype=self.dtype, device=self.device)

        if weight is not None and not isinstance(weight, Tensor):
            weight = torch.as_tensor(
                weight, dtype=self.dtype, device=self.device)

        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)

        # use l2 for value
        value = l2(value)

        if value.numel() == 0:
            return

        self.mean_value += (value * weight).sum()
        self.weight += weight.sum()


class L2Std(MeanMetric):
    label = "l2std"

    def __init__(self, ) -> None:
        super().__init__()

        self.l2_avg = L2Avg()
        self.sq_l2_avg = L2Avg()

    def update(
        self,
        value: Union[float, Tensor],
        weight: Union[float, Tensor] = 1.0,
    ) -> None:
        self.l2_avg.update(value, weight)
        self.sq_l2_avg.update(value ** 2, weight)

    def compute(self, ) -> Tensor:
        return self.sq_l2_avg.compute() - self.l2_avg.compute() ** 2


class RMSAvg(MeanMetric):
    label = "rmsavg"

    def __init__(self, ) -> None:
        super().__init__()

    def update(
        self,
        value: Union[float, Tensor],
        weight: Union[float, Tensor] = 1.0,
    ) -> None:
        # broadcast weight to value shape
        if not isinstance(value, Tensor):
            value = torch.as_tensor(
                value, dtype=self.dtype, device=self.device)

        if weight is not None and not isinstance(weight, Tensor):
            weight = torch.as_tensor(
                weight, dtype=self.dtype, device=self.device)

        weight = torch.broadcast_to(weight, value.shape)
        value, weight = self._cast_and_nan_check_input(value, weight)

        # use rms for value
        value = rms(value)

        if value.numel() == 0:
            return

        self.mean_value += (value * weight).sum()
        self.weight += weight.sum()


class RMSStd(MeanMetric):
    label = "rmsstd"

    def __init__(self, ) -> None:
        super().__init__()

        self.rms_avg = RMSAvg()
        self.sq_rms_avg = RMSAvg()

    def update(
        self,
        value: Union[float, Tensor],
        weight: Union[float, Tensor] = 1.0,
    ) -> None:
        self.rms_avg.update(value, weight)
        self.sq_rms_avg.update(value ** 2, weight)

    def compute(self, ) -> Tensor:
        return self.sq_rms_avg.compute() - self.rms_avg.compute() ** 2
