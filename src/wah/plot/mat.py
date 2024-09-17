import torch

from ..typing import Axes, Figure, Iterable, Optional, Tensor, Tuple
from .base import Plot2D

__all__ = [
    "MatShow2D",
]


class MatShow2D(Plot2D):
    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        fontsize: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        xticks: Optional[Iterable[float]] = None,
        xticklabels: Optional[Iterable[str]] = None,
        ylabel: Optional[str] = None,
        yticks: Optional[Iterable[float]] = None,
        yticklabels: Optional[Iterable[str]] = None,
        clabel: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        cticks: Optional[Tuple[float, float]] = None,
        cticklabels: Optional[Tuple[float, float]] = None,
    ) -> None:
        super().__init__(
            figsize=figsize,
            fontsize=fontsize,
            title=title,
            xlabel=xlabel,
            xlim=None,
            xticks=xticks,
            xticklabels=xticklabels,
            ylabel=ylabel,
            ylim=None,
            yticks=yticks,
            yticklabels=yticklabels,
            grid_alpha=0.0,
        )
        self.clabel = clabel
        self.clim = clim
        self.cticks = cticks
        self.cticklabels = (
            [str(c) for c in cticks]
            if cticks is not None and cticklabels is None
            else cticklabels
        )

    def _plot(
        self,
        fig: Figure,
        ax: Axes,
        mat: Tensor,
        cmap: Optional[str] = "viridis",
        *args,
        **kwargs,
    ) -> None:
        self.xlim = (0, mat.size(0) - 1)
        self.ylim = (0, mat.size(1) - 1)

        if self.clim is not None:
            cmin, cmax = self.clim
        else:
            cmin, cmax = torch.min(mat), torch.max(mat)
        plot = ax.matshow(mat, cmap=cmap, vmin=cmin, vmax=cmax, *args, **kwargs)
        cbar = fig.colorbar(plot, ax=ax)

        cbar.set_label(self.clabel)
        if self.cticks is not None:
            cbar.set_ticks(self.cticks)
            cbar.set_ticklabels(self.cticklabels)
