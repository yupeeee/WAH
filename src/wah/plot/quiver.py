import torch
from matplotlib import cm, colors

from ..typing import Axes, Figure, Iterable, Optional, Tensor, Tuple
from .base import Plot2D

__all__ = [
    "QuiverPlot2D",
    "TrajPlot2D",
]


class QuiverPlot2D(Plot2D):
    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        fontsize: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        xticks: Optional[Iterable[float]] = None,
        xticklabels: Optional[Iterable[str]] = None,
        ylabel: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        yticks: Optional[Iterable[float]] = None,
        yticklabels: Optional[Iterable[str]] = None,
        clabel: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        cticks: Optional[Tuple[float, float]] = None,
        cticklabels: Optional[Tuple[float, float]] = None,
        grid_alpha: Optional[float] = 0.0,
    ) -> None:
        super().__init__(
            figsize,
            fontsize,
            title,
            xlabel,
            xlim,
            xticks,
            xticklabels,
            ylabel,
            ylim,
            yticks,
            yticklabels,
            grid_alpha,
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
        x: Iterable[float],
        y: Iterable[float],
        vx: Iterable[float],
        vy: Iterable[float],
        scale: Optional[float] = 1.0,
        c: Optional[Iterable[float]] = None,
        cmap: Optional[str] = "viridis",
        *args,
        **kwargs,
    ) -> None:
        if c is not None:
            # normalize c in range [cmin, cmax]
            if self.clim is not None:
                cmin, cmax = self.clim
            else:
                cmin, cmax = min(c), max(c)
            cnorm = colors.Normalize(cmin, cmax)
            colormap = getattr(cm, cmap)
            c = colormap(cnorm(c))

            plot = ax.quiver(
                x,
                y,
                vx,
                vy,
                color=c,
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                *args,
                **kwargs,
            )
            cbar = fig.colorbar(plot, ax=ax)

            cbar.set_label(self.clabel)
            if self.cticks is not None:
                cbar.set_ticks(self.cticks)
                cbar.set_ticklabels(self.cticklabels)

        else:
            plot = ax.quiver(
                x,
                y,
                vx,
                vy,
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                *args,
                **kwargs,
            )


class TrajPlot2D(QuiverPlot2D):
    def __init__(
        self,
        figsize: Optional[Tuple[float, float]] = None,
        fontsize: Optional[float] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        xlim: Optional[Tuple[float, float]] = None,
        xticks: Optional[Iterable[float]] = None,
        xticklabels: Optional[Iterable[str]] = None,
        ylabel: Optional[str] = None,
        ylim: Optional[Tuple[float, float]] = None,
        yticks: Optional[Iterable[float]] = None,
        yticklabels: Optional[Iterable[str]] = None,
        clabel: Optional[str] = None,
        clim: Optional[Tuple[float, float]] = None,
        cticks: Optional[Tuple[float, float]] = None,
        cticklabels: Optional[Tuple[float, float]] = None,
        grid_alpha: Optional[float] = 0.0,
    ) -> None:
        super().__init__(
            figsize=figsize,
            fontsize=fontsize,
            title=title,
            xlabel=xlabel,
            xlim=xlim,
            xticks=xticks,
            xticklabels=xticklabels,
            ylabel=ylabel,
            ylim=ylim,
            yticks=yticks,
            yticklabels=yticklabels,
            clabel=clabel,
            clim=clim,
            cticks=cticks,
            cticklabels=cticklabels,
            grid_alpha=grid_alpha,
        )

    def _plot(
        self,
        fig: Figure,
        ax: Axes,
        traj: Tensor,
        scale: Optional[float] = 1.0,
        c: Optional[Iterable[float]] = None,
        cmap: Optional[str] = "viridis",
        *args,
        **kwargs,
    ) -> None:
        arrows = torch.diff(traj, dim=0)
        x, y = traj[:-1, 0], traj[:-1, 1]
        vx, vy = arrows[:, 0], arrows[:, 1]

        if c is not None:
            # normalize c in range [cmin, cmax]
            if self.clim is not None:
                cmin, cmax = self.clim
            else:
                cmin, cmax = min(c), max(c)
            cnorm = colors.Normalize(cmin, cmax)
            colormap = getattr(cm, cmap)
            c = colormap(cnorm(c))

            plot = ax.quiver(
                x,
                y,
                vx,
                vy,
                color=c,
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                *args,
                **kwargs,
            )
            cbar = fig.colorbar(plot, ax=ax)

            cbar.set_label(self.clabel)
            if self.cticks is not None:
                cbar.set_ticks(self.cticks)
                cbar.set_ticklabels(self.cticklabels)

        else:
            plot = ax.quiver(
                x,
                y,
                vx,
                vy,
                angles="xy",
                scale_units="xy",
                scale=1 / scale,
                *args,
                **kwargs,
            )
