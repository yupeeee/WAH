import matplotlib.pyplot as plt
import numpy as np

from .. import path as _path
from ..typing import (
    Axes,
    AxIndex,
    Colormap,
    Colornorm,
    Dict,
    Figure,
    Literal,
    NDArray,
    Optional,
    PathCollection,
    Sequence,
    Tuple,
    Union,
)
from .errorbar import _errorbar2d
from .hist import _hist2d, _ridge
from .image import _image
from .plot import _plot2d
from .scatter import _scatter2d
from .utils import _get_ax, _init_colors, _show_colorbar

__all__ = [
    "Plot",
]


class Plot:
    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Sequence[int] = None,
        **kwargs,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.figsize = figsize
        self.params = dict((k.replace("_", "."), v) for k, v in kwargs.items())

        self.fig: Figure
        self.axes: Union[Axes, NDArray]
        self.plots: Dict[AxIndex, PathCollection]
        self.colors: Dict[AxIndex, Tuple[Colormap, Colornorm]]

        self._init()

    def _init(self) -> None:
        plt.rcParams.update(self.params)
        fig, axes = plt.subplots(
            nrows=self.nrows,
            ncols=self.ncols,
            figsize=self.figsize,
        )
        self.fig = fig
        self.axes: Union[Axes, NDArray] = axes
        self.plots: Dict[AxIndex, PathCollection] = dict()
        self.colors: Dict[AxIndex, Tuple[Colormap, Colornorm]] = dict()

    def reset(self) -> None:
        self._init()

    def save(
        self,
        path: str,
        **kwargs,
    ) -> None:
        if not kwargs:
            kwargs["dpi"] = 300
            kwargs["bbox_inches"] = "tight"
            kwargs["pad_inches"] = 0.05

        _path.mkdir(_path.dirname(path))
        plt.savefig(path, **kwargs)

    def show(self) -> None:
        plt.show()

    def close(self) -> None:
        plt.draw()
        plt.close("all")

    def ax_settings(
        self,
        index: AxIndex = 0,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        xlim: Optional[Tuple[int, int]] = None,
        xticks: Optional[Sequence[float]] = None,
        xticklabels: Optional[Sequence[str]] = None,
        xscale: Optional[str] = None,
        ylabel: Optional[str] = None,
        ylim: Optional[Tuple[int, int]] = None,
        yticks: Optional[Sequence[float]] = None,
        yticklabels: Optional[Sequence[str]] = None,
        yscale: Optional[str] = None,
        colorbar: Optional[bool] = False,
        clabel: Optional[str] = None,
        cticks: Optional[Sequence[float]] = None,
        cticklabels: Optional[Sequence[str]] = None,
        gridalpha: Optional[float] = None,
    ) -> None:
        ax = _get_ax(self.axes, index)

        # title
        if title is not None:
            ax.set_title(title)

        # x-axis
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if xlim is not None:
            ax.set_xlim(xlim)
        if xticks is not None:
            ax.set_xticks(xticks, labels=xticklabels)
        if xscale is not None:
            ax.set_xscale(xscale)

        # y-axis
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if ylim is not None:
            ax.set_ylim(ylim)
        if yticks is not None:
            ax.set_yticks(yticks, labels=yticklabels)
        if yscale is not None:
            ax.set_yscale(yscale)

        # colorbar
        if colorbar:
            _show_colorbar(
                self.fig, ax, *self.colors[index], clabel, cticks, cticklabels
            )

        # grid
        if gridalpha is not None:
            ax.grid(alpha=gridalpha)

    def errorbar2d(
        self,
        index: AxIndex = 0,
        x: Sequence[float] = ...,
        y: Sequence[float] = ...,
        yerr: Sequence[float] = ...,
        xerr: Optional[Sequence[float]] = None,
        c: Optional[Union[str, float]] = None,
        cmap: Optional[str] = "viridis",
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        ax = _get_ax(self.axes, index)

        if c is not None and not isinstance(c, str):
            c, colormap, colornorm = _init_colors(c, cmap, cmin, cmax)
            self.colors[index] = (colormap, colornorm)
            kwargs["c"] = c

        plot = _errorbar2d(self.fig, ax, x, y, yerr, xerr, *args, **kwargs)
        self.plots[index] = plot

    def hist2d(
        self,
        index: AxIndex = 0,
        x: Sequence[float] = ...,
        num_bins: float = ...,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        c: Optional[Union[str, float, Sequence[float]]] = None,
        cmap: Optional[str] = "viridis",
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        ax = _get_ax(self.axes, index)

        if c is not None and not isinstance(c, str):
            c, colormap, colornorm = _init_colors(c, cmap, cmin, cmax)
            self.colors[index] = (colormap, colornorm)
            kwargs["c"] = c

        plot = _hist2d(self.fig, ax, x, num_bins, xmin, xmax, *args, **kwargs)
        self.plots[index] = plot

    def image(
        self,
        index: AxIndex = 0,
        x: Sequence[float] = ...,
        data_shape: Literal[
            "CHW",
            "HWC",
            "HW",
        ] = "CHW",
        *args,
        **kwargs,
    ) -> None:
        assert data_shape in [
            "CHW",
            "HWC",
            "HW",
        ], f"Unsupported data_shape: {data_shape}"

        if data_shape == "CHW":
            x = x.permute(1, 2, 0)
        else:
            x = x

        ax = _get_ax(self.axes, index)

        plot = _image(self.fig, ax, x, *args, **kwargs)
        self.plots[index] = plot

        ax.axis("off")

    def images(
        self,
        x: Sequence[float] = ...,
        data_shape: Literal[
            "CHW",
            "HWC",
            "HW",
        ] = "CHW",
        *args,
        **kwargs,
    ) -> None:
        num_images = len(x)
        assert num_images <= len(
            self.axes.flatten()
        ), f"Number of images must be less than the number of subfigures ({len(self.axes.flatten())})."

        assert data_shape in [
            "CHW",
            "HWC",
            "HW",
        ], f"Unsupported data_shape: {data_shape}"

        if data_shape == "CHW":
            x = x.permute(0, 2, 3, 1)
        else:
            x = x

        for index in range(num_images):
            ax = _get_ax(self.axes.flatten(), index)

            plot = _image(self.fig, ax, x[index], *args, **kwargs)
            self.plots[index] = plot

        for ax in self.axes.flatten():
            ax.axis("off")

    def plot2d(
        self,
        index: AxIndex = 0,
        x: Sequence[float] = ...,
        y: Sequence[float] = ...,
        c: Optional[Union[str, float]] = None,
        cmap: Optional[str] = "viridis",
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        ax = _get_ax(self.axes, index)

        if c is not None and not isinstance(c, str):
            c, colormap, colornorm = _init_colors(c, cmap, cmin, cmax)
            self.colors[index] = (colormap, colornorm)
            kwargs["c"] = c

        plot = _plot2d(self.fig, ax, x, y, *args, **kwargs)
        self.plots[index] = plot

    def ridge(
        self,
        index: AxIndex = 0,
        xs: Sequence[Sequence[float]] = ...,
        ts: Sequence[float] = ...,
        tlabels: Sequence[str] = None,
        num_bins: int = ...,
        stride: Optional[int] = 1,
        overlap: Optional[float] = 0.01,
        alpha: Optional[float] = 0.8,
        cmap: Optional[str] = "viridis",
        *args,
        **kwargs,
    ) -> None:
        ax = _get_ax(self.axes, index)

        plot = _ridge(
            self.fig, ax, xs, num_bins, stride, overlap, alpha, cmap, *args, **kwargs
        )
        self.plots[index] = plot

        yticks = np.arange(len(xs) // stride + 1) * overlap
        indices = np.insert(np.arange(stride, len(xs) + stride, stride) - 1, 0, 0)[::-1]
        yticks = [float(yticks[np.where(indices == t)]) for t in ts]
        ax.set_yticks(yticks)
        ax.set_yticklabels(tlabels if tlabels is not None else [str(t) for t in ts])

    def scatter2d(
        self,
        index: AxIndex = 0,
        x: Sequence[float] = ...,
        y: Sequence[float] = ...,
        c: Optional[Union[str, float, Sequence[float]]] = None,
        cmap: Optional[str] = "viridis",
        cmin: Optional[float] = None,
        cmax: Optional[float] = None,
        *args,
        **kwargs,
    ) -> None:
        ax = _get_ax(self.axes, index)

        if c is not None and not isinstance(c, str):
            c, colormap, colornorm = _init_colors(c, cmap, cmin, cmax)
            self.colors[index] = (colormap, colornorm)
            kwargs["c"] = c

        plot = _scatter2d(self.fig, ax, x, y, *args, **kwargs)
        self.plots[index] = plot
