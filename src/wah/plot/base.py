import matplotlib.pyplot as plt

from ..typing import (
    Axes,
    Iterable,
    Optional,
    Path,
    Tuple,
)

__all__ = [
    "Plot2D",
]


class Plot2D:
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
        grid_alpha: Optional[float] = 0.0,
    ) -> None:
        self.figsize = figsize
        self.fontsize = fontsize
        self.title = title

        # x
        self.xlabel = xlabel
        self.xlim = xlim
        self.xticks = xticks
        self.xticklabels = (
            [str(x) for x in xticks]
            if xticks is not None and yticklabels is None
            else xticklabels
        )

        # y
        self.ylabel = ylabel
        self.ylim = ylim
        self.yticks = yticks
        self.yticklabels = (
            [str(y) for y in yticks]
            if yticks is not None and yticklabels is None
            else yticklabels
        )

        self.grid_alpha = grid_alpha

    def _plot(self, ax: Axes, *args, **kwargs) -> None:
        raise NotImplementedError

    def plot(self, *args, **kwargs) -> None:
        if self.fontsize is not None:
            plt.rcParams.update({"font.size": self.fontsize})

        _, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=self.figsize,
        )
        ax.set_title(self.title)

        self._plot(ax, *args, **kwargs)

        ax.set_xlabel(self.xlabel)
        if self.xlim is not None:
            ax.set_xlim(*self.xlim)
        if self.xticks is not None:
            ax.set_xticks(self.xticks, self.xticklabels)

        ax.set_ylabel(self.ylabel)
        if self.ylim is not None:
            ax.set_ylim(*self.ylim)
        if self.yticks is not None:
            ax.set_yticks(self.yticks, self.yticklabels)

        ax.grid(alpha=self.grid_alpha)

    def show(
        self,
    ) -> None:
        plt.show()

    def save(
        self,
        save_path: Path,
        use_auto_settings: bool = True,
        **kwargs,
    ) -> None:
        if use_auto_settings:
            kwargs["dpi"] = 300
            kwargs["bbox_inches"] = "tight"
            kwargs["pad_inches"] = 0.01

        plt.savefig(save_path, **kwargs)
        plt.draw()
        plt.close("all")
