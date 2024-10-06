import matplotlib.pyplot as plt

from .. import path as _path
from ..typing import Axes, Figure, Iterable, Optional, Path, Tuple

__all__ = [
    "Plot2D",
]


class Plot2D:
    """
    A class for creating customizable 2D plots using matplotlib.

    ### Attributes
    - `figsize` (Tuple[float, float], optional): Size of the figure.
    - `fontsize` (float, optional): Font size for the plot text.
    - `title` (str, optional): Title of the plot.

    - `xlabel` (str, optional): Label for the x-axis.
    - `xlim` (Tuple[float, float], optional): Limits for the x-axis.
    - `xticks` (Iterable[float], optional): Tick positions on the x-axis.
    - `xticklabels` (Iterable[str], optional): Labels for the x-axis ticks.

    - `ylabel` (str, optional): Label for the y-axis.
    - `ylim` (Tuple[float, float], optional): Limits for the y-axis.
    - `yticks` (Iterable[float], optional): Tick positions on the y-axis.
    - `yticklabels` (Iterable[str], optional): Labels for the y-axis ticks.

    - `grid_alpha` (float, optional): Transparency of the grid. Defaults to `0.0`.

    ### Methods
    - `plot(*args, **kwargs)`: Generates and customizes the plot with provided data and parameters.
    - `show()`: Displays the plot.
    - `save(save_path: Path, use_auto_settings: bool = True, **kwargs)`: Saves the plot to a specified path.
    """

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
        """
        - `figsize` (Tuple[float, float], optional): Figure size.
        - `fontsize` (float, optional): Font size for the plot text.
        - `title` (str, optional): Title of the plot.

        - `xlabel` (str, optional): X-axis label.
        - `xlim` (Tuple[float, float], optional): X-axis limits.
        - `xticks` (Iterable[float], optional): X-axis tick positions.
        - `xticklabels` (Iterable[str], optional): X-axis tick labels.

        - `ylabel` (str, optional): Y-axis label.
        - `ylim` (Tuple[float, float], optional): Y-axis limits.
        - `yticks` (Iterable[float], optional): Y-axis tick positions.
        - `yticklabels` (Iterable[str], optional): Y-axis tick labels.

        - `grid_alpha` (float, optional): Alpha (transparency) for the grid. Defaults to `0.0`.
        """
        self.figsize = figsize
        self.fontsize = fontsize
        self.title = title

        # x
        self.xlabel = xlabel
        self.xlim = xlim
        self.xticks = xticks
        self.xticklabels = (
            [str(x) for x in xticks]
            if xticks is not None and xticklabels is None
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

    def _plot(self, fig: Figure, ax: Axes, *args, **kwargs) -> None:
        """
        Abstract method for plotting, intended to be implemented by subclasses.

        ### Parameters
        - `fig` (Figure): The matplotlib figure object.
        - `ax` (Axes): The matplotlib axes object.
        - `*args`: Additional arguments for plotting.
        - `**kwargs`: Additional keyword arguments for plotting.

        ### Raises
        - `NotImplementedError`: This method must be implemented in subclasses.
        """
        raise NotImplementedError

    def plot(self, *args, **kwargs) -> None:
        """
        Generates the plot based on the provided data and customization parameters.

        ### Parameters
        - `*args`: Data for plotting.
        - `**kwargs`: Additional keyword arguments to customize the plot.
        """
        params = [p for p in dir(self) if p[0] != "_" and p != "plot"]

        for kw in list(kwargs.keys()):
            if kw in params:
                setattr(self, kw, kwargs[kw])
                del kwargs[kw]

        if self.fontsize is not None:
            plt.rcParams.update({"font.size": self.fontsize})

        fig, ax = plt.subplots(
            nrows=1,
            ncols=1,
            figsize=self.figsize,
        )
        ax.set_title(self.title)

        self._plot(fig, ax, *args, **kwargs)

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

    def show(self) -> None:
        """
        Displays the generated plot.
        """
        plt.show()

    def save(
        self,
        save_path: Path,
        use_auto_settings: bool = True,
        **kwargs,
    ) -> None:
        """
        Saves the plot to the specified file path.

        ### Parameters
        - `save_path` (Path): The path where the plot will be saved.
        - `use_auto_settings` (bool, optional): Whether to apply default saving settings (DPI, tight layout). Defaults to `True`.
        - `**kwargs`: Additional arguments for `plt.savefig`.
        """
        if use_auto_settings:
            kwargs["dpi"] = 300
            kwargs["bbox_inches"] = "tight"
            kwargs["pad_inches"] = 0.01

        save_path = _path.clean(save_path)
        save_dir, _ = _path.split(save_path)
        _path.mkdir(save_dir)

        plt.savefig(save_path, **kwargs)
        plt.draw()
        plt.close("all")
