import matplotlib.pyplot as plt

from .. import path as _path
from ..typing import Axes, Figure, Iterable, Optional, Path, Tuple

__all__ = [
    "Plot2D",
]


class Plot2D:
    """
    A class for creating 2D plots with customizable settings.

    ### Attributes:
    - `figsize (Optional[Tuple[float, float]])`: The size of the figure.
    - `fontsize (Optional[float])`: The font size for the plot text.
    - `title (Optional[str])`: The title of the plot.
    - `xlabel (Optional[str])`: The label for the x-axis.
    - `xlim (Optional[Tuple[float, float]])`: The limits for the x-axis.
    - `xticks (Optional[Iterable[float]])`: The tick values for the x-axis.
    - `xticklabels (Optional[Iterable[str]])`: The tick labels for the x-axis.
    - `ylabel (Optional[str])`: The label for the y-axis.
    - `ylim (Optional[Tuple[float, float]])`: The limits for the y-axis.
    - `yticks (Optional[Iterable[float]])`: The tick values for the y-axis.
    - `yticklabels (Optional[Iterable[str]])`: The tick labels for the y-axis.
    - `grid_alpha (Optional[float])`: The alpha transparency for the grid.

    ### Methods:
    - `_plot(fig: Figure, ax: Axes, *args, **kwargs) -> None`: Abstract method to be implemented by subclasses for specific plotting logic.
    - `plot(*args, **kwargs) -> None`: Creates the plot with the specified settings.
    - `show() -> None`: Displays the plot.
    - `save(save_path: Path, use_auto_settings: bool = True, **kwargs) -> None`: Saves the plot to the specified path with optional settings.
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
        - `figsize (Optional[Tuple[float, float]])`: The size of the figure. Defaults to None.
        - `fontsize (Optional[float])`: The font size for the plot text. Defaults to None.
        - `title (Optional[str])`: The title of the plot. Defaults to None.
        - `xlabel (Optional[str])`: The label for the x-axis. Defaults to None.
        - `xlim (Optional[Tuple[float, float]])`: The limits for the x-axis. Defaults to None.
        - `xticks (Optional[Iterable[float]])`: The tick values for the x-axis. Defaults to None.
        - `xticklabels (Optional[Iterable[str]])`: The tick labels for the x-axis. Defaults to None.
        - `ylabel (Optional[str])`: The label for the y-axis. Defaults to None.
        - `ylim (Optional[Tuple[float, float]])`: The limits for the y-axis. Defaults to None.
        - `yticks (Optional[Iterable[float]])`: The tick values for the y-axis. Defaults to None.
        - `yticklabels (Optional[Iterable[str]])`: The tick labels for the y-axis. Defaults to None.
        - `grid_alpha (Optional[float])`: The alpha transparency for the grid. Defaults to 0.0.
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
        Abstract method to be implemented by subclasses for specific plotting logic.

        ### Parameters:
        - `fig (Figure)`: The Matplotlib figure.
        - `ax (Axes)`: The Matplotlib axes to plot on.
        - `*args`: Additional arguments for plotting.
        - `**kwargs`: Additional keyword arguments for plotting.

        ### Raises:
        - `NotImplementedError`: If the method is not implemented by a subclass.

        ### Notes:
        - This method should be overridden in subclasses to define specific plotting behavior.
        """
        raise NotImplementedError

    def plot(self, *args, **kwargs) -> None:
        """
        Creates the plot with the specified settings.

        ### Parameters:
        - `*args`: Additional arguments for plotting.
        - `**kwargs`: Additional keyword arguments for plotting.

        ### Notes:
        - This method sets up the plot using the provided settings (e.g., title, labels, limits) and calls the `_plot` method to handle the actual plotting logic.
        - If any of the settings (e.g., figsize, fontsize) are passed via `kwargs`, they will override the instance attributes.
        - The method uses Matplotlib to create the figure and axes.
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
        Displays the plot.

        ### Returns:
        - `None`

        ### Notes:
        - This method calls `plt.show()` to display the plot in an interactive window.
        """
        plt.show()

    def save(
        self,
        save_path: Path,
        use_auto_settings: bool = True,
        **kwargs,
    ) -> None:
        """
        Saves the plot to the specified path with optional settings.

        ### Parameters:
        - `save_path (Path)`: The path to save the plot.
        - `use_auto_settings (bool)`: Whether to use automatic settings for saving the plot. Defaults to True.
        - `**kwargs`: Additional keyword arguments for saving the plot.

        ### Notes:
        - This method saves the plot to the specified `save_path` using Matplotlib's `plt.savefig()` function.
        - If `use_auto_settings` is True, the method automatically adjusts settings such as DPI and padding to optimize the saved plot.
        - The directory specified in `save_path` is created if it does not exist.
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
