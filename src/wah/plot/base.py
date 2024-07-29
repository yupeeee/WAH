import os

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
    """
    A class for creating 2D plots with customizable settings.

    ### Parameters
    - `figsize (Tuple[float, float], optional)`: The size of the figure. Defaults to None.
    - `fontsize (float, optional)`: The font size for the plot text. Defaults to None.
    - `title (str, optional)`: The title of the plot. Defaults to None.
    - `xlabel (str, optional)`: The label for the x-axis. Defaults to None.
    - `xlim (Tuple[float, float], optional)`: The limits for the x-axis. Defaults to None.
    - `xticks (Iterable[float], optional)`: The tick values for the x-axis. Defaults to None.
    - `xticklabels (Iterable[str], optional)`: The tick labels for the x-axis. Defaults to None.
    - `ylabel (str, optional)`: The label for the y-axis. Defaults to None.
    - `ylim (Tuple[float, float], optional)`: The limits for the y-axis. Defaults to None.
    - `yticks (Iterable[float], optional)`: The tick values for the y-axis. Defaults to None.
    - `yticklabels (Iterable[str], optional)`: The tick labels for the y-axis. Defaults to None.
    - `grid_alpha (float, optional)`: The alpha transparency for the grid. Defaults to 0.0.

    ### Methods
    - `plot`: Creates the plot with the specified settings.
    - `show`: Displays the plot.
    - `save`: Saves the plot to the specified path with optional settings.

    ### Notes
    - This class provides a flexible interface for creating and customizing 2D plots using Matplotlib.
    - The `_plot` method should be overridden by subclasses to implement specific plotting logic.
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
        - `figsize (Tuple[float, float], optional)`: The size of the figure. Defaults to None.
        - `fontsize (float, optional)`: The font size for the plot text. Defaults to None.
        - `title (str, optional)`: The title of the plot. Defaults to None.
        - `xlabel (str, optional)`: The label for the x-axis. Defaults to None.
        - `xlim (Tuple[float, float], optional)`: The limits for the x-axis. Defaults to None.
        - `xticks (Iterable[float], optional)`: The tick values for the x-axis. Defaults to None.
        - `xticklabels (Iterable[str], optional)`: The tick labels for the x-axis. Defaults to None.
        - `ylabel (str, optional)`: The label for the y-axis. Defaults to None.
        - `ylim (Tuple[float, float], optional)`: The limits for the y-axis. Defaults to None.
        - `yticks (Iterable[float], optional)`: The tick values for the y-axis. Defaults to None.
        - `yticklabels (Iterable[str], optional)`: The tick labels for the y-axis. Defaults to None.
        - `grid_alpha (float, optional)`: The alpha transparency for the grid. Defaults to 0.0.
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
        """
        Abstract method to be implemented by subclasses for specific plotting logic.

        ### Parameters
        - `ax (Axes)`: The Matplotlib axes to plot on.
        - `*args`: Additional arguments for plotting.
        - `**kwargs`: Additional keyword arguments for plotting.

        ### Raises
        - `NotImplementedError`: If the method is not implemented by a subclass.
        """
        raise NotImplementedError

    def plot(self, *args, **kwargs) -> None:
        """
        Creates the plot with the specified settings.

        ### Parameters
        - `*args`: Additional arguments for plotting.
        - `**kwargs`: Additional keyword arguments for plotting.

        ### Returns
        - `None`
        """
        params = [p for p in dir(self) if p[0] != "_" and p != "plot"]

        for kw in list(kwargs.keys()):
            if kw in params:
                setattr(self, kw, kwargs[kw])
                del kwargs[kw]

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

    def show(self) -> None:
        """
        Displays the plot.

        ### Parameters
        - `None`

        ### Returns
        - `None`
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

        ### Parameters
        - `save_path (Path)`: The path to save the plot.
        - `use_auto_settings (bool, optional)`: Whether to use automatic settings for saving the plot. Defaults to True.
        - `**kwargs`: Additional keyword arguments for saving the plot.

        ### Returns
        - `None`
        """
        if use_auto_settings:
            kwargs["dpi"] = 300
            kwargs["bbox_inches"] = "tight"
            kwargs["pad_inches"] = 0.01

        save_dir, _ = os.path.split(save_path)
        save_dir = os.path.normpath(save_dir)
        os.makedirs(save_dir, exist_ok=True)

        plt.savefig(save_path, **kwargs)
        plt.draw()
        plt.close("all")
