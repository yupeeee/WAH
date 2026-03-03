import os
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar

__all__ = [
    "Plot",
]


class Plot:
    """Thin wrapper around matplotlib Figure and Axes for easier indexing and saving.

    Axes are always exposed as a 1D array so `plot.axes[i]` works for any grid.
    For a single subplot, `plot.ax` is a shortcut for `plot.axes[0]`.

    ### Example
    ```python
    plot = Plot(2, 2, figsize=(8, 6))
    sc = plot.axes[0].scatter(x, y)
    plot.axes[1].plot(t, f)
    plot.add_colorbar(sc, ticks=[0, 1], ticklabels=["low", "high"])
    plot.save("out.png")
    ```
    """

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
    ) -> None:
        fig, ax_array = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
        self.fig: plt.Figure = fig
        # Always 1D so plot.axes[0] works for single or multi panel
        self._axes_2d: np.ndarray = ax_array
        self.axes: np.ndarray = ax_array.flatten()

    @property
    def ax(self) -> plt.Axes:
        """Single axis when grid is 1x1; use plot.ax instead of plot.axes[0]."""
        if self.axes.size != 1:
            raise AttributeError(
                "Plot.ax is only available for single-panel figures; use plot.axes[i]"
            )
        return self.axes[0]

    def add_colorbar(
        self,
        mappable: ScalarMappable,
        ticks: Optional[Union[List[float], np.ndarray]] = None,
        ticklabels: Optional[List[str]] = None,
        alpha: float = 1.0,
        **kwargs,
    ) -> Colorbar:
        """Add a colorbar for the given mappable (e.g. from scatter(..., c=...), imshow, pcolormesh).

        The axes to attach the colorbar to are taken from `mappable.axes`. The colorbar
        is drawn with the given `alpha` and is not affected by the plot axes' transparency.

        ### Args
            - `mappable`: ScalarMappable (e.g. return value of scatter/imshow/pcolormesh with c=).
            - `ticks`: Optional tick positions (data coordinates along the colorbar).
            - `ticklabels`: Optional list of strings for tick labels (same length as `ticks`).
            - `alpha`: Opacity of the colorbar background (1.0 = opaque).
            - `**kwargs`: Passed to `fig.colorbar` (e.g. label=..., orientation='horizontal', shrink=...).

        ### Returns
            The created Colorbar (use cbar.set_ticks / cbar.set_ticklabels for more control).
        """
        assert hasattr(mappable, "axes"), "Mappable must have an axes attribute"
        cbar = self.fig.colorbar(mappable, ax=mappable.axes, **kwargs)
        if ticks is not None:
            cbar.set_ticks(ticks)
        if ticklabels is not None:
            cbar.set_ticklabels(ticklabels)
        # Keep colorbar opaque (or explicitly alpha) so plot transparency doesn't affect it
        cbar.ax.patch.set_alpha(alpha)
        cbar.ax.patch.set_facecolor(plt.rcParams.get("figure.facecolor", "white"))
        return cbar

    def save(
        self,
        path: Union[str, os.PathLike],
        dpi: int = 300,
        bbox_inches: str = "tight",
        pad_inches: float = 0.05,
        close: bool = True,
        **kwargs,
    ) -> None:
        """Save figure to file."""
        self.fig.savefig(
            path,
            dpi=dpi,
            bbox_inches=bbox_inches,
            pad_inches=pad_inches,
            **kwargs,
        )
        if close:
            self.close()

    def show(self) -> None:
        """Display the figure (e.g. in notebooks or GUI)."""
        plt.show(block=False)

    def close(self) -> None:
        """Close the figure and release resources."""
        plt.close(self.fig)

    def __enter__(self) -> "Plot":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
