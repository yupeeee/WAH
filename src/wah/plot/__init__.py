import os
from pathlib import Path
from typing import Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colorbar import Colorbar
from PIL import Image

__all__ = [
    "Plot",
    "make_gif",
]


class Plot:
    """Thin wrapper around matplotlib Figure and Axes for easier indexing and saving.

    Axes are always exposed as a 1D array so `plot.axes[i]` works for any grid.
    For a single subplot, `plot.ax` is a shortcut for `plot.axes[0]`.

    ### Example
    ```python
    plot = Plot(2, 2, figsize=(8, 6))
    sc = plot.axes[0].scatter(x, y, c=z)
    plot.axes[1].plot(t, f)
    cbar = plot.add_colorbar(sc, label="colorbar label")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["low", "high"])
    plot.save("out.png")
    ```
    """

    def __init__(
        self,
        nrows: int = 1,
        ncols: int = 1,
        figsize: Optional[Tuple[float, float]] = None,
        fontsize: Optional[float] = None,
        font: Optional[str] = None,
        mathfont: Optional[str] = None,
        **kwargs,
    ) -> None:
        # Note: plt.rcParams.update() must be called before plt.subplots()
        if fontsize is not None:
            plt.rcParams.update({"font.size": fontsize})
        if font is not None:
            plt.rcParams.update({"font.family": font})
        if mathfont is not None:
            plt.rcParams.update({"mathtext.fontset": mathfont})

        fig, ax_array = plt.subplots(
            nrows, ncols, figsize=figsize, squeeze=False, **kwargs
        )
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
        _mappable: ScalarMappable,
        clim: Optional[Tuple[float, float]] = None,
        cmap: Optional[str] = None,
        **kwargs,
    ) -> Colorbar:
        """Add a colorbar for the given mappable (e.g. from scatter(..., c=...), imshow, pcolormesh).

        The axes to attach the colorbar to are taken from `mappable.axes`.

        ### Args
            - `mappable`: ScalarMappable (e.g. return value of scatter/imshow/pcolormesh with c=).
            - `**kwargs`: Passed to `fig.colorbar` (e.g. label=..., orientation='horizontal', shrink=...).

        ### Returns
            The created Colorbar (use cbar.set_ticks / cbar.set_ticklabels for more control).
        """
        assert hasattr(_mappable, "axes"), "_mappable must have an axes attribute"
        ax = _mappable.axes
        vmin, vmax = clim if clim is not None else _mappable.get_clim()
        cmap = cmap if cmap is not None else _mappable.get_cmap()
        mappable = ScalarMappable(
            norm=plt.Normalize(vmin=vmin, vmax=vmax),
            cmap=cmap,
        )
        cbar = self.fig.colorbar(mappable, ax=ax, **kwargs)
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


def make_gif(
    path: Union[str, os.PathLike],
    duration: int = 100,
    loop: int = 0,
    pattern: str = "*.png",
    output: Optional[Union[str, os.PathLike]] = None,
) -> None:
    """Make an animated GIF from images in a directory.

    If `path` is a directory, frames are read from that directory.
    If `path` ends with ".gif", frames are read from its parent directory and `path` is used as output.
    """
    p = Path(path)
    if output is None:
        if p.suffix.lower() == ".gif":
            frames_dir = p.parent
            output_path = p
        else:
            frames_dir = p
            output_path = p / "animation.gif"
    else:
        frames_dir = p
        output_path = Path(output)

    frame_paths = sorted(frames_dir.glob(pattern))
    if not frame_paths:
        raise FileNotFoundError(
            f"No frames found in '{frames_dir}' matching pattern '{pattern}'"
        )

    frames = []
    for fp in frame_paths:
        with Image.open(fp) as im:
            frames.append(im.convert("RGBA").copy())

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=loop,
    )
