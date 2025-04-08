import matplotlib.pyplot as plt
import numpy as np
import tqdm
from PIL import Image

from ..misc import path as _path
from ..misc.typing import Image as ImageType
from ..misc.typing import List, Optional, Path, Tensor, Tuple, Union

__all__ = [
    "denoising_gif",
    "denoising_traj_2d",
]


def denoising_gif(
    noises: List[ImageType],
    images: List[ImageType],
    resize: Optional[Tuple[int, int]] = None,
    fps: Optional[float] = 10.0,
    accel: Optional[float] = 0.0,
    min_duration: Optional[float] = 10.0,
    save_path: Path = "denoising.gif",
    verbose: bool = False,
) -> None:
    """Generates a GIF showing the denoising process of a diffusion model.

    ### Args
        - `noises` (List[Image]): List of noise predictions at each step
        - `images` (List[Image]): List of denoised images at each step
        - `resize` (Tuple[int, int], optional): Size to resize images to. Defaults to None.
        - `fps` (float, optional): Frames per second for the GIF. Defaults to 10.0.
        - `accel` (float, optional): Acceleration factor for frame duration. Defaults to 0.0.
            - `0.0`: Equal duration between frames
            - `>(<)0.0`: Exponentially faster(slower) towards the end. Slowest duration in miliseconds is 1000 / fps.
        - `min_duration` (float, optional): Minimum duration for each frame in milliseconds. Defaults to 10.0.
        - `save_path` (Path): Path to save the GIF to. Defaults to "denoising.gif".
        - `verbose` (bool): Whether to show progress bar. Defaults to False.

    ### Returns
        - `None`: Saves GIF to specified path

    ### Example
    ```python
    >>> noises = [noise1, noise2, noise3]  # List of noise prediction images
    >>> images = [image0, image1, image2, image3]  # List of denoised images
    >>> denoising_gif(noises, images, resize=(224, 224), fps=10.0, save_path="output.gif")
    # Creates output.gif showing the denoising process
    ```
    """
    assert len(noises) + 1 == len(images)
    noises = noises[::-1]
    images = images[::-1]

    if resize is not None:
        noises = [n.resize(resize) for n in noises]
        images = [i.resize(resize) for i in images]

    # Create figure once outside the loop
    plt.ioff()  # Turn off interactive mode
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    frames = []

    # Pre-configure axes to avoid repeated calls
    for ax in axes:
        ax.axis("off")

    # Draw the figure once to get the correct size
    plt.subplots_adjust(wspace=0.1, left=0, right=1, bottom=0.1, top=1)
    fig.canvas.draw()

    # Get the size of the figure in pixels
    width, height = fig.canvas.get_width_height()

    for i in tqdm.trange(
        len(noises),
        desc=f"Generating \033[1m{save_path}\033[0m",
        disable=not verbose,
    ):
        step_i = len(noises) - i

        # Clear previous content
        for ax in axes:
            ax.clear()
            ax.axis("off")

        # Update plots
        axes[0].imshow(images[i])
        axes[0].set_title(f"$x_{{{step_i}}}$", y=-0.1, fontsize=15)

        axes[1].imshow(noises[i])
        axes[1].set_title(f"$\epsilon_{{{step_i}}}$", y=-0.1, fontsize=15)

        axes[2].imshow(images[i + 1])
        axes[2].set_title(f"$x_{{{step_i - 1}}}$", y=-0.1, fontsize=15)

        # Draw and convert to image
        fig.canvas.draw()
        frame = Image.frombytes(
            "RGB",
            (width, height),  # Use consistent size
            fig.canvas.tostring_rgb(),
        )
        frames.append(frame)

    # Save as animated gif
    if accel == 0.0:
        durations = [1000 / fps] * len(frames)
    else:
        durations = 1000 / fps * np.linspace(0, 1, len(frames)) ** abs(accel)
        durations = durations.clip(min_duration, None)
        if accel > 0.0:
            durations = durations[::-1]
    durations = durations.tolist()

    _path.mkdir(_path.dirname(save_path))
    frames[0].save(
        save_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    plt.close("all")


def denoising_traj_2d(
    noise_preds_proj_2d: Union[List[Tensor], Tensor],
    latents_proj_2d: Union[List[Tensor], Tensor],
    noise_scale: float = 0.1,
    save_path: Path = "denoising_traj_2d.gif",
) -> None:
    """Plot 2D denoising trajectories.

    ### Args
        - `noise_preds_proj_2d` (Union[List[Tensor], Tensor]): Predicted noise tensors projected to 2D.
            Shape (num_steps, 2) or list of such tensors.
        - `latents_proj_2d` (Union[List[Tensor], Tensor]): Latent tensors projected to 2D.
            Shape (num_steps+1, 2) or list of such tensors.
        - `save_path` (Path, optional): Path to save the plot. Defaults to "denoising_traj_2d.gif".

    ### Returns
        - `None`

    ### Example
    ```python
    >>> noise_preds = torch.randn(50, 2)  # 50 timesteps in 2D
    >>> latents = torch.randn(51, 2)      # 51 states in 2D
    >>> denoising_traj_2d(noise_preds, latents, noise_scale=0.1, save_path="trajectory.png")
    # Creates trajectory.png showing:
    # - Blue dots and lines: Denoising trajectory through latent space
    # - Red arrows: Predicted noise direction at each step
    ```
    """
    if isinstance(noise_preds_proj_2d, Tensor):
        noise_preds_proj_2d = [noise_preds_proj_2d]
    if isinstance(latents_proj_2d, Tensor):
        latents_proj_2d = [latents_proj_2d]
    assert len(noise_preds_proj_2d) == len(latents_proj_2d)

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    for eps, x in zip(noise_preds_proj_2d, latents_proj_2d):
        assert len(eps) + 1 == len(x)
        eps = eps.flip(dims=[0])
        x = x.flip(dims=[0])
        ax.plot(
            x[:, 0],
            x[:, 1],
            c="black",
            linestyle="--",
            linewidth=0.5,
        )
        ax.scatter(
            x[:, 0],
            x[:, 1],
            s=10,
            c=plt.cm.viridis(np.linspace(0, 1, len(x))),
            marker="o",
        )
        ax.quiver(
            x[:-1, 0],  # Arrow start x
            x[:-1, 1],  # Arrow start y
            -eps[:, 0],  # Arrow dx
            -eps[:, 1],  # Arrow dy
            color="red",
            angles="xy",
            scale_units="xy",
            scale=1 / noise_scale,
            width=0.003,  # Reduce arrow thickness
        )
    ax.grid(alpha=0.25)
    ax.axis("equal")
    _path.mkdir(_path.dirname(save_path))
    fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0)
    plt.close("all")
