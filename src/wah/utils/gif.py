import os

from PIL import Image

from .. import path as _path
from ..typing import Optional, Path

__all__ = [
    "make_gif",
]


def make_gif(
    images_dir: Path,
    save_path: Path,
    fext: Optional[str] = None,
    fps: Optional[float] = 20,
) -> None:
    image_paths = _path.ls(images_dir, fext, sort=True, absolute=True)
    images = [Image.open(os.path.join(image_path)) for image_path in image_paths]

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=False,
        duration=1000 / fps,
        loop=0,
    )
