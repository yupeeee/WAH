from ...misc.typing import Union
from .sd1_ import SDv1
from .sd2_ import SDv2

__all__ = [
    "StableDiffusion",
]

supported_versions = [
    version
    for version in list(sd1_.model_ids.keys()) + list(sd2_.model_ids.keys())
    # + list(sd3_5.model_ids.keys())
]


def StableDiffusion(
    version: str,
    scheduler: str,
    blur_nsfw: bool = True,
    verbose: bool = True,
    **kwargs,
) -> Union[SDv1, SDv2]:
    if version not in supported_versions:
        raise ValueError(
            f"Version {version} is not supported for Stable Diffusion.\n"
            f"Supported versions: {supported_versions}"
        )
    if version.startswith("1."):
        return SDv1(version, scheduler, blur_nsfw, verbose, **kwargs)
    elif version.startswith("2"):
        return SDv2(version, scheduler, blur_nsfw, verbose, **kwargs)
    # elif version.startswith("3.5-"):
    #     return sd3_5.load_pipeline(version, scheduler, **kwargs)
    else:
        raise
