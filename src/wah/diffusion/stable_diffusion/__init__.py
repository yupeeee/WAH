from ...misc.typing import Union
from . import v1, v2
from .log import Logger

__all__ = [
    "StableDiffusion",
    "Logger",
]

supported_versions = [
    version
    for version in list(v1.model_ids.keys()) + list(v2.model_ids.keys())
    # + list(v3_5.model_ids.keys())
]


def StableDiffusion(
    version: str,
    scheduler: str,
    verbose: bool = True,
    **kwargs,
) -> Union[v1.SDv1, v2.SDv2]:
    if version not in supported_versions:
        raise ValueError(
            f"Version {version} is not supported for Stable Diffusion.\n"
            f"Supported versions: {supported_versions}"
        )
    if version.startswith("1."):
        return v1.SDv1(version, scheduler, verbose, **kwargs)
    elif version.startswith("2"):
        return v2.SDv2(version, scheduler, verbose, **kwargs)
    # elif version.startswith("3.5-"):
    #     return sd3_5.load_pipeline(version, scheduler, verbose, **kwargs)
    else:
        raise
