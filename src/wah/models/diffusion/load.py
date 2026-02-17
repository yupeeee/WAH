from typing import Dict

import diffusers

__all__ = [
    "_load_scheduler",
]


def _load_scheduler(
    version: str,
    strategy: str,
    model_path_dict: Dict[str, str],
    **kwargs,
) -> diffusers.SchedulerMixin:
    """
    Load a scheduler from the diffusers library.

    ### Args
        - `version` (str): Version of the model to load the scheduler for.
        - `strategy` (str): Name of the scheduler strategy (e.g. "DDPM", "DDIM").
        - `model_path_dict` (Dict[str, str]): Dictionary of model paths for each version.
        - `**kwargs`: Additional keyword arguments to the scheduler's `from_pretrained` method.

    ### Returns
        - `diffusers.SchedulerMixin`: Instantiated scheduler.

    ### Raises
        - `ValueError`: If the requested scheduler does not exist in the diffusers library.

    ### Example
    ```python
    >>> scheduler = _load_scheduler(
    ...     version="1.4",
    ...     strategy="DDIM",
    ...     model_path_dict={
    ...         ...,
    ...         "1.4": "CompVis/stable-diffusion-v1-4",
    ...         ...,
    ...     },
    ... )
    ```
    """
    try:
        scheduler = getattr(diffusers, f"{strategy}Scheduler").from_pretrained(
            pretrained_model_name_or_path=model_path_dict[version],
            subfolder="scheduler",
            **kwargs,
        )
        return scheduler
    except AttributeError as e:
        raise ValueError(f"Unknown scheduler: {strategy}Scheduler ({e})")
