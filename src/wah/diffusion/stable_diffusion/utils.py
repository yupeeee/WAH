import subprocess

import diffusers
import torch
from diffusers import SchedulerMixin

from ...misc.typing import Device, Dict, Literal

__all__ = [
    "login",
    "is_valid_version",
    "load_generator",
    "load_scheduler",
]


def login() -> None:
    """Login to Hugging Face using access token via CLI.

    ### Example
    ```python
    >>> login()
    ```
    """
    # Check if already logged in by attempting to run whoami
    result = subprocess.run(
        ["huggingface-cli", "whoami"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0 and result.stdout == "Not logged in\n":
        # If not logged in, login
        process = subprocess.Popen(
            ["huggingface-cli", "login"],
            stdin=subprocess.PIPE,
            text=True,
        )
        process.communicate(input="n\n")
        process.wait()
    else:
        print("Logged in as", result.stdout.strip())


def is_valid_version(version: str, ids: Dict[str, str]) -> bool:
    """Check if a version is valid for a given model ID."""
    if version in ids.keys():
        return True
    else:
        raise ValueError(
            f"Invalid version ({version})\n"
            f"Valid versions and their model IDs are:\n"
            + "\n".join(f"{k} ({v})" for k, v in ids.items())
        )


def load_generator(seed: int = None, device: Device = "cpu") -> torch.Generator:
    """Load a generator with a given seed and device."""
    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    return generator


def load_scheduler(
    version: str,
    model_ids: Dict[str, str],
    strategy: Literal[
        "DDIM",
        "LMSDiscrete",
        "EulerDiscrete",
        "EulerAncestralDiscrete",
        "DPMSolverMultistep",
    ] = "DDIM",
    # **kwargs,
) -> SchedulerMixin:
    scheduler = getattr(diffusers, f"{strategy}Scheduler").from_pretrained(
        model_ids[version],
        subfolder="scheduler",
        # **kwargs,
    )
    return scheduler
