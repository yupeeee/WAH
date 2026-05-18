import os
from typing import Union

from huggingface_hub import login as _login

from . import path as _path

__all__ = [
    "login",
]


def login(
    hf_token: str,
    hf_home: Union[str, os.PathLike],
    hf_hub_cache: Union[str, os.PathLike],
) -> None:
    if not hf_token or not hf_token.strip():
        raise ValueError("Hugging Face login failed: token cannot be empty.")

    hf_home = _path.clean(hf_home)
    hf_hub_cache = _path.clean(hf_hub_cache)
    os.makedirs(hf_home, exist_ok=True)
    os.makedirs(hf_hub_cache, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = hf_hub_cache

    _login(
        token=hf_token,
        add_to_git_credential=False,
        skip_if_logged_in=True,
    )
    os.environ["HF_TOKEN"] = hf_token
