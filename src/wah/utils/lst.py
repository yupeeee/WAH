import os

from ..typing import (
    List,
    Path,
)

__all__ = [
    "load_txt",
    "save_in_txt",
]


def load_txt(
    path: Path,
    dtype: type = str,
) -> List[str]:
    path = os.path.normpath(path)

    txt_in_str_list = [line.rstrip("\n") for line in open(path, "r")]
    lst_mapped = map(dtype, txt_in_str_list)
    lst = list(lst_mapped)

    return lst


def save_in_txt(
    lst: List[str],
    save_name: str,
    save_dir: Path = ".",
) -> None:
    save_dir = os.path.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{save_name}.txt")

    with open(save_path, "w+") as f:
        f.write("\n".join([str(v) for v in lst]))
