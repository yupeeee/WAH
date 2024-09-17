import re

from .. import path as _path
from ..typing import Any, List, Path, Tuple, Union

__all__ = [
    "load_txt_to_list",
    "save_list_to_txt",
    "sort_str_list",
]


def load_txt_to_list(
    path: Path,
    dtype: type = str,
) -> List[str]:
    path = _path.clean(path)

    txt_in_str_list = [line.rstrip("\n") for line in open(path, "r")]
    lst_mapped = map(dtype, txt_in_str_list)
    lst = list(lst_mapped)

    return lst


def save_list_to_txt(
    lst: List[Any],
    save_name: str,
    save_dir: Path = ".",
) -> None:
    _path.mkdir(save_dir)
    save_path = _path.join(save_dir, f"{save_name}.txt")

    with open(save_path, "w+") as f:
        f.write("\n".join([str(v) for v in lst]))


def sort_str_list(
    str_list: List[str],
    return_indices: bool = False,
) -> Union[
    List[str],  # return_indices = False
    Tuple[List[str], List[int]],  # return_indices = True
]:
    convert = lambda text: int(text) if text.isdigit() else text

    sorted_str_list = sorted(
        str_list,
        key=lambda key: [convert(c) for c in re.split("([0-9]+)", key)],
    )

    if not return_indices:
        return sorted_str_list

    else:
        indices = sorted(
            range(len(str_list)),
            key=lambda key: [convert(c) for c in re.split("([0-9]+)", str_list[key])],
        )

        return sorted_str_list, indices
