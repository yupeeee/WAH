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
    """
    Loads a text file and returns its contents as a list of strings.

    ### Parameters
    - `path` (Path): Path to the text file.
    - `dtype` (type, optional): Data type to map each line to. Defaults to `str`.

    ### Returns
    - `List[str]`: List of lines from the text file, optionally cast to the specified `dtype`.
    """
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
    """
    Saves a list to a text file.

    ### Parameters
    - `lst` (List[Any]): List to be saved to the file.
    - `save_name` (str): Name for the saved text file (without extension).
    - `save_dir` (Path, optional): Directory to save the text file. Defaults to the current directory.

    ### Notes
    - Each element of the list is saved as a new line in the text file.
    """
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
    """
    Sorts a list of strings in natural order (e.g., "item2" before "item10").

    ### Parameters
    - `str_list` (List[str]): List of strings to be sorted.
    - `return_indices` (bool, optional): If `True`, returns both sorted list and indices. Defaults to `False`.

    ### Returns
    - `List[str]`: Sorted list of strings if `return_indices` is `False`.
    - `Tuple[List[str], List[int]]`: Tuple of sorted list of strings and corresponding original indices if `return_indices` is `True`.
    """
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
