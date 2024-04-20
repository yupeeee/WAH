import re

from ..typing import (
    List,
    Tuple,
    Union,
)

__all__ = [
    "sort_str_list",
]


def sort_str_list(
    str_list: List[str],
    return_indices: bool = False,
) -> Union[
    List[str],                      # return_indices = False
    Tuple[List[str], List[int]],    # return_indices = True
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
            key=lambda key: [convert(c) for c in re.split('([0-9]+)', str_list[key])],
        )

        return sorted_str_list, indices
