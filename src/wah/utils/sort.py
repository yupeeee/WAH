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
    List[str],  # return_indices = False
    Tuple[List[str], List[int]],  # return_indices = True
]:
    """
    Sorts a list of strings in natural order, optionally returning the sorted indices.

    ### Parameters
    - `str_list (List[str])`: The list of strings to sort.
    - `return_indices (bool)`: Whether to return the sorted indices along with the sorted list. Defaults to `False`.

    ### Returns
    - `Union[List[str], Tuple[List[str], List[int]]]`:
      - A sorted list of strings if `return_indices` is `False`.
      - A tuple containing the sorted list of strings and the sorted indices if `return_indices` is `True`.

    ### Notes
    - This function uses natural sorting (e.g., "item2" comes before "item10").
    - If `return_indices` is `True`, the function also returns the indices that would sort the original list.
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
