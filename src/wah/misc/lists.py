import re

from . import path as _path
from .typing import Any, List, Path, Tuple, Union

__all__ = [
    "load",
    "save",
    "sort",
]


def load(
    path: Path,
    dtype: type = str,
) -> List[str]:
    """Load a text file and convert it to a list.

    ### Args
        - `path` (Path): Path to the text file
        - `dtype` (type): Type to convert each line to. Defaults to `str`.

    ### Returns
        - `List[str]`: List containing the text file data with each line as an element

    ### Example
    ```python
    # data.txt contents:
    # 1
    # 2
    # 3
    >>> data = load('data.txt')
    >>> print(data)
    ['1', '2', '3']

    # Using dtype=int to convert lines to integers
    >>> data = load('data.txt', dtype=int)
    >>> print(data)
    [1, 2, 3]
    ```
    """
    assert _path.splitext(path) == ".txt", f"File must be a .txt file, got {path}"
    with open(path, "r") as f:
        return [dtype(line.rstrip()) for line in f]


def save(
    l: List[Any],
    path: Path,
) -> None:
    """Save a list to a text file.

    ### Args
        - `l` (List[Any]): List to save
        - `path` (Path): Path to save the text file

    ### Returns
        - `None`

    ### Example
    ```python
    >>> data = [1, 2, 3]
    >>> save(data, "output.txt")
    # Creates output.txt with contents:
    # 1
    # 2
    # 3

    >>> data = ["hello", "world"]
    >>> save(data, "output.txt")
    # Creates output.txt with contents:
    # hello
    # world
    ```
    """
    assert _path.splitext(path) == ".txt", f"File must be a .txt file, got {path}"
    _path.mkdir(_path.dirname(path))
    with open(path, "w") as f:
        f.write("\n".join(str(item) for item in l))


def sort(
    l: List[str],
    return_indices: bool = False,
) -> Union[
    List[str],  # return_indices = False
    Tuple[List[str], List[int]],  # return_indices = True
]:
    """Sort a list using natural sorting.

    Natural sorting ensures that strings containing numbers are sorted in a way that matches
    human intuition. For example, "10" comes before "2" in standard string sorting, but
    with natural sorting "10" comes after "2" as expected.

    ### Args
        - `l` (List[str]): List to sort
        - `return_indices` (bool, optional): Whether to return the indices of the sorted elements.
            Defaults to False.

    ### Returns
        - `Union[List[str], Tuple[List[str], List[int]]]`: If return_indices is False, returns
            the sorted list. If return_indices is True, returns a tuple of (sorted list, indices).

    ### Example
    ```python
    >>> l = ["file2.txt", "file10.txt", "file1.txt"]
    >>> sort(l)
    ['file1.txt', 'file2.txt', 'file10.txt']

    >>> l = ["file2.txt", "file10.txt", "file1.txt"]
    >>> sort(l, return_indices=True)
    (['file1.txt', 'file2.txt', 'file10.txt'], [2, 0, 1])
    ```
    """

    def natural_key(text):
        # Convert number strings to integers for proper numerical sorting
        return [
            int(c) if c.isdigit() else c.lower() for c in re.split("([0-9]+)", text)
        ]

    # Store original types and convert all items to strings
    types = [type(x) for x in l]
    str_l = [str(x) for x in l]

    # Sort using natural sort key on string versions
    indices = sorted(range(len(str_l)), key=lambda i: natural_key(str_l[i]))

    # Create sorted list preserving original types
    l_sorted = [types[i](str_l[i]) for i in indices]

    return (l_sorted, indices) if return_indices else l_sorted
