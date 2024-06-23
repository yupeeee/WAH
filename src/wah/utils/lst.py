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
    """
    Loads a text file and returns its contents as a list of strings or specified type.

    ### Parameters
    - `path` (Path):
      The path to the text file.
    - `dtype` (Type):
      The type to which each line should be converted.
      Defaults to `str`.

    ### Returns
    - `List[str]`:
      A list of strings or specified type representing the lines in the text file.

    ### Notes
    - This function reads the text file line by line, strips the newline characters, and converts each line to the specified type.
    """
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
    """
    Saves a list of strings to a text file.

    ### Parameters
    - `lst` (List[str]):
      The list of strings to save.
    - `save_name` (str):
      The name of the text file to save.
    - `save_dir` (Path):
      The directory to save the text file in.
      Defaults to the current directory.

    ### Returns
    - `None`

    ### Notes
    - This function creates the specified directory if it does not exist.
    - The list of strings is written to a text file, with each element on a new line.
    """
    save_dir = os.path.normpath(save_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{save_name}.txt")

    with open(save_path, "w+") as f:
        f.write("\n".join([str(v) for v in lst]))
