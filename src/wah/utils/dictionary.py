import os

import pandas as pd
import torch

from ..typing import (
    Any,
    DataFrame,
    Dict,
    List,
    Path,
    Tensor,
)

__all__ = [
    "load_csv",
    "to_df",
    "save_in_csv",
    "to_tensor",
]


def load_csv(
    csv_path: Path,
    index_col: Any = 0,
) -> Dict[Any, List[Any]]:
    """
    Loads a CSV file into a dictionary.

    ### Parameters
    - `csv_path (Path)`: The path to the CSV file.
    - `index_col (Any)`: The column to set as index. Defaults to 0.

    ### Returns
    - `Dict[Any, List[Any]]`: A dictionary where keys are column names and values are lists of column data.

    ### Notes
    - This function reads the CSV file from the given path and converts it into a dictionary with columns as lists.
    """
    df = pd.read_csv(csv_path, index_col=index_col)

    return df.to_dict(orient="list")


def to_df(
    dictionary: Dict,
    index_col: Any = None,
) -> DataFrame:
    """
    Converts a dictionary to a DataFrame.

    ### Parameters
    - `dictionary (Dict)`: The dictionary to convert.
    - `index_col (Any)`: The column to set as index. Defaults to None.

    ### Returns
    - `DataFrame`: A DataFrame created from the dictionary.

    ### Notes
    - This function creates a DataFrame from the given dictionary.
    - If `index_col` is specified, it sets the index of the DataFrame to that column.
    """
    df = pd.DataFrame(dictionary)

    if index_col is not None:
        df.set_index(index_col)

    return df


def save_in_csv(
    dictionary: Dict,
    save_dir: Path,
    save_name: str,
    index_col: Any = None,
) -> None:
    """
    Saves a dictionary to a CSV file.

    ### Parameters
    - `dictionary (Dict)`: The dictionary to save.
    - `save_dir (Path)`: The directory to save the CSV file in.
    - `save_name (str)`: The name of the CSV file.
    - `index_col (Any)`: The column to set as index. Defaults to None.

    ### Returns
    - `None`

    ### Notes
    - This function converts the given dictionary to a DataFrame and saves it as a CSV file.
    - The directory is created if it does not exist.
    - If `index_col` is specified, it sets the index of the DataFrame to that column before saving.
    """
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(
        save_dir,
        f"{save_name}.csv",
    )

    df = to_df(dictionary, index_col)

    if index_col is not None:
        df.to_csv(save_path, mode="w", index=False)

    else:
        df.to_csv(save_path, mode="w")


def to_tensor(
    dictionary: Dict,
) -> Tensor:
    """
    Converts a dictionary of lists into a tensor.

    ### Parameters
    - `dictionary (Dict)`:
      The dictionary to convert, where each key maps to a list of numerical values.

    ### Returns
    - `Tensor`:
      A tensor containing the values from the dictionary.

    ### Notes
    - This function extracts the values from each key in the dictionary, assuming they are lists of numerical values, and converts them into a tensor.
    - The resulting tensor will have a shape determined by the lengths of the lists in the dictionary values.
    """
    values = []

    for v in dictionary.values():
        values.append(v)

    return torch.Tensor(values)
