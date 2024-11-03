import pandas as pd
import torch
import yaml

from .. import path as _path
from ..typing import Any, DataFrame, Dict, List, Path, Tensor

__all__ = [
    "dict_to_df",
    "dict_to_tensor",
    "load_csv_to_dict",
    "load_yaml_to_dict",
    "save_dict_to_csv",
]


def dict_to_df(
    dictionary: Dict,
    index_col: Any = None,
) -> DataFrame:
    """
    Converts a dictionary to a Pandas DataFrame.

    ### Parameters
    - `dictionary` (Dict): Dictionary to convert to DataFrame.
    - `index_col` (Any, optional): Column to set as the index. Defaults to `None`.

    ### Returns
    - `DataFrame`: Pandas DataFrame created from the input dictionary.
    """
    df = pd.DataFrame(dictionary)

    if index_col is not None:
        df.set_index(index_col)

    return df


def dict_to_tensor(
    dictionary: Dict,
) -> Tensor:
    """
    Converts a dictionary of values into a PyTorch Tensor.

    ### Parameters
    - `dictionary` (Dict): Dictionary whose values will be converted into a Tensor.

    ### Returns
    - `Tensor`: A PyTorch Tensor created from the values of the input dictionary.
    """
    values = []

    for v in dictionary.values():
        values.append(v)

    return torch.Tensor(values)


def load_csv_to_dict(
    csv_path: Path,
    index_col: Any = 0,
) -> Dict[Any, List[Any]]:
    """
    Loads a CSV file and converts it into a dictionary.

    ### Parameters
    - `csv_path` (Path): Path to the CSV file to load.
    - `index_col` (Any, optional): Column to set as the index. Defaults to `0`.

    ### Returns
    - `Dict[Any, List[Any]]`: Dictionary created from the CSV file where keys are column headers and values are lists of column data.
    """
    df = pd.read_csv(csv_path, index_col=index_col)

    return df.to_dict(orient="list")


def load_yaml_to_dict(
    path: Path,
) -> Dict[Any, Any]:
    """
    Loads a YAML file and converts it into a dictionary.

    ### Parameters
    - `path` (Path): Path to the YAML file to load.

    ### Returns
    - `Dict[Any, Any]`: Dictionary created from the YAML file.
    """
    path = _path.clean(path)

    with open(path, "r") as f:
        yaml_dict = yaml.load(f, Loader=yaml.FullLoader)

    return yaml_dict


def save_dict_to_csv(
    dictionary: Dict,
    save_dir: Path,
    save_name: str,
    index_col: Any = None,
) -> None:
    """
    Saves a dictionary as a CSV file.

    ### Parameters
    - `dictionary` (Dict): Dictionary to save as a CSV file.
    - `save_dir` (Path): Directory to save the CSV file in.
    - `save_name` (str): Name to give to the saved CSV file (without extension).
    - `index_col` (Any, optional): Column to set as the index. Defaults to `None`.
    """
    for k, v in dictionary.items():
        if not isinstance(v, list):
            dictionary[k] = [v]

    df = dict_to_df(dictionary, index_col)

    _path.mkdir(save_dir)
    save_path = _path.join(save_dir, f"{save_name}.csv")

    if index_col is not None:
        df.to_csv(save_path, mode="w", index=False)

    else:
        df.to_csv(save_path, mode="w")
