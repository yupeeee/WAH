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
    df = pd.DataFrame(dictionary)

    if index_col is not None:
        df.set_index(index_col)

    return df


def dict_to_tensor(
    dictionary: Dict,
) -> Tensor:
    values = []

    for v in dictionary.values():
        values.append(v)

    return torch.Tensor(values)


def load_csv_to_dict(
    csv_path: Path,
    index_col: Any = 0,
) -> Dict[Any, List[Any]]:
    df = pd.read_csv(csv_path, index_col=index_col)

    return df.to_dict(orient="list")


def load_yaml_to_dict(
    path: Path,
) -> Dict[Any, Any]:
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
    df = dict_to_df(dictionary, index_col)

    _path.mkdir(save_dir)
    save_path = _path.join(save_dir, f"{save_name}.csv")

    if index_col is not None:
        df.to_csv(save_path, mode="w", index=False)

    else:
        df.to_csv(save_path, mode="w")
