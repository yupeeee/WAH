import os

import pandas as pd

from ..typing import (
    Any,
    DataFrame,
    Dict,
    List,
    Path,
)

__all__ = [
    "load_csv",
    "to_df",
    "save_in_csv",
]


def load_csv(
    csv_path: Path,
    index_col: Any = 0,
) -> Dict[Any, List[Any]]:
    df = pd.read_csv(csv_path, index_col=index_col)

    return df.to_dict(orient="list")


def to_df(
    dictionary: Dict,
    index_col: Any = None,
) -> DataFrame:
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
