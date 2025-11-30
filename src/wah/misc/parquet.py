import os

import pandas as pd

__all__ = [
    "load",
]


def load(
    path: os.PathLike,
) -> pd.DataFrame:
    assert (
        os.path.splitext(path)[1] == ".parquet"
    ), f"File must be a .parquet file, got {path}"
    return pd.read_parquet(path)
