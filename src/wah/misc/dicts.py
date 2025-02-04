import pandas as pd
import torch
import yaml

from . import path as _path
from .typing import DataFrame, Dict, Optional, Path, Tensor, Union

__all__ = [
    "load",
    "round_nums",
    "save",
    "to_df",
    "to_tensor",
]


def load_csv(
    path: Path,
    **kwargs,
) -> DataFrame:
    """Load a CSV file and convert it to a dictionary.

    ### Args
        - `path` (Path): Path to the CSV file
        - `**kwargs`: Additional keyword arguments passed to pandas.read_csv()

    ### Returns
        - `Dict`: Dictionary containing the CSV data with column names as keys and values as lists

    ### Example
    ```python
    # data.csv contents:
    # id,col1,col2
    # 1,a,1.1
    # 2,b,2.2
    # 3,c,3.3
    >>> data = load_csv('data.csv')
    >>> print(data)
    {'id': [1, 2, 3], 'col1': ['a', 'b', 'c'], 'col2': [1.1, 2.2, 3.3]}

    # Using index_col to ignore the id column
    >>> data = load_csv('data.csv', index_col=0)
    >>> print(data)
    {'col1': ['a', 'b', 'c'], 'col2': [1.1, 2.2, 3.3]}
    >>> data = load_csv('data.csv', index_col='id')
    >>> print(data)
    {'col1': ['a', 'b', 'c'], 'col2': [1.1, 2.2, 3.3]}
    ```
    """
    df: DataFrame = pd.read_csv(
        path,
        **kwargs,
    )
    return df.to_dict(orient="list")


def load_yaml(
    path: Path,
    **kwargs,
) -> Dict:
    """Load a YAML file and convert it to a dictionary.

    ### Args
        - `path` (Path): Path to the YAML file
        - `**kwargs`: Additional keyword arguments passed to yaml.load()

    ### Returns
        - `Dict`: Dictionary containing the YAML data

    ### Example
    ```python
    # data.yaml contents:
    # key1: value1
    # key2:
    #     - item1
    #     - item2
    # key3:
    #     nested_key: nested_value
    >>> data = load_yaml('data.yaml')
    >>> print(data)
    {
        'key1': 'value1',
        'key2': ['item1', 'item2'],
        'key3': {'nested_key': 'nested_value'}
    }
    ```
    """
    with open(path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader, **kwargs)


def load(
    path: Path,
    **kwargs,
) -> Dict:
    """Load a file into a dictionary.
    Currently supports .csv and .yaml files.

    ### Args
        - `path` (Path): Path to the file
        - `**kwargs`: Additional keyword arguments passed to the respective loader function

    ### Returns
        - `Dict`: Dictionary containing the file data

    ### Raises
        - `ValueError`: If file extension is not supported

    ### Example (load csv)
    ```python
    # data.csv contents:
    # id,col1,col2
    # 1,a,1.1
    # 2,b,2.2
    # 3,c,3.3
    >>> data = load('data.csv')
    >>> print(data)
    {'id': [1, 2, 3], 'col1': ['a', 'b', 'c'], 'col2': [1.1, 2.2, 3.3]}

    # Using index_col to ignore the id column
    >>> data = load('data.csv', index_col=0)
    >>> print(data)
    {'col1': ['a', 'b', 'c'], 'col2': [1.1, 2.2, 3.3]}
    >>> data = load('data.csv', index_col='id')
    >>> print(data)
    {'col1': ['a', 'b', 'c'], 'col2': [1.1, 2.2, 3.3]}
    ```

    ### Example (load yaml)
    ```python
    # data.yaml contents:
    # key1: value1
    # key2:
    #     - item1
    #     - item2
    # key3:
    #     nested_key: nested_value
    >>> data = load('data.yaml')
    >>> print(data)
    {
        'key1': 'value1',
        'key2': ['item1', 'item2'],
        'key3': {'nested_key': 'nested_value'}
    }
    ```
    """
    # Get the file extension without the dot
    ext = _path.splitext(path).lower()

    if ext == ".csv":
        return load_csv(path, **kwargs)
    elif ext in (".yaml", ".yml"):
        return load_yaml(path, **kwargs)
    else:
        raise NotImplementedError(
            f"Could not load file with extension {ext}. Supported extensions are: .csv, .yaml, .yml"
        )


def round_nums(
    d: Dict,
    decimal: int = 2,
) -> Dict:
    """Round numeric values in a dictionary to a specified decimal place.

    ### Args
        - `d` (Dict): Input dictionary containing numeric values
        - `decimal` (int, optional): Number of decimal places to round to. Defaults to 2.

    ### Returns
        - `Dict`: Dictionary with rounded numeric values. Non-numeric values are left unchanged.

    ### Example
    ```python
    >>> data = {"val1": 1.2345, "val2": 2.7891, "text": "hello"}
    >>> rounded = round_nums(data, decimal=2)
    >>> print(rounded)
    {'val1': 1.23, 'val2': 2.79, 'text': 'hello'}
    ```
    """
    return {
        k: (
            v
            if isinstance(v, int)
            else round(float(v), decimal) if isinstance(v, float) else v
        )
        for k, v in d.items()
    }


def save_in_csv(
    d: Dict,
    path: Path,
    mode: str = "w",
    round: int = None,
    **kwargs,
) -> None:
    """Save a dictionary to a CSV file.

    ### Args
        - `d` (Dict): Dictionary to save
        - `path` (Path): Path to save the CSV file
        - `mode` (str, optional): File opening mode. Defaults to "w".
        - `round` (int, optional): Number of decimal places to round numeric values to. Defaults to None.
        - `**kwargs`: Additional keyword arguments passed to pandas.DataFrame.to_csv()

    ### Returns
        - `None`

    ### Example
    ```python
    >>> data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    >>> save_in_csv(data, "output.csv")
    # Creates output.csv with contents:
    # name,age
    # Alice,25
    # Bob,30

    # Round numeric values
    >>> data = {"val1": [1.2345, 2.3456], "val2": [3.4567, 4.5678]}
    >>> save_in_csv(data, "output.csv", round=2)
    # Creates output.csv with contents:
    # val1,val2
    # 1.23,3.46
    # 2.35,4.57
    ```
    """
    assert (
        _path.splitext(path).lower() == ".csv"
    ), f"File extension must be .csv, got {_path.splitext(path)}"
    for k, v in d.items():
        if not isinstance(v, list):
            d[k] = [v]
    df = to_df(d)
    if round is not None:
        df = round_nums(df, round)
    df.to_csv(path, mode=mode, index=False, **kwargs)


def save_in_yaml(
    d: Dict,
    path: Path,
    mode: str = "w",
    **kwargs,
) -> None:
    """Save a dictionary to a YAML file.

    ### Args
        - `d` (Dict): Dictionary to save
        - `path` (Path): Path to save the YAML file
        - `mode` (str, optional): File opening mode. Defaults to "w".
        - `**kwargs`: Additional keyword arguments passed to yaml.dump()

    ### Returns
        - `None`

    ### Example
    ```python
    >>> data = {"name": "Alice", "age": 25}
    >>> save_in_yaml(data, "output.yaml")
    # Creates output.yaml with contents:
    # name: Alice
    # age: 25

    >>> data = {"person": {"name": "Bob", "age": 30}}
    >>> save_in_yaml(data, "output.yaml")
    # Creates output.yaml with contents:
    # person:
    #   name: Bob
    #   age: 30
    ```
    """
    assert (
        _path.splitext(path).lower() == ".yaml"
    ), f"File extension must be .yaml, got {_path.splitext(path)}"
    with open(path, mode) as f:
        yaml.dump(d, f, **kwargs)


def save(
    d: Dict,
    path: Path,
    **kwargs,
) -> None:
    """Save a dictionary to a file. The file format is determined by the file extension.

    ### Args
        - `d` (Dict): Dictionary to save
        - `path` (Path): Path to save the file
        - `**kwargs`: Additional keyword arguments passed to the appropriate save function

    ### Returns
        - `None`

    ### Example
    ```python
    >>> data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    >>> save(data, "output.csv")
    # Creates output.csv with contents:
    # name,age
    # Alice,25
    # Bob,30

    >>> data = {"person": {"name": "Bob", "age": 30}}
    >>> save(data, "output.yaml")
    # Creates output.yaml with contents:
    # person:
    #   name: Bob
    #   age: 30
    ```
    """
    ext = _path.splitext(path).lower()
    assert ext in (
        ".csv",
        ".yaml",
        ".yml",
    ), f"File extension must be .csv, .yaml, or .yml, got {ext}"
    _path.mkdir(_path.dirname(path))
    if ext == ".csv":
        save_in_csv(d, path, **kwargs)
    elif ext == ".yaml" or ext == ".yml":
        save_in_yaml(d, path, **kwargs)
    else:
        raise NotImplementedError(
            f"Could not save file with extension {ext}. Supported extensions are: .csv, .yaml, .yml"
        )


def to_df(
    d: Dict,
    index: Optional[Union[str, pd.Index]] = None,
    **kwargs,
) -> DataFrame:
    """Convert dictionary to a pandas DataFrame.

    ### Args
        - `d` (Dict): Input dictionary
        - `index` (Optional[Union[str, pd.Index]], optional): Index for the DataFrame.
            If a string is provided and matches a column name, that column will be used as the index.
            If a pd.Index object is provided (e.g., a list like [1, 2, 3] or ['a', 'b', 'c']), it will be used directly as the index of the DataFrame.
            Defaults to None.
        - `**kwargs`: Additional arguments passed to pd.DataFrame()

    ### Returns
        - `DataFrame`: A pandas DataFrame created from the dictionary

    ### Example
    ```python
    >>> data = {"name": ["Alice", "Bob"], "age": [25, 30]}
    >>> df = to_df(data)
    >>> print(df)
        name  age
    0  Alice   25
    1    Bob   30
    # Use a column as index
    >>> df = to_df(data, index="name")
    >>> print(df)
            age
    Alice   25
    Bob     30
    # Use a custom list as index
    >>> df = to_df(data, index=['person1', 'person2'])
    >>> print(df)
                name  age
    person1  Alice   25
    person2    Bob   30
    ```
    """
    df = pd.DataFrame(d, **kwargs)
    if index is not None:
        if isinstance(index, str) and index in df.columns:
            df = df.set_index(index)
            setattr(
                df.index, index, None
            )  # Remove the name attribute from the index to avoid redundancy
        else:
            df.index = index
    return df


def to_tensor(
    d: Dict,
    **kwargs,
) -> Tensor:
    """Convert dictionary values to a tensor.

    ### Args
        - `d` (Dict): Input dictionary with numeric values
        - `**kwargs`: Additional arguments passed to torch.tensor()

    ### Returns
        - `Tensor`: A tensor containing the dictionary values

    ### Example
    ```python
    >>> data = {"val1": [1.0, 1.2], "val2": [2.5, 2.7]}
    >>> tensor = to_tensor(data)
    >>> print(tensor)
    tensor([[1.0, 1.2],
            [2.5, 2.7]])
    # Convert with specific dtype
    >>> tensor_int = to_tensor(data, dtype=torch.int64)
    >>> print(tensor_int)
    tensor([[1, 1],
            [2, 2]], dtype=torch.int64)
    ```
    """
    return torch.tensor([v for v in d.values()], **kwargs)
