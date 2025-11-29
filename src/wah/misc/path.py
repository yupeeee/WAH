import os
from typing import List, Optional

from . import lists

__all__ = [
    "clean",
    "ls",
    "walk",
]


def clean(
    path: os.PathLike,
) -> os.PathLike:
    """Clean a path by expanding user directory and normalizing path separators.

    ### Args
        - `path` (os.PathLike): Path to clean

    ### Returns
        - `os.PathLike`: Cleaned path with expanded user directory and normalized separators

    ### Example
    ```python
    >>> clean('~/Documents')
    '/home/user/Documents'

    >>> clean('path/to/../folder/')
    'path/folder'
    ```
    """
    path = os.path.expanduser(path)
    path = os.path.normpath(path)
    return path


def ls(
    path: os.PathLike,
    fext: Optional[str] = None,
    sort: Optional[bool] = True,
    absolute: Optional[bool] = False,
) -> List[str]:
    """List files in a directory.

    ### Args
        - `path` (os.PathLike): Path to the directory
        - `fext` (Optional[str]): File extension filter. If "dir", only list directories.
            If None or empty string, list all files. Otherwise, list files with matching extension.
            Defaults to None.
        - `sort` (Optional[bool]): Whether to sort the file list. Defaults to True.
        - `absolute` (Optional[bool]): Whether to return absolute paths. Defaults to False.

    ### Returns
        - `List[str]`: List of files in the directory

    ### Example
    ```python
    >>> ls('path/to/directory')
    ['file1.txt', 'file2.txt', 'subdirectory']

    >>> ls('path/to/directory', fext='txt')
    ['file1.txt', 'file2.txt']

    >>> ls('path/to/directory', fext='dir')
    ['subdirectory']

    >>> ls('path/to/directory', fext='txt', absolute=True)
    ['/absolute/path/to/directory/file1.txt', '/absolute/path/to/directory/file2.txt']
    ```
    """
    file_list = os.listdir(path)
    if sort:
        file_list = lists.sort(file_list)

    # Directories only
    if fext == "dir":
        file_list = [f for f in file_list if os.path.isdir(os.path.join(path, f))]
    # Files with extension
    elif fext not in [None, ""]:
        # Add dot prefix if missing
        if not fext.startswith("."):
            fext = "." + fext
        # Return files with matching extension
        file_list = [f for f in file_list if os.path.splitext(f)[1] == fext]
    # All files
    else:
        file_list = file_list

    # Absolute paths
    if absolute:
        file_list = [os.path.join(path, f) for f in file_list]

    return file_list


def walk(
    root: os.PathLike,
    absolute: Optional[bool] = False,
) -> List[os.PathLike]:
    """Walk a directory recursively and return all file paths.

    This function traverses a directory tree starting from the given root path,
    and returns a list of all file paths found. It follows symbolic links and
    includes files in all subdirectories.

    ### Args
        - `root` (os.PathLike): Root directory path to start walking from
        - `absolute` (Optional[bool]): If True, return absolute paths. If False, return paths relative to root.
                                     Defaults to False.

    ### Returns
        - `List[os.PathLike]`: List of paths to all files found under the root directory.
                       If absolute=False (default), paths are relative to the root directory.
                       If absolute=True, paths are absolute.

    ### Example
    ```python
    >>> walk('data')
    ['train.txt', 'test.txt', 'models/resnet18.pth']

    >>> walk('data', absolute=True)
    ['/home/user/data/train.txt', '/home/user/data/test.txt', '/home/user/data/models/resnet18.pth']
    ```

    ### Notes
        - Hidden files and directories (starting with '.') are included
        - Directory paths themselves are not included in the output list
        - All paths use forward slashes ('/'), even on Windows systems
    """
    paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        for dirname in dirnames:
            subdir = os.path.join(dirpath, dirname)
            if os.path.isdir(subdir):
                paths.extend(walk(subdir))
        for filename in filenames:
            paths.append(os.path.join(dirpath, filename))
    if absolute:
        paths = [os.path.join(root, path) for path in paths]
    return paths
