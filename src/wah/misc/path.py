import os
import shutil

from . import lists
from .typing import List, Optional, Path, Tuple

__all__ = [
    "basename",
    "clean",
    "dirname",
    "exists",
    "isdir",
    "join",
    "ls",
    "mkdir",
    "rmdir",
    "rmfile",
    "split",
    "splitext",
    "walk",
]


def basename(
    path: Path,
) -> str:
    """Get the base name of a path.

    ### Args
        - `path` (Path): Path to get base name from

    ### Returns
        - `str`: Base name of the path

    ### Example
    ```python
    >>> basename('/home/user/file.txt')
    'file.txt'

    >>> basename('path/to/directory/')
    'directory'
    ```
    """
    return os.path.basename(clean(path))


def clean(
    path: Path,
) -> Path:
    """Clean a path by expanding user directory and normalizing path separators.

    ### Args
        - `path` (Path): Path to clean

    ### Returns
        - `Path`: Cleaned path with expanded user directory and normalized separators

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


def dirname(
    path: Path,
) -> Path:
    """Get the directory name of a path.

    ### Args
        - `path` (Path): Path to get directory name from

    ### Returns
        - `Path`: Directory name of the path

    ### Example
    ```python
    >>> dirname('/home/user/file.txt')
    '/home/user'

    >>> dirname('path/to/directory/')
    'path/to'
    ```
    """
    return os.path.dirname(clean(path))


def exists(
    path: Path,
) -> bool:
    """Check if a path exists.

    ### Args
        - `path` (Path): Path to check

    ### Returns
        - `bool`: True if the path exists, False otherwise

    ### Example
    ```python
    >>> exists('/path/to/existing/file.txt')
    True

    >>> exists('/path/to/nonexistent/file.txt')
    False
    ```
    """
    return os.path.exists(clean(path))


def isdir(
    path: Path,
) -> bool:
    """Check if a path is a directory.

    ### Args
        - `path` (Path): Path to check

    ### Returns
        - `bool`: True if the path is a directory, False otherwise

    ### Example
    ```python
    >>> isdir('/path/to/directory')
    True

    >>> isdir('/path/to/file.txt')
    False
    ```
    """
    return os.path.isdir(clean(path))


def join(
    *paths,
) -> Path:
    """Join multiple path components.

    ### Args
        - `*paths`: Variable number of path components to join

    ### Returns
        - `Path`: Joined path

    ### Example
    ```python
    >>> join('path', 'to', 'file.txt')
    'path/to/file.txt'

    >>> join('/absolute/path', 'file.txt')
    '/absolute/path/file.txt'
    ```
    """
    return clean(os.path.join(*paths))


def ls(
    path: Path,
    fext: Optional[str] = None,
    sort: Optional[bool] = True,
    absolute: Optional[bool] = False,
) -> List[str]:
    """List files in a directory.

    ### Args
        - `path` (Path): Path to the directory
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
    file_list = os.listdir(clean(path))
    if sort:
        file_list = lists.sort(file_list)

    # Directories only
    if fext == "dir":
        file_list = [f for f in file_list if isdir(join(path, f))]
    # Files with extension
    elif fext not in [None, ""]:
        # Add dot prefix if missing
        if not fext.startswith("."):
            fext = "." + fext
        # Return files with matching extension
        file_list = [f for f in file_list if splitext(f) == fext]
    # All files
    else:
        file_list = file_list

    # Absolute paths
    if absolute:
        file_list = [join(path, f) for f in file_list]

    return file_list


def mkdir(
    path: Path,
) -> None:
    """Create a directory and any necessary parent directories.

    ### Args
        - `path` (Path): Path to create directory at

    ### Returns
        - `None`

    ### Example
    ```python
    >>> mkdir('path/to/new/directory')
    # Creates all directories in path if they don't exist
    ```
    """
    os.makedirs(
        name=clean(path),
        exist_ok=True,
    )


def rmdir(
    path: Path,
) -> None:
    """Remove a directory and all its contents.

    ### Args
        - `path` (Path): Path to directory to remove

    ### Returns
        - `None`

    ### Example
    ```python
    >>> rmdir('path/to/directory')
    # Removes directory and all contents
    ```
    """
    shutil.rmtree(clean(path))


def rmfile(
    fpath: Path,
) -> None:
    """Remove a file.

    ### Args
        - `fpath` (Path): Path to file to remove

    ### Returns
        - `None`

    ### Example
    ```python
    >>> rmfile('path/to/file.txt')
    # Removes file
    ```
    """
    os.remove(clean(fpath))


def split(
    path: Path,
) -> Tuple[Path, str]:
    """Split a path into directory and file components.

    ### Args
        - `path` (Path): Path to split

    ### Returns
        - `Tuple[Path, str]`: Tuple containing (directory path, file name)

    ### Example
    ```python
    >>> split('/home/user/file.txt')
    ('/home/user', 'file.txt')

    >>> split('path/to/directory/')
    ('path/to', 'directory')
    ```
    """
    return os.path.split(clean(path))


def splitext(
    path: Path,
) -> str:
    """Split the extension from a path.

    ### Args
        - `path` (Path): Path to split extension from

    ### Returns
        - `str`: File extension (including the dot)

    ### Example
    ```python
    >>> splitext('/home/user/file.txt')
    '.txt'

    >>> splitext('path/to/file')
    ''
    ```
    """
    return os.path.splitext(clean(path))[1]


def walk(
    root: Path,
    absolute: Optional[bool] = False,
) -> List[Path]:
    """Walk a directory recursively and return all file paths.

    This function traverses a directory tree starting from the given root path,
    and returns a list of all file paths found. It follows symbolic links and
    includes files in all subdirectories.

    ### Args
        - `root` (Path): Root directory path to start walking from
        - `absolute` (Optional[bool]): If True, return absolute paths. If False, return paths relative to root.
                                     Defaults to False.

    ### Returns
        - `List[Path]`: List of paths to all files found under the root directory.
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
    for dirpath, dirnames, filenames in os.walk(clean(root)):
        for dirname in dirnames:
            subdir = join(dirpath, dirname)
            if isdir(subdir):
                paths.extend(walk(subdir))
        for filename in filenames:
            paths.append(join(dirpath, filename))
    if absolute:
        paths = [join(root, path) for path in paths]
    return paths
