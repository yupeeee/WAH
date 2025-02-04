import time as _time

__all__ = [
    "time",
]


def time() -> str:
    """Get current time as a string.

    ### Returns
        - `str`: Current time in format YYYYMMDDHHMMSSXXXXXXXXX where X represents nanoseconds

    ### Example
    ```python
    >>> time()
    '20250123123456123456789'
    ```
    """
    return f"{_time.strftime('%Y%m%d%H%M%S', _time.localtime())}{_time.time_ns() % 1_000_000_000:09d}"
