import time

__all__ = [
    "current_time",
]


def current_time() -> str:
    """
    Returns the current time in the format `YYYYMMDDHHMMSSNNNNNNNNN`.

    ### Parameters
    - `None`

    ### Returns
    - `str`: The current time formatted as `YYYYMMDDHHMMSSNNNNNNNNN`.

    ### Notes
    - The function uses `time.time_ns()` to get the current time in nanoseconds.
    - The time is then split into seconds and nanoseconds.
    - The seconds are formatted into `YYYYMMDDHHMMSS` using `time.strftime`.
    - The nanoseconds are appended to the formatted string to achieve nanosecond precision.
    """
    now_ns = time.time_ns()

    seconds = now_ns // 1_000_000_000
    nanoseconds = now_ns % 1_000_000_000

    current_time = time.strftime("%Y%m%d%H%M%S", time.localtime(seconds))
    current_time = f"{current_time}{nanoseconds:09d}"

    return current_time
