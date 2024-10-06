import time as _time

__all__ = [
    "time",
]


def time() -> str:
    """
    Returns the current time as a formatted string including both seconds and nanoseconds.

    ### Returns
    - `str`: The current time in the format `YYYYMMDDHHMMSSNNNNNNNNN`,
    where the first part represents the date and time in years, months, days, hours, minutes, and seconds, followed by nanoseconds.
    """
    current_time_ns = _time.time_ns()

    seconds = current_time_ns // 1_000_000_000
    nanoseconds = current_time_ns % 1_000_000_000

    current_time = _time.strftime("%Y%m%d%H%M%S", _time.localtime(seconds))
    current_time = f"{current_time}{nanoseconds:09d}"

    return current_time
