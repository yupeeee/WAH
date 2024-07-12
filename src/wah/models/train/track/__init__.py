from ....typing import (
    Config,
    List,
)
from . import (
    feature_rms,
    feature_sign,
    grad_l2,
    param_svdval_max,
    state_dict,
)


__all__ = [
    "Track",
    "load_track",
    "feature_rms",
    "feature_sign",
    "grad_l2",
    "param_svdval_max",
    "state_dict",
]


class Track:
    FEATURE_RMS = False
    FEATURE_SIGN = False
    GRAD_L2 = False
    PARAM_SVDVAL_MAX = False
    STATE_DICT = False


def load_track(
    config: Config,
) -> Track:
    track: Track = Track

    if (
        "track" not in config.keys()
        or config["track"] == "None"
        or config["track"][0] == "None"
    ):
        return track

    track_cfg: List[str] = config["track"]

    for t in track_cfg:
        try:
            setattr(track, t.upper(), True)
        except AttributeError:
            raise ValueError(f"Unsupported track: {t}")

    return track
