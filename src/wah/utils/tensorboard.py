import os

import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from .. import path as _path
from ..typing import Callable, Dict, List, Path
from .dictionary import save_dict_to_csv

__all__ = [
    "extract_tensorboard_logs",
]


def load_all_log_paths(
    root_dir: Path,
) -> List[Path]:
    log_paths: List[Path] = []

    for dirpath, _, fnames in os.walk(root_dir):
        for fname in fnames:
            if "events.out.tfevents" in fname:
                log_paths.append(_path.join(dirpath, fname))

    return log_paths


def extract_scalars_from_log(
    log_path: Path,
) -> None:
    event_acc = EventAccumulator(log_path)
    event_acc.Reload()  # Load the log file

    log_dir, _ = _path.split(log_path)

    for tag in event_acc.Tags()["scalars"]:
        scalar_data: Dict[str, List] = {
            "wall_time": [],
            "step": [],
            "value": [],
        }
        events = event_acc.Scalars(tag)
        for event in events:
            scalar_data["wall_time"].append(event.wall_time)
            scalar_data["step"].append(event.step)
            scalar_data["value"].append(event.value)
        save_dir, save_name = _path.split(
            _path.join(log_dir, "tensorboard/scalars", tag)
        )
        save_dict_to_csv(
            dictionary=scalar_data,
            save_dir=save_dir,
            save_name=save_name,
            index_col="wall_time",
        )


extract_fn_map: Dict[str, Callable] = {
    "scalars": extract_scalars_from_log,
}


def extract_tensorboard_logs(
    root_dir: Path,
    extract: List[str] = [
        "scalars",
    ],
) -> None:
    log_paths = load_all_log_paths(root_dir)
    for log_path in tqdm.tqdm(
        log_paths,
        desc=f"Extracting TensorBoard logs in {root_dir}...",
    ):
        for log_type in extract:
            extract_fn_map[log_type](log_path)
    print("done.")
