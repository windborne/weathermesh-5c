from __future__ import annotations

import fcntl
import os
import tempfile
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def processed_path(root: str | Path, source: str, hour: datetime) -> Path:
    if hour.tzinfo is None:
        hour = hour.replace(tzinfo=timezone.utc)
    hour = hour.astimezone(timezone.utc)
    return Path(root) / source / f"{hour:%Y%m}" / f"{int(hour.timestamp())}.npz"


def _deduplicate(x: np.ndarray, available_at: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not len(x):
        return x, available_at
    rows = np.ascontiguousarray(x).reshape(len(x), -1)
    tokens = rows.view(np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))).ravel()
    _, indices = np.unique(tokens, return_index=True)
    indices.sort()
    return x[indices], available_at[indices]


def write_hours(
    root: str | Path,
    source: str,
    hours: Mapping[datetime, Mapping[str, list | np.ndarray]],
) -> list[Path]:
    written = []
    for hour, payload in sorted(hours.items()):
        path = processed_path(root, source, hour)
        path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = path.with_suffix(path.suffix + ".lock")
        with lock_path.open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                x = np.asarray(payload["x"])
                available_at = np.asarray(payload["available_at"], dtype=np.float64)
                if path.exists():
                    with np.load(path) as previous:
                        x = np.concatenate((previous["x"], x))
                        available_at = np.concatenate((previous["available_at"], available_at))
                x, available_at = _deduplicate(x, available_at)
                descriptor, temporary_name = tempfile.mkstemp(
                    prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
                )
                try:
                    with os.fdopen(descriptor, "wb") as temporary:
                        np.savez_compressed(temporary, x=x, available_at=available_at)
                        temporary.flush()
                        os.fsync(temporary.fileno())
                    os.replace(temporary_name, path)
                except Exception:
                    Path(temporary_name).unlink(missing_ok=True)
                    raise
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        written.append(path)
    return written
