from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta, timezone

from .process import process_windborne_observations

OBSERVATION_WINDOW_HOURS = (-1.5, 1.5)
REQUEST_MINUTES = 10
MAX_PAGES_PER_CHUNK = 100


def _super_observation_client():
    try:
        from windborne import get_super_observations_page
    except ImportError as error:
        raise RuntimeError(
            "WindBorne ingestion needs the official WindBorne Python client. "
            "Install the 'windborne' package extra."
        ) from error
    return get_super_observations_page


def _utc_hour(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    value = value.astimezone(timezone.utc)
    if value.minute or value.second or value.microsecond:
        raise ValueError("WindBorne valid time must be on an exact UTC hour")
    return value


def _integer(value: object, name: str) -> int:
    number = float(value)
    if not number.is_integer():
        raise ValueError(f"WindBorne {name} must be an integer, got {value!r}")
    return int(number)


def _deduplicate(observations: list[Mapping[str, object]]) -> list[Mapping[str, object]]:
    by_id: dict[int, Mapping[str, object]] = {}
    for observation in observations:
        if observation.get("id") is None:
            raise ValueError("WindBorne observation is missing id")
        observation_id = _integer(observation["id"], "id")
        previous = by_id.get(observation_id)
        if previous is None:
            by_id[observation_id] = observation
            continue
        previous_update = _integer(previous.get("updated_at") or 0, "updated_at")
        current_update = _integer(observation.get("updated_at") or 0, "updated_at")
        if current_update > previous_update:
            by_id[observation_id] = observation
        elif current_update == previous_update and observation != previous:
            raise ValueError(f"WindBorne observation id {observation_id} has conflicting records")
    return sorted(
        by_id.values(),
        key=lambda observation: (
            _integer(observation["timestamp"], "timestamp"),
            _integer(observation["id"], "id"),
        ),
    )


def fetch_windborne(
    valid_time: datetime,
    *,
    available_at: float | None = None,
    request_minutes: int = REQUEST_MINUTES,
    max_pages_per_chunk: int = MAX_PAGES_PER_CHUNK,
    page_client: Callable[..., Mapping[str, object] | None] | None = None,
):
    """Fetch and convert official WindBorne super-observations for one AIDA window."""

    valid_time = _utc_hour(valid_time)
    if request_minutes < 1:
        raise ValueError("WindBorne request_minutes must be positive")
    if max_pages_per_chunk < 1:
        raise ValueError("WindBorne max_pages_per_chunk must be positive")

    start = int((valid_time + timedelta(hours=OBSERVATION_WINDOW_HOURS[0])).timestamp())
    end = int((valid_time + timedelta(hours=OBSERVATION_WINDOW_HOURS[1])).timestamp())
    request_seconds = request_minutes * 60
    client = page_client or _super_observation_client()
    observations: list[Mapping[str, object]] = []

    chunk_start = start
    while chunk_start <= end:
        next_chunk = min(chunk_start + request_seconds, end)
        chunk_end = next_chunk if next_chunk == end else next_chunk - 1
        cursor = chunk_start
        seen_cursors = {cursor}

        for page_number in range(1, max_pages_per_chunk + 1):
            response = client(
                since=cursor,
                min_time=chunk_start,
                max_time=chunk_end,
                include_ids=True,
                include_mission_name=True,
                include_updated_at=True,
            )
            if not isinstance(response, Mapping):
                raise RuntimeError("WindBorne API returned an invalid response")
            page = response.get("observations", [])
            if not isinstance(page, list):
                raise RuntimeError("WindBorne API observations payload must be a list")
            for observation in page:
                if not isinstance(observation, Mapping):
                    raise RuntimeError("WindBorne API returned a non-object observation")
                if observation.get("timestamp") is None:
                    raise ValueError("WindBorne observation is missing timestamp")
                timestamp = _integer(observation["timestamp"], "timestamp")
                if not chunk_start <= timestamp <= chunk_end:
                    raise ValueError(
                        "WindBorne API returned an observation outside the requested window"
                    )
                observations.append(observation)

            if not response.get("has_next_page"):
                break
            if page_number == max_pages_per_chunk:
                raise RuntimeError(
                    f"WindBorne API exceeded {max_pages_per_chunk} pages for one request"
                )
            if response.get("next_since") is None:
                raise RuntimeError("WindBorne paginated response is missing next_since")
            next_cursor = _integer(response["next_since"], "next_since")
            if next_cursor <= cursor or next_cursor in seen_cursors:
                raise RuntimeError("WindBorne API pagination cursor did not move forward")
            seen_cursors.add(next_cursor)
            cursor = next_cursor

        chunk_start = chunk_end + 1

    return process_windborne_observations(
        _deduplicate(observations),
        available_at=time.time() if available_at is None else available_at,
    )
