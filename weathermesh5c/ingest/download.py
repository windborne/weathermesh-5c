from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

from .sources import SOURCES

NOMADS_OBSPROC = "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod"
FEEDS = ("gdas", "gfs", "rap")


def normalize_cycle(cycle: datetime) -> datetime:
    if cycle.tzinfo is None:
        cycle = cycle.replace(tzinfo=timezone.utc)
    return cycle.astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)


def obsproc_url(source: str, cycle: datetime, *, feed: str = "gdas") -> str:
    spec = SOURCES[source]
    cycle = normalize_cycle(cycle)
    if source == "windborne":
        raise ValueError("WindBorne observations do not come from NOAA NOMADS")
    if feed not in FEEDS:
        raise ValueError(f"Unknown observation feed: {feed}")
    if feed in {"gdas", "gfs"} and cycle.hour % 6:
        raise ValueError(f"{feed} observation cycles must be at 00, 06, 12, or 18 UTC")
    filename = f"{feed}.t{cycle:%H}z.{source}.tm00.{spec.extension}"
    return f"{NOMADS_OBSPROC}/{feed}.{cycle:%Y%m%d}/{filename}"


def raw_path(root: str | Path, source: str, cycle: datetime, *, feed: str = "gdas") -> Path:
    spec = SOURCES[source]
    cycle = normalize_cycle(cycle)
    filename = f"{feed}.t{cycle:%H}z.{source}.tm00.{spec.extension}"
    return Path(root) / source / feed / f"{cycle:%Y%m%d}" / filename


def is_available(
    source: str,
    cycle: datetime,
    *,
    feed: str = "gdas",
    session: requests.Session | None = None,
) -> bool:
    client = session or requests.Session()
    try:
        response = client.head(
            obsproc_url(source, cycle, feed=feed),
            allow_redirects=True,
            timeout=(10, 20),
        )
        return response.status_code == requests.codes.ok
    except requests.RequestException:
        return False


def latest_cycle(
    source: str,
    *,
    feed: str = "gdas",
    now: datetime | None = None,
    lookback: int = 8,
    session: requests.Session | None = None,
) -> datetime:
    now = normalize_cycle(now or datetime.now(timezone.utc))
    step = 1 if feed == "rap" else 6
    candidate = now.replace(hour=(now.hour // step) * step)
    for offset in range(lookback):
        cycle = candidate - timedelta(hours=offset * step)
        if is_available(source, cycle, feed=feed, session=session):
            return cycle
    raise FileNotFoundError(f"No {feed} {source} file was available in the last {lookback} cycles")


def latest_common_cycle(
    sources: tuple[str, ...],
    *,
    feed: str = "gdas",
    now: datetime | None = None,
    lookback: int = 8,
    session: requests.Session | None = None,
) -> datetime:
    if not sources:
        raise ValueError("At least one observation source is required")
    now = normalize_cycle(now or datetime.now(timezone.utc))
    step = 1 if feed == "rap" else 6
    candidate = now.replace(hour=(now.hour // step) * step)
    for offset in range(lookback):
        cycle = candidate - timedelta(hours=offset * step)
        if all(is_available(source, cycle, feed=feed, session=session) for source in sources):
            return cycle
    names = ", ".join(sources)
    raise FileNotFoundError(
        f"No common {feed} cycle for {names} was available in the last {lookback} cycles"
    )


def download(
    source: str,
    root: str | Path,
    *,
    cycle: datetime | None = None,
    feed: str = "gdas",
    session: requests.Session | None = None,
) -> Path:
    client = session or requests.Session()
    cycle = cycle or latest_cycle(source, feed=feed, session=client)
    destination = raw_path(root, source, cycle, feed=feed)
    if destination.is_file() and destination.stat().st_size:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    response = client.get(
        obsproc_url(source, cycle, feed=feed),
        stream=True,
        timeout=(10, 600),
    )
    response.raise_for_status()

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        temporary = Path(temporary_name)
        with temporary.open("rb") as downloaded:
            signature = downloaded.read(4)
        if temporary.stat().st_size < 8 or signature != b"BUFR":
            url = obsproc_url(source, cycle, feed=feed)
            raise ValueError(f"Downloaded file is not BUFR: {url}")
        os.replace(temporary, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    finally:
        response.close()
    return destination
