from __future__ import annotations

import json
import os
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import requests

OPEN_IFS_ROOT = "https://data.ecmwf.int/forecasts"
OISST_ROOT = (
    "https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr"
)
OPEN_LEVELS = (10, 50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
PRESSURE_PARAMS = {
    "129_z": ("gh", "z"),
    "130_t": ("t",),
    "131_u": ("u",),
    "132_v": ("v",),
    "133_q": ("q",),
}
SURFACE_PARAMS = {
    "165_10u": "10u",
    "166_10v": "10v",
    "167_2t": "2t",
    "151_msl": "msl",
    "168_2d": "2d",
    "246_100u": "100u",
    "247_100v": "100v",
    "skt": "skt",
    "lsm": "lsm",
}


@dataclass(frozen=True, order=True)
class IndexEntry:
    offset: int
    length: int
    param: str
    level: int | None


def normalize_cycle(cycle: datetime) -> datetime:
    if cycle.tzinfo is None:
        cycle = cycle.replace(tzinfo=timezone.utc)
    cycle = cycle.astimezone(timezone.utc)
    if cycle.minute or cycle.second or cycle.microsecond or cycle.hour % 6:
        raise ValueError("Open IFS cycles must be at 00, 06, 12, or 18 UTC")
    return cycle


def validate_member(member: int) -> int:
    if not 1 <= member <= 50:
        raise ValueError("Open IFS perturbed ensemble members are numbered 1 through 50")
    return member


def open_ifs_urls(cycle: datetime) -> tuple[str, str]:
    cycle = normalize_cycle(cycle)
    stem = f"{cycle:%Y%m%d%H}0000-0h-enfo-ef"
    root = f"{OPEN_IFS_ROOT}/{cycle:%Y%m%d}/{cycle:%H}z/ifs/0p25/enfo"
    return f"{root}/{stem}.index", f"{root}/{stem}.grib2"


def open_ifs_path(root: str | Path, cycle: datetime, member: int) -> Path:
    cycle = normalize_cycle(cycle)
    validate_member(member)
    return (
        Path(root)
        / "ecmwf-open"
        / f"{cycle:%Y%m}"
        / f"{cycle:%Y%m%d%H}0000-0h-enfo-ef-member{member:02d}.grib2"
    )


def _oisst_names(day: datetime) -> tuple[str, str]:
    stem = f"oisst-avhrr-v02r01.{day:%Y%m%d}"
    return f"{stem}_preliminary.nc", f"{stem}.nc"


def _oisst_paths(root: str | Path, day: datetime) -> tuple[Path, Path]:
    directory = Path(root) / "noaa-oisst" / f"{day:%Y%m}"
    return tuple(directory / name for name in _oisst_names(day))


def _oisst_candidates(root: str | Path, cycle: datetime, lookback_days: int):
    if lookback_days < 1:
        raise ValueError("lookback_days must be positive")
    cycle = normalize_cycle(cycle)
    for offset in range(1, lookback_days + 1):
        day = cycle - timedelta(days=offset)
        yield day, _oisst_names(day), _oisst_paths(root, day)


def existing_oisst_path(
    root: str | Path,
    cycle: datetime,
    *,
    lookback_days: int = 7,
) -> Path:
    for _, _, paths in _oisst_candidates(root, cycle, lookback_days):
        for path in paths:
            if path.is_file() and path.stat().st_size:
                return path
    raise FileNotFoundError(f"No NOAA OISST file is present for {normalize_cycle(cycle):%Y%m%d%H}")


def _request(
    client,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    stream: bool = False,
    timeout: tuple[int, int] = (10, 90),
):
    error: Exception | None = None
    for attempt in range(4):
        response = None
        try:
            response = client.get(url, headers=headers, stream=stream, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if response is not None:
                response.close()
            error = exc
            if getattr(exc.response, "status_code", None) == requests.codes.not_found:
                raise
            if attempt < 3:
                time.sleep(2**attempt)
    raise RuntimeError(f"Failed to download {url}") from error


def _parse_index(text: str) -> list[dict[str, object]]:
    rows = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid ECMWF index JSON on line {line_number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Invalid ECMWF index row on line {line_number}")
        rows.append(row)
    if not rows:
        raise ValueError("The ECMWF index is empty")
    return rows


def _matches(row: dict[str, object], *, member: int, levtype: str) -> bool:
    try:
        return (
            int(row.get("number", -1)) == member
            and int(row.get("step", -1)) == 0
            and row.get("levtype") == levtype
        )
    except (TypeError, ValueError):
        return False


def _entry(row: dict[str, object]) -> IndexEntry:
    try:
        level = row.get("levelist")
        return IndexEntry(
            offset=int(row["_offset"]),
            length=int(row["_length"]),
            param=str(row["param"]),
            level=None if level is None else int(level),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid ECMWF index entry: {row}") from error


def select_index_entries(text: str, *, member: int) -> list[IndexEntry]:
    validate_member(member)
    rows = _parse_index(text)
    pressure_rows = [row for row in rows if _matches(row, member=member, levtype="pl")]
    surface_rows = [row for row in rows if _matches(row, member=member, levtype="sfc")]
    selected: list[IndexEntry] = []
    missing = []

    for variable, aliases in PRESSURE_PARAMS.items():
        for level in OPEN_LEVELS:
            matches = [
                row
                for alias in aliases
                for row in pressure_rows
                if row.get("param") == alias and int(row.get("levelist", -1)) == level
            ]
            if not matches:
                missing.append(f"{variable}@{level}hPa")
                continue
            selected.append(_entry(matches[0]))

    for variable, param in SURFACE_PARAMS.items():
        matches = [row for row in surface_rows if row.get("param") == param]
        if not matches:
            missing.append(variable)
            continue
        selected.append(_entry(matches[0]))

    if missing:
        raise ValueError(f"Open IFS index is missing: {', '.join(missing)}")
    if len({entry.offset for entry in selected}) != len(selected):
        raise ValueError("The ECMWF index selected duplicate GRIB messages")
    return sorted(selected)


def _download_range(client, url: str, entry: IndexEntry) -> tuple[int, bytes]:
    end = entry.offset + entry.length - 1
    response = _request(client, url, headers={"Range": f"bytes={entry.offset}-{end}"})
    try:
        if response.status_code != requests.codes.partial_content:
            raise ValueError(f"ECMWF range request returned HTTP {response.status_code}")
        if len(response.content) != entry.length:
            raise ValueError(
                f"ECMWF range length was {len(response.content)}, expected {entry.length}"
            )
        return entry.offset, response.content
    finally:
        response.close()


def download_open_ifs(
    root: str | Path,
    cycle: datetime,
    *,
    member: int = 1,
    workers: int = 8,
    session: requests.Session | None = None,
) -> Path:
    cycle = normalize_cycle(cycle)
    destination = open_ifs_path(root, cycle, member)
    if destination.is_file() and destination.stat().st_size:
        return destination
    if workers < 1:
        raise ValueError("workers must be positive")

    client = session or requests.Session()
    index_url, data_url = open_ifs_urls(cycle)
    response = _request(client, index_url)
    try:
        entries = select_index_entries(response.text, member=member)
    finally:
        response.close()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        messages = sorted(pool.map(lambda entry: _download_range(client, data_url, entry), entries))
    payload_size = sum(len(message) for _, message in messages)
    expected_size = sum(entry.length for entry in entries)
    if payload_size != expected_size:
        raise ValueError(f"Downloaded {payload_size} ECMWF bytes, expected {expected_size}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            for _, message in messages:
                output.write(message)
            output.flush()
            os.fsync(output.fileno())
        temporary = Path(temporary_name)
        with temporary.open("rb") as downloaded:
            first = downloaded.read(4)
            downloaded.seek(-4, os.SEEK_END)
            last = downloaded.read(4)
        if first != b"GRIB" or last != b"7777":
            raise ValueError("Downloaded ECMWF payload is not a complete GRIB stream")
        os.replace(temporary, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def download_oisst(
    root: str | Path,
    cycle: datetime,
    *,
    lookback_days: int = 7,
    session: requests.Session | None = None,
) -> Path:
    cycle = normalize_cycle(cycle)
    try:
        return existing_oisst_path(root, cycle, lookback_days=lookback_days)
    except FileNotFoundError:
        pass

    client = session or requests.Session()
    last_error: Exception | None = None
    for day, names, paths in _oisst_candidates(root, cycle, lookback_days):
        for name, destination in zip(names, paths, strict=True):
            url = f"{OISST_ROOT}/{day:%Y%m}/{name}"
            try:
                response = _request(client, url, stream=True, timeout=(10, 600))
            except requests.HTTPError as error:
                last_error = error
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
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
                if temporary.stat().st_size < 1024 or signature not in {
                    b"\x89HDF",
                    b"CDF\x01",
                    b"CDF\x02",
                }:
                    raise ValueError(f"Downloaded NOAA file is not NetCDF: {url}")
                os.replace(temporary, destination)
                return destination
            except Exception:
                Path(temporary_name).unlink(missing_ok=True)
                raise
            finally:
                response.close()
    raise FileNotFoundError(
        f"NOAA OISST is unavailable in the {lookback_days} days before {cycle:%Y-%m-%d}"
    ) from last_error


def latest_open_cycle(
    *,
    now: datetime | None = None,
    lookback: int = 12,
    session: requests.Session | None = None,
) -> datetime:
    if lookback < 1:
        raise ValueError("lookback must be positive")
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    now = now.astimezone(timezone.utc)
    candidate = now.replace(hour=(now.hour // 6) * 6, minute=0, second=0, microsecond=0)
    client = session or requests.Session()
    for offset in range(lookback):
        cycle = candidate - timedelta(hours=6 * offset)
        index_url, _ = open_ifs_urls(cycle)
        try:
            response = client.get(index_url, timeout=(10, 30))
        except requests.RequestException:
            continue
        try:
            if response.status_code == requests.codes.ok and response.text.strip():
                return cycle
        finally:
            response.close()
    raise FileNotFoundError(f"No Open IFS ensemble cycle was available in the last {lookback} runs")
