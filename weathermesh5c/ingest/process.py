from __future__ import annotations

import csv
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from importlib.resources import files
from math import isfinite
from pathlib import Path

import numpy as np

from ._realchans import real_chans
from .sources import MICROWAVE_STATS, SOURCES


def _hours():
    return defaultdict(lambda: {"x": [], "available_at": []})


def _bufr_module():
    try:
        import ncepbufr
    except ImportError as error:
        raise RuntimeError(
            "Observation processing needs NOAA NCEPLIBS-bufr. "
            "Run wm5c-install-ncepbufr inside the active virtual environment."
        ) from error
    return ncepbufr


def _numeric_reader(bufr):
    """Read numeric subsets with NaNs, reusing a buffer for each query.

    Use the pinned binding's native entry points to avoid constructing masked
    arrays per report. Other bindings can use the public reader. A returned view
    remains valid until the next call with the same query and replication mode.
    """
    module = _bufr_module()
    native = getattr(module, "_bufrlib", None)
    max_levels = getattr(module, "_maxdim", None)
    if native is None or max_levels is None:
        return lambda query, rep=False: bufr.read_subset(query, rep=rep).filled(np.nan)
    missing = bufr.missing_value
    tolerance = 1e-8 + 1e-5 * abs(missing)  # np.ma.masked_values defaults
    buffers = {}

    def read(query, rep=False):
        key = (query, rep)
        if key not in buffers:
            buffers[key] = np.empty((len(query.split()), max_levels), order="F")
        buffer = buffers[key]
        decode = native.ufbrep if rep else native.ufbint
        count = decode(bufr.lunit, buffer, query, *buffer.shape)
        values = buffer[:, :count]
        values[np.abs(values - missing) <= tolerance] = np.nan
        return values

    return read


def _reltime(date: datetime) -> tuple[float, datetime]:
    next_hour = date.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    value = (date - next_hour).total_seconds() / 3600
    if not -1 <= value < 0:
        raise ValueError(f"Observation relative time must be in [-1, 0), got {value}")
    return value, next_hour


def _datetime(header, keys: tuple[str, ...], *, minutes: bool = True, seconds: bool = True):
    values = {key: int(header[index]) for index, key in enumerate(keys)}
    second = min(values.get("second", 0), 59) if seconds else 0
    minute = values.get("minute", 0) if minutes else 0
    return datetime(
        values["year"],
        values["month"],
        values["day"],
        values["hour"],
        minute,
        second,
        tzinfo=timezone.utc,
    )


def _is_missing(value) -> bool:
    return bool(np.ma.is_masked(value) or not isfinite(float(value)))


def _append(hours, hour: datetime, row: np.ndarray, available_at: float) -> None:
    hours[hour]["x"].append(row)
    hours[hour]["available_at"].append(available_at)


def process_microwave(source: str, path: str | Path, *, available_at: float):
    spec = SOURCES[source]
    mean, std = MICROWAVE_STATS[source]
    n_channels = len(mean)
    header_names = (
        "sat_id",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "lat",
        "lon",
        "satellite_zenith",
        "solar_zenith",
        "lat_hd",
        "lon_hd",
    )
    header_query = "SAID YEAR MNTH DAYS HOUR MINU SECO CLAT CLON SAZA SOZA CLATH CLONH"
    indices = {name: spec.columns.index(name) for name in spec.columns}
    hours = _hours()
    bufr = _bufr_module().open(str(path))
    try:
        read = _numeric_reader(bufr)
        while bufr.advance() == 0:
            while bufr.load_subset() == 0:
                header = read(header_query).squeeze()
                values = dict(zip(header_names, header, strict=True))
                lat = values["lat_hd"] if not _is_missing(values["lat_hd"]) else values["lat"]
                lon = values["lon_hd"] if not _is_missing(values["lon_hd"]) else values["lon"]
                required = (
                    lat,
                    lon,
                    values["sat_id"],
                    values["satellite_zenith"],
                    values["solar_zenith"],
                )
                if any(_is_missing(value) for value in required):
                    continue
                date = _datetime(header[1:7], header_names[1:7])
                reltime, next_hour = _reltime(date)
                channels = read("TMBR", rep=True).squeeze()
                if channels.size != n_channels:
                    continue
                channels = (channels - mean) / std
                latdec, latint = np.modf(float(lat))
                londec, lonint = np.modf(float(lon))
                row = np.full(len(spec.columns), np.nan, dtype=spec.dtype)
                row[indices["reltime_hours"]] = reltime
                row[indices["said"]] = values["sat_id"]
                row[indices["latint"]] = latint
                row[indices["latdec"]] = latdec
                row[indices["lonint"]] = lonint
                row[indices["londec"]] = londec
                row[indices["saza_rad"]] = np.deg2rad(values["satellite_zenith"])
                row[indices["soza_rad"]] = np.deg2rad(values["solar_zenith"])
                row[indices["chan_0"] : indices[f"chan_{n_channels - 1}"] + 1] = channels
                _append(hours, next_hour, row, available_at)
    finally:
        bufr.close()
    return hours


def _load_iasi_projection():
    path = files("weathermesh5c.ingest") / "assets/iasi_projection.npz"
    with np.load(path) as projection:
        return projection["weights"], projection["bias"]


@lru_cache(maxsize=128)
def _iasi_channel_scales(table: bytes) -> np.ndarray:
    """Expand a BUFR float64 scale table once for each distinct table."""
    starts, ends, scale_values = np.frombuffer(table, dtype=np.float64).reshape(3, -1)
    valid_scales = np.isfinite(scale_values) & (scale_values < 1e11)
    scales = np.ones(scale_values.shape, dtype=np.float64)
    scales[valid_scales] = 10 ** (-np.round(scale_values[valid_scales]))
    in_bin = (real_chans[:, None] >= starts) & (real_chans[:, None] <= ends)
    selected_scales = np.ones(len(real_chans), dtype=np.float64)
    matched = in_bin.any(axis=1)
    selected_scales[matched] = scales[in_bin.argmax(axis=1)[matched]]
    return selected_scales


def process_iasi(path: str | Path, *, available_at: float):
    spec = SOURCES["mtiasi"]
    weights, bias = _load_iasi_projection()
    header_names = (
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "lat",
        "lon",
        "satellite_zenith",
        "bearing_azimuth",
        "solar_zenith",
        "solar_azimuth",
        "field_of_view",
        "scan_line",
        "quality",
    )
    header_query = (
        "YEAR MNTH DAYS HOUR MINU SECO CLATH CLONH SAZA BEARAZ SOZA SOLAZI FOVN SLNM QGFQ"
    )
    columns = {name: spec.columns.index(name) for name in spec.columns}
    hours = _hours()
    bufr = _bufr_module().open(str(path))
    try:
        read = _numeric_reader(bufr)
        while bufr.advance() == 0:
            while bufr.load_subset() == 0:
                header = read(header_query).squeeze()
                values = dict(zip(header_names, header, strict=True))
                if int(values["quality"]) != 0 or int(values["scan_line"]) == -1:
                    continue
                if _is_missing(values["lat"]) or _is_missing(values["lon"]):
                    continue
                data = read("SCRA CHNM", rep=True).squeeze().astype(np.float32)
                channel_data = read("STCH ENCH CHSF", rep=True)
                if data.shape != (2, 616) or not np.isclose(data[1].sum(), 1_832_093):
                    continue
                selected_scales = _iasi_channel_scales(channel_data.tobytes())
                radiances = np.nan_to_num(data[0] * selected_scales, nan=0)
                projection = radiances @ weights + bias

                date = _datetime(header[:6], header_names[:6])
                reltime, next_hour = _reltime(date)
                latdec, latint = np.modf(float(values["lat"]))
                londec, lonint = np.modf(float(values["lon"]))
                row = np.full(len(spec.columns), np.nan, dtype=spec.dtype)
                row[columns["reltime_hours"]] = reltime
                row[columns["latint"]] = latint
                row[columns["latdec"]] = latdec
                row[columns["lonint"]] = lonint
                row[columns["londec"]] = londec
                for prefix, value in (
                    ("saza", values["satellite_zenith"]),
                    ("bear", values["bearing_azimuth"]),
                    ("soza", values["solar_zenith"]),
                    ("solazi", values["solar_azimuth"]),
                ):
                    row[columns[f"s{prefix}"]] = np.sin(value)
                    row[columns[f"c{prefix}"]] = np.cos(value)
                row[columns["ifovn"]] = ((int(values["field_of_view"]) - 1) / 2) / 60
                row[columns["pca_0"] : columns["pca_65"] + 1] = projection
                _append(hours, next_hour, row, available_at)
    finally:
        bufr.close()
    return hours


def _fit_refractivity(height, value):
    value -= 339.5 * np.exp(-1.343e-4 * height)
    return value / (26.3 * np.exp(-2.6e-4 * height) + 2.13)


def _fit_bending_angle(height, value):
    value -= 0.02839 * np.exp(-1.452e-4 * height)
    return value / (0.0111284 * np.exp(-0.00026371 * height) + 0.00021544)


def _nearest(xi, x, y):
    spacing = np.diff(x) / 2
    centers = x + np.hstack((spacing, spacing[-1]))
    return np.hstack((y, y[-1]))[np.searchsorted(centers, xi)]


def process_gpsro(path: str | Path, *, available_at: float):
    spec = SOURCES["gpsro"]
    columns = {name: spec.columns.index(name) for name in spec.columns}
    hours = _hours()
    bufr = _bufr_module().open(str(path))
    try:
        while bufr.advance() == 0:
            while bufr.load_subset() == 0:
                header = bufr.read_subset("YEAR MNTH DAYS HOUR MINU PCCF").squeeze()
                if _is_missing(header[5]) or int(header[5]) == 0:
                    continue
                first = bufr.read_subset("ROSEQ1", seq=True).filled(np.nan)
                counts = bufr.read_subset("{ROSEQ2}").squeeze().filled(np.nan)
                third = bufr.read_subset("ROSEQ3", seq=True).filled(np.nan)
                profile = {name: [] for name in ("lat", "lon", "height", "refractivity", "bending")}
                for level in range(third.shape[1]):
                    lat, lon = first[0, level], first[1, level]
                    height, refractivity = third[0, level], third[1, level]
                    if not all(np.isfinite(value) for value in (lat, lon, height, refractivity)):
                        continue
                    bending = np.nan
                    for index in range(int(counts[level])):
                        base = 6 * (index + 1) - 3
                        frequency = first[base, level]
                        bending = first[base + 2, level]
                        if int(frequency) == 0:
                            break
                    refractivity = _fit_refractivity(height, refractivity)
                    bending = _fit_bending_angle(height, bending)
                    if not np.isfinite(bending):
                        continue
                    if 0 <= height <= 30_000 and (abs(bending) > 4.5 or abs(refractivity) > 5):
                        continue
                    for name, value in zip(
                        profile, (lat, lon, height, refractivity, bending), strict=True
                    ):
                        profile[name].append(value)
                if len(profile["height"]) < 2:
                    continue
                order = np.argsort(profile["height"])
                # Round profiles before interpolation, as in the training pipeline.
                values = {
                    name: np.asarray(data, dtype=spec.dtype)[order]
                    for name, data in profile.items()
                }
                altitude = np.arange(0, 30_000, 100)
                lat = _nearest(altitude, values["height"], values["lat"])
                lon = _nearest(altitude, values["height"], values["lon"])
                refractivity = np.interp(
                    altitude,
                    values["height"],
                    values["refractivity"],
                    left=np.nan,
                    right=np.nan,
                )
                bending = np.interp(
                    altitude,
                    values["height"],
                    values["bending"],
                    left=np.nan,
                    right=np.nan,
                )
                present = np.isfinite(refractivity) & np.isfinite(bending)
                latdec, latint = np.modf(lat)
                londec, lonint = np.modf(lon)
                date = _datetime(
                    header[:5],
                    ("year", "month", "day", "hour", "minute"),
                    seconds=False,
                )
                reltime, next_hour = _reltime(date)
                data = np.full((300, len(spec.columns)), np.nan, dtype=spec.dtype)
                data[:, columns["reltime_hours"]] = reltime
                data[:, columns["latint"]] = latint
                data[:, columns["latdec"]] = latdec
                data[:, columns["lonint"]] = lonint
                data[:, columns["londec"]] = londec
                data[:, columns["rint"]] = np.nan_to_num(refractivity)
                data[:, columns["bint"]] = np.nan_to_num(bending)
                data[:, columns["all_present"]] = present
                _append(hours, next_hour, data, available_at)
    finally:
        bufr.close()
    return hours


def process_satwind(path: str | Path, *, available_at: float):
    spec = SOURCES["satwnd"]
    columns = {name: spec.columns.index(name) for name in spec.columns}
    names = (
        "sat_id",
        "year",
        "month",
        "day",
        "hour",
        "minute",
        "second",
        "lat_hd",
        "lon_hd",
        "lat",
        "lon",
        "satellite_zenith",
    )
    query = "SAID YEAR MNTH DAYS HOUR MINU SECO CLATH CLONH CLAT CLON SAZA"
    hours = _hours()
    bufr = _bufr_module().open(str(path))
    try:
        read = _numeric_reader(bufr)
        while bufr.advance() == 0:
            while bufr.load_subset() == 0:
                header = read(query).squeeze()
                values = dict(zip(names, header, strict=True))
                if _is_missing(values["satellite_zenith"]):
                    continue
                data = read("PRLC WDIR WSPD").squeeze()
                if data.ndim != 1 or len(data) != 3 or any(_is_missing(value) for value in data):
                    continue
                lat = values["lat_hd"] if not _is_missing(values["lat_hd"]) else values["lat"]
                lon = values["lon_hd"] if not _is_missing(values["lon_hd"]) else values["lon"]
                if _is_missing(lat) or _is_missing(lon):
                    continue
                date = _datetime(header[1:7], names[1:7])
                reltime, next_hour = _reltime(date)
                pressure, direction, speed = data
                row = np.full(len(spec.columns), np.nan, dtype=spec.dtype)
                row[columns["id"]] = values["sat_id"]
                row[columns["reltime_hours"]] = reltime
                row[columns["lat_deg"]] = lat
                row[columns["lon_deg"]] = lon
                row[columns["satzenith_deg"]] = values["satellite_zenith"]
                row[columns["pres_pa"]] = pressure
                row[columns["unnorm_ucomp_mps"]] = -speed * np.sin(np.deg2rad(direction))
                row[columns["unnorm_vcomp_mps"]] = -speed * np.cos(np.deg2rad(direction))
                _append(hours, next_hour, row, available_at)
    finally:
        bufr.close()
    return hours


def _as_vector(value) -> np.ndarray:
    return np.atleast_1d(np.ma.asarray(value).filled(np.nan))


def process_radiosonde(path: str | Path, *, available_at: float):
    spec = SOURCES["adpupa"]
    columns = {name: spec.columns.index(name) for name in spec.columns}
    hours = _hours()
    bufr = _bufr_module().open(str(path))
    try:
        while bufr.advance() == 0:
            while bufr.load_subset() == 0:
                header = bufr.read_subset("RPID YEAR MNTH DAYS HOUR CLAT CLON").squeeze()
                if _is_missing(header[-2]) or _is_missing(header[-1]):
                    continue
                data = bufr.read_subset("PRLC WDIR WSPD TMDB TMDP").squeeze()
                values = [_as_vector(data[index]) for index in range(5)]
                if len({len(value) for value in values}) != 1:
                    continue
                date = _datetime(
                    header[1:5], ("year", "month", "day", "hour"), minutes=False, seconds=False
                )
                reltime, next_hour = _reltime(date)
                for pressure, direction, speed, temperature, dewpoint in zip(*values, strict=True):
                    if not np.isfinite(pressure):
                        continue
                    wind_present = np.isfinite(direction) and np.isfinite(speed)
                    temperature_present = np.isfinite(temperature)
                    humidity_present = temperature_present and np.isfinite(dewpoint)
                    ucomp = -speed * np.sin(np.deg2rad(direction)) if wind_present else np.nan
                    vcomp = -speed * np.cos(np.deg2rad(direction)) if wind_present else np.nan
                    humidity = np.nan
                    if humidity_present and temperature != 0 and dewpoint != 0:
                        humidity = np.exp((17.625 * dewpoint) / (243.04 + dewpoint)) / np.exp(
                            (17.625 * temperature) / (243.04 + temperature)
                        )
                    row = np.full(len(spec.columns), np.nan, dtype=spec.dtype)
                    row[columns["id"]] = int(header[0])
                    row[columns["reltime_hours"]] = reltime
                    row[columns["lat_deg"]] = header[-2]
                    row[columns["lon_deg"]] = header[-1]
                    row[columns["pres_pa"]] = pressure
                    row[columns["unnorm_ucomp_mps"]] = ucomp
                    row[columns["unnorm_vcomp_mps"]] = vcomp
                    row[columns["unnorm_temp_k"]] = temperature
                    row[columns["unnorm_rh"]] = humidity
                    _append(hours, next_hour, row, available_at)
    finally:
        bufr.close()
    return hours


def _windborne_number(values: Mapping[str, object], name: str) -> float:
    value = values.get(name)
    return float(value) if value not in (None, "") else np.nan


def process_windborne_observations(
    observations: Iterable[Mapping[str, object]],
    *,
    available_at: float,
):
    spec = SOURCES["windborne"]
    columns = {name: spec.columns.index(name) for name in spec.columns}
    hours = _hours()
    for values in observations:
        pressure = _windborne_number(values, "pressure") * 100
        if not np.isfinite(pressure):
            continue
        date = datetime.fromtimestamp(
            _windborne_number(values, "timestamp"),
            tz=timezone.utc,
        )
        reltime, next_hour = _reltime(date)
        row = np.full(len(spec.columns), np.nan, dtype=spec.dtype)
        for column, value in (
            ("id", _windborne_number(values, "id")),
            ("reltime_hours", reltime),
            ("lat_deg", _windborne_number(values, "latitude")),
            ("lon_deg", _windborne_number(values, "longitude")),
            ("pres_pa", pressure),
            ("unnorm_ucomp_mps", _windborne_number(values, "speed_u")),
            ("unnorm_vcomp_mps", _windborne_number(values, "speed_v")),
            ("unnorm_temp_k", _windborne_number(values, "temperature") + 273.15),
            ("unnorm_rh", _windborne_number(values, "humidity")),
            ("unnorm_sh_mgpkg", _windborne_number(values, "specific_humidity")),
        ):
            row[columns[column]] = value
        _append(hours, next_hour, row, available_at)
    return hours


def process_windborne(path: str | Path, *, available_at: float):
    with Path(path).open(newline="") as stream:
        return process_windborne_observations(
            csv.DictReader(stream),
            available_at=available_at,
        )


_SURFACE_IDENTITY_QUERY = "YEAR MNTH DAYS HOUR MINU CLAT CLON SELV"
_SURFACE_VALUE_QUERY = "PRES PMSL ALSE WDIR WSPD TMDB TMDP REHU HOVI TP03 TP24"


def _surface_report_kind(message_type: str) -> bool | None:
    if len(message_type) != 8 or not message_type.startswith("NC"):
        return None
    record_code = message_type[2:5]
    if record_code not in {"000", "001"}:
        return None
    try:
        subtype = int(message_type[5:])
    except ValueError:
        return None
    # GDAS carries duplicate 1xx WMO templates beside the operational NCEP reports.
    return record_code == "001" if 0 <= subtype < 100 else None


def _surface_value(value, *, scale: float = 1, offset: float = 0, digits: int = 1) -> float:
    if _is_missing(value):
        return np.nan
    converted = float(value) * scale + offset
    return float(f"{converted:.{digits}f}")


def process_surface(path: str | Path, *, available_at: float):
    spec = SOURCES["adpsfc"]
    columns = {name: spec.columns.index(name) for name in spec.columns}
    hours = _hours()
    bufr = _bufr_module().open(str(path))
    try:
        while bufr.advance() == 0:
            is_ship = _surface_report_kind(bufr.msg_type)
            if is_ship is None:
                continue
            while bufr.load_subset() == 0:
                identity = bufr.read_subset(_SURFACE_IDENTITY_QUERY).squeeze()
                basic = bufr.read_subset(_SURFACE_VALUE_QUERY).squeeze()
                if np.ma.getmaskarray(identity[:7]).any():
                    continue
                if np.ma.getmaskarray(basic).all():
                    continue

                date = _datetime(
                    identity[:5],
                    ("year", "month", "day", "hour", "minute"),
                    seconds=False,
                )
                reltime, next_hour = _reltime(date)
                lat = _surface_value(identity[5], digits=2)
                lon = _surface_value(identity[6], digits=2)
                if not -90 <= lat <= 90 or not -180 <= lon <= 180:
                    continue

                elevation = 99999.0 if _is_missing(identity[7]) else float(int(identity[7]))
                direction = _surface_value(basic[3])
                speed = _surface_value(basic[4])
                values = (
                    is_ship,
                    reltime,
                    lat,
                    lon,
                    elevation,
                    _surface_value(basic[1], scale=0.01),
                    _surface_value(basic[5], offset=-273.16),
                    _surface_value(basic[6], offset=-273.16),
                    -speed * np.sin(np.deg2rad(direction)),
                    -speed * np.cos(np.deg2rad(direction)),
                    _surface_value(basic[8]),
                    _surface_value(basic[9]),
                    _surface_value(basic[10]),
                )
                row = np.asarray(values, dtype=spec.dtype)
                required = [columns[name] for name in ("reltime_hours", "lat_deg", "lon_deg")]
                if not np.isnan(row[required]).any():
                    _append(hours, next_hour, row, available_at)
    finally:
        bufr.close()
    return hours


def process_file(
    source: str,
    path: str | Path,
    *,
    available_at: float | None = None,
):
    available_at = time.time() if available_at is None else available_at
    if source in MICROWAVE_STATS:
        return process_microwave(source, path, available_at=available_at)
    if source == "mtiasi":
        return process_iasi(path, available_at=available_at)
    if source == "gpsro":
        return process_gpsro(path, available_at=available_at)
    if source == "satwnd":
        return process_satwind(path, available_at=available_at)
    if source == "adpupa":
        return process_radiosonde(path, available_at=available_at)
    if source == "windborne":
        return process_windborne(path, available_at=available_at)
    if source == "adpsfc":
        return process_surface(path, available_at=available_at)
    raise ValueError(f"Unknown observation source: {source}")
