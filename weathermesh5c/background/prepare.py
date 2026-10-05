from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from functools import cache
from importlib.resources import files
from pathlib import Path

import netCDF4
import numpy as np
import pygrib
from scipy.interpolate import RegularGridInterpolator

from .download import (
    OPEN_LEVELS,
    PRESSURE_PARAMS,
    SURFACE_PARAMS,
    normalize_cycle,
    validate_member,
)

GRAVITY = np.float32(9.80665)
MODEL_LEVELS = (10, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
ENCODER_LEVELS = (
    10,
    30,
    50,
    70,
    100,
    150,
    200,
    250,
    300,
    350,
    400,
    450,
    500,
    550,
    600,
    650,
    700,
    750,
    800,
    850,
    900,
    925,
    950,
    975,
    1000,
)
CORE_SURFACE_VARS = ("165_10u", "166_10v", "167_2t", "151_msl")
EXTRA_SURFACE_VARS = ("168_2d", "246_100u", "247_100v")
BACKGROUND_CHANNELS = len(PRESSURE_PARAMS) * len(ENCODER_LEVELS) + 12
GRID_SHAPE = (720, 1440)


@dataclass(frozen=True)
class OpenIFSFields:
    pressure: dict[str, dict[int, np.ndarray]]
    surface: dict[str, np.ndarray]

    def validate(self, expected_shape: tuple[int, int] | None = GRID_SHAPE) -> tuple[int, int]:
        shapes = []
        for variable in PRESSURE_PARAMS:
            levels = self.pressure.get(variable, {})
            if tuple(sorted(levels)) != OPEN_LEVELS:
                raise ValueError(
                    f"{variable} levels are {sorted(levels)}, expected {list(OPEN_LEVELS)}"
                )
            shapes.extend(values.shape for values in levels.values())
        for variable in SURFACE_PARAMS:
            if variable not in self.surface:
                raise ValueError(f"Open IFS surface field is missing {variable}")
            shapes.append(self.surface[variable].shape)
        if not shapes or any(shape != shapes[0] for shape in shapes):
            raise ValueError("Open IFS fields do not share one grid shape")
        if expected_shape is not None and shapes[0] != expected_shape:
            raise ValueError(f"Open IFS grid is {shapes[0]}, expected {expected_shape}")
        return shapes[0]


@dataclass(frozen=True)
class PreparedBackground:
    values: np.ndarray
    cycle: datetime
    member: int
    sst_date: date | None


@dataclass(frozen=True)
class BackgroundInfo:
    cycle: datetime
    member: int
    sst_date: date | None


@cache
def _norms() -> dict[str, object]:
    path = files("weathermesh5c._core.norm") / "norm.json"
    with path.open() as stream:
        return json.load(stream)


def _aligned_values(message) -> np.ndarray:
    values = np.asarray(message.values, dtype=np.float32)
    if values.shape != (721, 1440):
        raise ValueError(f"Open IFS GRIB grid is {values.shape}, expected (721, 1440)")
    first_latitude = float(message.latitudeOfFirstGridPointInDegrees)
    if math.isclose(first_latitude, -90.0):
        values = values[::-1]
    elif not math.isclose(first_latitude, 90.0):
        raise ValueError(f"Unsupported first GRIB latitude: {first_latitude}")
    first_longitude = float(message.longitudeOfFirstGridPointInDegrees)
    if math.isclose(abs(first_longitude), 180.0):
        values = np.roll(values, -720, axis=1)
    elif not math.isclose(first_longitude, 0.0):
        raise ValueError(f"Unsupported first GRIB longitude: {first_longitude}")
    return np.ascontiguousarray(values[:720])


def load_open_ifs(path: str | Path) -> OpenIFSFields:
    pressure = {variable: {} for variable in PRESSURE_PARAMS}
    surface: dict[str, np.ndarray] = {}
    pressure_aliases = {
        alias: variable for variable, aliases in PRESSURE_PARAMS.items() for alias in aliases
    }
    surface_aliases = {alias: variable for variable, alias in SURFACE_PARAMS.items()}

    with pygrib.open(str(path)) as messages:
        for message in messages:
            name = message.shortName
            if name in pressure_aliases:
                variable = pressure_aliases[name]
                values = _aligned_values(message)
                if name == "gh":
                    values *= GRAVITY
                pressure[variable][int(message.level)] = values
            elif name in surface_aliases:
                surface[surface_aliases[name]] = _aligned_values(message)

    fields = OpenIFSFields(pressure, surface)
    fields.validate()
    return fields


def remap_oisst(
    source: np.ndarray,
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    latitudes = np.asarray(latitudes, dtype=np.float32)
    longitudes = np.mod(np.asarray(longitudes, dtype=np.float32), 360)
    if source.shape != (len(latitudes), len(longitudes)):
        raise ValueError("OISST coordinates do not match the SST array")
    if np.any(np.diff(latitudes) <= 0):
        order = np.argsort(latitudes)
        latitudes = latitudes[order]
        source = source[order]
    longitude_order = np.argsort(longitudes)
    longitudes = longitudes[longitude_order]
    source = source[:, longitude_order]

    periodic_longitudes = np.concatenate((longitudes[-1:] - 360, longitudes, longitudes[:1] + 360))
    periodic_source = np.concatenate((source[:, -1:], source, source[:, :1]), axis=1)
    interpolator = RegularGridInterpolator(
        (latitudes, periodic_longitudes),
        periodic_source,
        bounds_error=False,
        fill_value=None,
    )
    target_latitudes = np.arange(90.0, -90.0, -0.25, dtype=np.float32)
    target_longitudes = np.arange(0.0, 360.0, 0.25, dtype=np.float32)
    result = np.empty(GRID_SHAPE, dtype=np.float32)
    for index, latitude in enumerate(target_latitudes):
        points = np.column_stack(
            (
                np.full(len(target_longitudes), np.clip(latitude, latitudes[0], latitudes[-1])),
                target_longitudes,
            )
        )
        result[index] = interpolator(points)
    return result


def load_oisst(path: str | Path) -> np.ndarray:
    with netCDF4.Dataset(path) as dataset:
        variable = dataset.variables["sst"]
        source = np.ma.filled(variable[0, 0], np.nan).astype(np.float32)
        units = str(getattr(variable, "units", "")).lower()
        if "celsius" not in units and "degc" not in units:
            raise ValueError(f"Unsupported OISST units: {getattr(variable, 'units', None)}")
        source += np.float32(273.15)
        latitudes = np.asarray(dataset.variables["lat"][:], dtype=np.float32)
        longitudes = np.asarray(dataset.variables["lon"][:], dtype=np.float32)
    return remap_oisst(source, latitudes, longitudes)


def _interpolate_level(
    levels: dict[int, np.ndarray],
    target: int,
    *,
    logarithmic_values: bool,
) -> np.ndarray:
    available = np.asarray(sorted(levels))
    upper_index = int(np.searchsorted(available, target))
    if upper_index == 0 or upper_index == len(available):
        raise ValueError(f"Cannot interpolate {target} hPa from levels {available.tolist()}")
    lower = int(available[upper_index - 1])
    upper = int(available[upper_index])
    weight = math.log(target / lower) / math.log(upper / lower)
    lower_values = levels[lower]
    upper_values = levels[upper]
    if logarithmic_values:
        epsilon = np.float32(1e-10)
        lower_values = np.log(np.maximum(lower_values, epsilon))
        upper_values = np.log(np.maximum(upper_values, epsilon))
    result = lower_values + np.float32(weight) * (upper_values - lower_values)
    return np.exp(result) if logarithmic_values else result


def interpolate_missing_levels(
    pressure: dict[str, dict[int, np.ndarray]],
) -> dict[str, dict[int, np.ndarray]]:
    complete = {}
    for variable in PRESSURE_PARAMS:
        levels = dict(pressure[variable])
        for target in MODEL_LEVELS:
            if target not in levels:
                levels[target] = _interpolate_level(
                    levels,
                    target,
                    logarithmic_values=variable == "133_q",
                )
        complete[variable] = levels
    return complete


def _normalize(values: np.ndarray, norm: dict[str, object], level: int | None = None):
    if level is None:
        return (values - float(norm["mean"])) / float(norm["std"])
    norm_levels = _norms()["levels"]
    index = norm_levels.index(level)
    return (values - float(norm["mean"][index])) / float(norm["std"][index])


def _expand_levels(normalized: dict[int, np.ndarray]) -> list[np.ndarray]:
    # Deep stores normalized input levels as half before interpolating them.
    normalized = {level: values.astype(np.float16) for level, values in normalized.items()}
    channels = []
    for level in ENCODER_LEVELS:
        if level in normalized:
            channels.append(normalized[level])
            continue
        upper_index = int(np.searchsorted(MODEL_LEVELS, level))
        lower = MODEL_LEVELS[upper_index - 1]
        upper = MODEL_LEVELS[upper_index]
        weight = (level - lower) / (upper - lower)
        # PyTorch half multiplication uses float32 scalars, then rounds each product.
        left = np.multiply(normalized[lower], 1 - weight, dtype=np.float32).astype(np.float16)
        right = np.multiply(normalized[upper], weight, dtype=np.float32).astype(np.float16)
        channels.append(left + right)
    return channels


def build_sst(fields: OpenIFSFields, oisst: np.ndarray | None) -> np.ndarray:
    shape = fields.validate(expected_shape=None)
    if oisst is not None and oisst.shape != shape:
        raise ValueError(f"OISST grid is {oisst.shape}, expected {shape}")
    skin_temperature = fields.surface["skt"]
    candidate = (
        skin_temperature if oisst is None else np.where(np.isfinite(oisst), oisst, skin_temperature)
    )
    ocean = fields.surface["lsm"] < 0.5
    return np.where(ocean, candidate, np.nan).astype(np.float32)


def build_background(
    fields: OpenIFSFields,
    *,
    oisst: np.ndarray | None,
    clamp: float = 13,
) -> np.ndarray:
    shape = fields.validate(expected_shape=None)
    if clamp <= 0:
        raise ValueError("clamp must be positive")
    norms = _norms()
    complete_pressure = interpolate_missing_levels(fields.pressure)
    channels = []
    for variable, levels in complete_pressure.items():
        normalized = {
            level: _normalize(values, norms[variable], level)
            for level, values in levels.items()
            if level in MODEL_LEVELS
        }
        channels.extend(_expand_levels(normalized))
    for variable in (*CORE_SURFACE_VARS, *EXTRA_SURFACE_VARS):
        channels.append(_normalize(fields.surface[variable], norms[variable]))
    sst = build_sst(fields, oisst)
    channels.append(np.nan_to_num(_normalize(sst, norms["034_sstk"]), nan=0.0))
    channels.extend(np.zeros(shape, dtype=np.float32) for _ in range(4))

    background = np.stack(channels, axis=-1)
    if background.shape != (*shape, BACKGROUND_CHANNELS):
        raise ValueError(
            f"Prepared background is {background.shape}, expected {(*shape, BACKGROUND_CHANNELS)}"
        )
    background = np.clip(background, -clamp, clamp).astype(np.float16)
    if not np.isfinite(background).all():
        raise ValueError("Prepared background contains non-finite values")
    return background


def background_path(root: str | Path, cycle: datetime, member: int) -> Path:
    cycle = normalize_cycle(cycle)
    return Path(root) / f"{cycle:%Y%m}" / f"{cycle:%Y%m%d%H}-member{member:02d}.npz"


def write_background(
    path: str | Path,
    values: np.ndarray,
    *,
    cycle: datetime,
    member: int,
    sst_date: date | None = None,
) -> Path:
    cycle = normalize_cycle(cycle)
    validate_member(member)
    if values.shape != (*GRID_SHAPE, BACKGROUND_CHANNELS):
        raise ValueError(
            f"Background shape is {values.shape}, expected {(*GRID_SHAPE, BACKGROUND_CHANNELS)}"
        )
    if values.dtype != np.float16 or not np.isfinite(values).all():
        raise ValueError("Background values must be finite float16")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez(
                output,
                x=values,
                cycle=np.int64(cycle.timestamp()),
                member=np.int16(member),
                sst_date="" if sst_date is None else sst_date.isoformat(),
            )
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def prepare_background(
    grib_path: str | Path,
    oisst_path: str | Path,
    output_root: str | Path,
    *,
    cycle: datetime,
    member: int = 1,
) -> Path:
    fields = load_open_ifs(grib_path)
    oisst = load_oisst(oisst_path)
    values = build_background(fields, oisst=oisst)
    match = re.search(r"\.(\d{8})(?:_preliminary)?\.nc$", Path(oisst_path).name)
    if not match:
        raise ValueError(f"Cannot determine OISST date from {oisst_path}")
    sst_date = datetime.strptime(match.group(1), "%Y%m%d").date()
    return write_background(
        background_path(output_root, cycle, member),
        values,
        cycle=cycle,
        member=member,
        sst_date=sst_date,
    )


def load_background(path: str | Path) -> PreparedBackground:
    with np.load(path) as stored:
        values = np.asarray(stored["x"])
        cycle = datetime.fromtimestamp(int(stored["cycle"]), tz=timezone.utc)
        member = int(stored["member"])
        stored_sst_date = str(stored["sst_date"]) if "sst_date" in stored.files else ""
    normalize_cycle(cycle)
    validate_member(member)
    if values.shape != (*GRID_SHAPE, BACKGROUND_CHANNELS):
        raise ValueError(
            f"Background shape is {values.shape}, expected {(*GRID_SHAPE, BACKGROUND_CHANNELS)}"
        )
    if values.dtype != np.float16 or not np.isfinite(values).all():
        raise ValueError("Background values must be finite float16")
    sst_date = date.fromisoformat(stored_sst_date) if stored_sst_date else None
    return PreparedBackground(values, cycle, member, sst_date)


def background_info(path: str | Path) -> BackgroundInfo:
    """Read lightweight background metadata without inflating its forecast array."""

    with np.load(path) as stored:
        cycle = datetime.fromtimestamp(int(stored["cycle"]), tz=timezone.utc)
        member = int(stored["member"])
        stored_sst_date = str(stored["sst_date"]) if "sst_date" in stored.files else ""
    normalize_cycle(cycle)
    validate_member(member)
    sst_date = date.fromisoformat(stored_sst_date) if stored_sst_date else None
    return BackgroundInfo(cycle, member, sst_date)
