from collections.abc import Callable, Iterable
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from ...assets import load_array
from ..utils import get_date

ADDITIONAL_VARIABLES = 4
_GRID_SHAPE = (720, 1440)
_LATITUDES = np.radians(np.arange(90, -90.01, -0.25)[:-1])[:, np.newaxis]
_LONGITUDES = np.arange(0, 359.99, 0.25)[np.newaxis, :]
_SIN_LATITUDES = np.sin(_LATITUDES)
_COS_LATITUDES = np.cos(_LATITUDES)
assert (_LATITUDES.shape[0], _LONGITUDES.shape[1]) == _GRID_SHAPE


def _solar_altitude_values(day: int, hour: int) -> np.ndarray:
    calendar_day = day + 1
    declination = np.radians(23.45 * np.sin((2 * np.pi / 365.0) * (calendar_day - 81)))
    phase = 2 * np.pi / 364.0 * (calendar_day - 81)
    equation = 9.87 * np.sin(2 * phase) - 7.53 * np.cos(phase) - 1.5 * np.sin(phase)
    solar_time = (hour * 60 + 4 * _LONGITUDES + equation) / 60
    hour_angle = np.radians(15.0 * (solar_time - 12.0))
    elevation = _COS_LATITUDES * np.cos(declination) * np.cos(hour_angle) + _SIN_LATITUDES * np.sin(
        declination
    )
    return np.degrees(np.arcsin(np.clip(elevation, -1, 1)))


def _solar_altitude(day: int, hour: int) -> np.ndarray:
    return _solar_altitude_values(day, hour).astype(np.float32)


def _direct_radiation(day: int) -> np.ndarray:
    calendar_day = day + 1
    altitude = _solar_altitude_values(day, 0)
    flux = 1160 + 75 * np.sin(2 * np.pi / 365 * (calendar_day - 275))
    optical_depth = 0.174 + 0.035 * np.sin(2 * np.pi / 365 * (calendar_day - 100))
    with np.errstate(over="ignore", under="ignore"):
        radiation = flux * np.exp(-optical_depth / np.sin(np.radians(altitude)))
    return np.where(altitude < 0, 0, radiation).astype(np.float32)


@lru_cache(maxsize=64)
def _solar_field(day: int, hour: int) -> torch.Tensor:
    return torch.HalfTensor(_solar_altitude(day, hour) / (180 / np.pi))


@lru_cache(maxsize=32)
def _radiation_field(day: int, hour: int) -> torch.Tensor:
    if hour != 0:
        raise ValueError(f"Radiation source hour must be zero, got {hour}")
    return torch.HalfTensor((_direct_radiation(day) - 300) / 400)


def _daily_field(
    loader: Callable[[int, int], torch.Tensor],
    days: int,
    day: int,
    hour: int,
) -> torch.Tensor:
    def load(field_day: int) -> torch.Tensor:
        return loader(field_day % days, hour)

    if day % 2:
        return load(day - 1) * 0.5 + load(day + 1) * 0.5
    return load(day)


def get_additional_vars(timestamps: torch.Tensor) -> torch.Tensor:
    device = timestamps.device
    dates = [get_date(timestamp.item()) for timestamp in timestamps]
    year_starts = [date.replace(month=1, day=1) for date in dates]
    days = [
        int((date - year_start).total_seconds() / 86_400)
        for date, year_start in zip(dates, year_starts, strict=True)
    ]

    def stack(fields: Iterable[torch.Tensor]) -> torch.Tensor:
        return torch.cat(
            [field[:720, :, np.newaxis].unsqueeze(0).to(device) for field in fields],
            dim=0,
        )

    # Preserve the feature contract used to train WeatherMesh-5c. The original
    # pipeline selected the radiation day from the UTC hour and read hour zero.
    radiation = stack(_daily_field(_radiation_field, 25, date.hour, 0) for date in dates)
    solar_angle = stack(
        _daily_field(_solar_field, 366, day, date.hour)
        for day, date in zip(days, dates, strict=True)
    )
    hours = torch.tensor([date.hour / 24 for date in dates], device=device)
    time_of_day = torch.zeros_like(radiation) + hours[:, None, None, None]
    result = torch.cat(
        (radiation, time_of_day, torch.sin(solar_angle), torch.cos(solar_angle)),
        dim=-1,
    ).half()
    if result.shape[-1] != ADDITIONAL_VARIABLES:
        raise ValueError(f"Expected {ADDITIONAL_VARIABLES} derived inputs, got {result.shape[-1]}")
    return result


def get_constant_vars(mesh, root: Path, *, resolve=None):
    path = resolve if resolve is not None else lambda name: root / name
    coordinates = torch.FloatTensor(mesh.xpos)
    scaled = coordinates * torch.Tensor([np.pi / 2, np.pi])
    variables = ["sinlat", "sinlon", "coslat", "coslon"]
    values = [torch.sin(scaled), torch.cos(scaled)]

    land = _downsample(load_array(path("land_mask.npy")), mesh.xpos.shape)
    variables.append("land_mask")
    values.append(torch.BoolTensor(np.round(land)).unsqueeze(-1))

    soil = _downsample(load_array(path("soil_type.npy")), mesh.xpos.shape, reduce=np.min)
    soil = torch.BoolTensor(_one_hot(soil))
    variables.extend(f"soil_type{index}" for index in range(soil.shape[-1]))
    values.append(soil)

    elevation = _downsample(load_array(path("topography.npy")), mesh.xpos.shape, reduce=np.mean)
    elevation = torch.FloatTensor(elevation / np.max(elevation)).unsqueeze(-1)
    variables.append("elevation")
    values.append(elevation)

    data = torch.cat(values, dim=-1)
    if data.shape[-1] != len(variables):
        raise ValueError(f"Constant data has {data.shape[-1]} channels, expected {len(variables)}")
    return data, variables


def _downsample(values, shape, reduce=np.mean):
    factor = (values.shape[0] - 1) // shape[0]
    if values.shape[1] // shape[1] != factor:
        raise ValueError("Latitude and longitude need the same downsampling factor")
    reshaped = values[:-1].reshape(shape[0], factor, shape[1], factor)
    result = reduce(reshaped, axis=(1, 3))
    if result.shape != shape[:2]:
        raise ValueError(f"Downsampled shape is {result.shape}, expected {shape[:2]}")
    return result


def _one_hot(values):
    values = values.astype(int)
    return np.eye(np.max(values) + 1)[values]
