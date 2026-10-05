from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from dataclasses import fields as dataclass_fields
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import netCDF4
import numpy as np
import torch

from ._core.utils import stretched_tau


@dataclass(frozen=True)
class Field:
    internal_name: str
    name: str
    units: str
    standard_name: str | None = None
    height: float | None = None


PRESSURE_FIELDS = (
    Field("129_z", "geopotential", "m2 s-2", "geopotential"),
    Field("130_t", "air_temperature", "K", "air_temperature"),
    Field("131_u", "eastward_wind", "m s-1", "eastward_wind"),
    Field("132_v", "northward_wind", "m s-1", "northward_wind"),
    Field("133_q", "specific_humidity", "kg kg-1", "specific_humidity"),
)
SURFACE_FIELDS = (
    Field("165_10u", "eastward_wind_10m", "m s-1", "eastward_wind", 10),
    Field("166_10v", "northward_wind_10m", "m s-1", "northward_wind", 10),
    Field("167_2t", "air_temperature_2m", "K", "air_temperature", 2),
    Field("151_msl", "air_pressure_at_mean_sea_level", "Pa", "air_pressure_at_mean_sea_level"),
    Field("168_2d", "dew_point_temperature_2m", "K", "dew_point_temperature", 2),
    Field("246_100u", "eastward_wind_100m", "m s-1", "eastward_wind", 100),
    Field("247_100v", "northward_wind_100m", "m s-1", "northward_wind", 100),
    Field(
        "15_msnswrf",
        "mean_surface_net_short_wave_radiation_flux",
        "W m-2",
        "surface_net_downward_shortwave_flux",
    ),
    Field("45_tcc", "total_cloud_cover", "1", "cloud_area_fraction"),
    Field("ntp-3h", "total_precipitation_3h", "mm"),
    Field("mx2t-3h", "maximum_air_temperature_2m_3h", "K", height=2),
    Field("mn2t-3h", "minimum_air_temperature_2m_3h", "K", height=2),
)
DERIVED_SURFACE_FIELDS = (
    Field("wind_speed_10m", "wind_speed_10m", "m s-1", "wind_speed", 10),
    Field("wind_speed_100m", "wind_speed_100m", "m s-1", "wind_speed", 100),
)


@dataclass(frozen=True)
class ForecastFile:
    path: Path
    valid_time: datetime
    size_bytes: int


@dataclass(frozen=True)
class PhysicalProducts:
    """Physical forecast products for one ensemble trajectory and base lead."""

    surface: np.ndarray | None = None
    pressure: np.ndarray | None = None


@dataclass(frozen=True)
class DecodedProducts:
    """Minimal decoder tensors sent from the GPU to the ensemble worker."""

    surface: torch.Tensor
    pressure: torch.Tensor | None = None


@dataclass(frozen=True)
class ProductTransform:
    """CPU normalization values shared by every ensemble trajectory."""

    global_means: np.ndarray
    global_standard_deviations: np.ndarray

    @classmethod
    def from_model(cls, model) -> ProductTransform:
        global_mesh = model.decoders[0].mesh
        return cls(
            global_means=global_mesh.state_norm_means.detach().float().cpu().numpy().copy(),
            global_standard_deviations=(
                global_mesh.state_norm_stds.detach().float().cpu().numpy().copy()
            ),
        )


@dataclass(frozen=True)
class OutputGrid:
    levels: tuple[int, ...]
    lats: np.ndarray
    lons: np.ndarray

    @classmethod
    def from_mesh(cls, mesh) -> OutputGrid:
        return cls(
            tuple(int(level) for level in mesh.levels),
            np.asarray(mesh.lats, dtype=np.float32).copy(),
            np.asarray(mesh.lons, dtype=np.float32).copy(),
        )


def _field_attributes(field: Field) -> dict[str, object]:
    attributes = {"units": field.units, "internal_name": field.internal_name}
    if field.standard_name:
        attributes["standard_name"] = field.standard_name
    if field.height is not None:
        attributes.update(height=field.height, height_units="m")
    return attributes


def _create_field(dataset, field: Field, dimensions: tuple[str, ...], *, compression: bool):
    chunks = tuple(
        min(
            len(dataset.dimensions[name]),
            360 if name == "longitude" else 180,
        )
        if name in {"latitude", "longitude"}
        else 1
        for name in dimensions
    )
    variable = dataset.createVariable(
        field.name,
        "f4",
        dimensions,
        compression="zlib" if compression else None,
        complevel=1,
        shuffle=compression,
        fill_value=np.float32(np.nan),
        chunksizes=chunks,
    )
    variable.setncatts(_field_attributes(field))
    return variable


def write_manifest(path: str | Path, manifest: dict[str, object]) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w") as output:
            json.dump(manifest, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return destination


def select_global_products(
    decoded: torch.Tensor,
    mesh,
    *,
    surface_only: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Select the compact tensors transferred by the operational runner."""

    if decoded.ndim != 4 or decoded.shape[0] != 1:
        raise ValueError(f"Decoded forecast has unsupported shape {tuple(decoded.shape)}")
    surface_count = len(SURFACE_FIELDS)
    surface_start = 0 if surface_only else mesh.n_pr
    surface_stop = surface_start + surface_count
    if decoded.shape[-1] < surface_stop + len(mesh.sfc_quantile_vars):
        raise ValueError("Decoded forecast does not contain all surface products")

    surface = decoded[..., surface_start:surface_stop].clone()
    precipitation_index = next(
        index for index, field in enumerate(SURFACE_FIELDS) if field.internal_name == "ntp-3h"
    )
    quantile_start = decoded.shape[-1] - len(mesh.sfc_quantile_vars)
    quantile_start += precipitation_index * 30
    precipitation_quantiles = decoded[..., quantile_start : quantile_start + 30].float()
    precipitation_quantiles = torch.exp(precipitation_quantiles) * 4 - 1
    tau = _precipitation_tau(decoded.device)
    surface[..., precipitation_index] = torch.trapezoid(precipitation_quantiles, tau, dim=-1)
    pressure = None if surface_only else decoded[..., : mesh.n_pr]
    return surface, pressure


@lru_cache
def _precipitation_tau(device: torch.device) -> torch.Tensor:
    return stretched_tau(30, k=1.0).to(device)


def aggregate_decoded_products(
    decoded: DecodedProducts,
    transform: ProductTransform,
) -> PhysicalProducts:
    """Convert copied decoder tensors to the operational aggregation representation."""

    surface = decoded.surface[0].float().numpy()
    expected_surface = (*surface.shape[:2], len(SURFACE_FIELDS))
    if surface.shape != expected_surface:
        raise ValueError(f"Surface product is {surface.shape}, expected {expected_surface}")

    surface_norm_means = transform.global_means[-len(SURFACE_FIELDS) :]
    surface_norm_stds = transform.global_standard_deviations[-len(SURFACE_FIELDS) :]
    u10 = surface[..., 0] * surface_norm_stds[0] + surface_norm_means[0]
    v10 = surface[..., 1] * surface_norm_stds[1] + surface_norm_means[1]
    u100 = surface[..., 5] * surface_norm_stds[5] + surface_norm_means[5]
    v100 = surface[..., 6] * surface_norm_stds[6] + surface_norm_means[6]
    surface = np.concatenate(
        (
            surface,
            np.sqrt(np.square(u10) + np.square(v10))[..., None],
            np.sqrt(np.square(u100) + np.square(v100))[..., None],
        ),
        axis=-1,
    )
    surface = surface.transpose(2, 0, 1).astype(np.float32, copy=False)

    pressure = None
    if decoded.pressure is not None:
        raw_pressure = decoded.pressure[0].float().numpy()
        level_count = len(transform.global_means) - len(SURFACE_FIELDS)
        if raw_pressure.shape != (*raw_pressure.shape[:2], level_count):
            raise ValueError(
                f"Pressure product is {raw_pressure.shape}; expected {level_count} channels"
            )
        pressure = raw_pressure.reshape(
            *raw_pressure.shape[:2], len(PRESSURE_FIELDS), -1
        ).transpose(2, 3, 0, 1)
        pressure = pressure.astype(np.float32, copy=False)

    products = PhysicalProducts(surface, pressure)
    for field in dataclass_fields(products):
        value = getattr(products, field.name)
        if value is not None and not np.isfinite(value).all():
            raise ValueError(f"{field.name.capitalize()} forecast contains non-finite values")
    return products


def physicalize_statistics(
    products: PhysicalProducts,
    transform: ProductTransform,
    statistic: str,
) -> PhysicalProducts:
    """Apply linear field normalization after ensemble aggregation."""

    if statistic not in {"mean", "standard_deviation"}:
        raise ValueError(f"Unknown statistic: {statistic}")
    is_mean = statistic == "mean"

    surface = None
    if products.surface is not None:
        surface = np.asarray(products.surface, dtype=np.float32).copy(order="K")
        count = len(SURFACE_FIELDS)
        scale = transform.global_standard_deviations[-count:, None, None]
        if is_mean:
            surface[:count] *= scale
            surface[:count] += transform.global_means[-count:, None, None]
        else:
            surface[:count] *= np.abs(scale)
        precipitation_index = next(
            index for index, field in enumerate(SURFACE_FIELDS) if field.internal_name == "ntp-3h"
        )
        surface[precipitation_index] = products.surface[precipitation_index]

    pressure = None
    if products.pressure is not None:
        pressure = np.asarray(products.pressure, dtype=np.float32).copy(order="K")
        pressure_count = len(transform.global_means) - len(SURFACE_FIELDS)
        scale = transform.global_standard_deviations[:pressure_count].reshape(
            len(PRESSURE_FIELDS), -1, 1, 1
        )
        if is_mean:
            pressure *= scale
            pressure += transform.global_means[:pressure_count].reshape(
                len(PRESSURE_FIELDS), -1, 1, 1
            )
        else:
            pressure *= np.abs(scale)

    return PhysicalProducts(surface=surface, pressure=pressure)


def _write_coordinate_fields(dataset, mesh, valid_time: datetime) -> None:
    dataset.createDimension("time", 1)
    dataset.createDimension("pressure", len(mesh.levels))
    dataset.createDimension("latitude", len(mesh.lats))
    dataset.createDimension("longitude", len(mesh.lons))

    time = dataset.createVariable("time", "i8", ("time",))
    time.standard_name = "time"
    time.units = "seconds since 1970-01-01 00:00:00 UTC"
    time.calendar = "standard"
    time[:] = [int(valid_time.timestamp())]

    pressure = dataset.createVariable("pressure", "i4", ("pressure",))
    pressure.standard_name = "air_pressure"
    pressure.units = "hPa"
    pressure.positive = "down"
    pressure[:] = mesh.levels

    latitude = dataset.createVariable("latitude", "f4", ("latitude",))
    latitude.standard_name = "latitude"
    latitude.units = "degrees_north"
    latitude[:] = mesh.lats

    longitude = dataset.createVariable("longitude", "f4", ("longitude",))
    longitude.standard_name = "longitude"
    longitude.units = "degrees_east"
    longitude[:] = mesh.lons


def _write_physical_products(
    dataset,
    products: tuple[PhysicalProducts, ...],
    labels: tuple[str, ...],
    mesh,
    *,
    compression: bool,
) -> None:
    multiple = len(products) > 1
    prefix = ("ensemble_statistic",) if multiple else ()
    if multiple:
        dataset.createDimension("ensemble_statistic", len(labels))
        statistic = dataset.createVariable("ensemble_statistic", str, ("ensemble_statistic",))
        statistic[:] = np.asarray(labels)

    surface_values = [product.surface for product in products]
    if any(value is not None for value in surface_values):
        if not all(value is not None for value in surface_values):
            raise ValueError("Surface product availability differs between statistics")
        for index, field in enumerate((*SURFACE_FIELDS, *DERIVED_SURFACE_FIELDS)):
            variable = _create_field(
                dataset, field, (*prefix, "time", "latitude", "longitude"), compression=compression
            )
            data = (
                np.stack([value[index] for value in surface_values])
                if multiple
                else surface_values[0][index]
            )
            variable[(slice(None),) * len(prefix) + (0,)] = data

    pressure_values = [product.pressure for product in products]
    if any(value is not None for value in pressure_values):
        if not all(value is not None for value in pressure_values):
            raise ValueError("Pressure product availability differs between statistics")
        for index, field in enumerate(PRESSURE_FIELDS):
            variable = _create_field(
                dataset,
                field,
                (*prefix, "time", "pressure", "latitude", "longitude"),
                compression=compression,
            )
            data = (
                np.stack([value[index] for value in pressure_values])
                if multiple
                else pressure_values[0][index]
            )
            variable[(slice(None),) * len(prefix) + (0,)] = data


def write_product_file(
    path: str | Path,
    products: tuple[PhysicalProducts, ...],
    labels: tuple[str, ...],
    mesh,
    *,
    background_time: datetime,
    aida_valid_time: datetime,
    lead_hours: int,
    input_member: int | None = None,
    pressure_supermember: int | None = None,
    trajectory_count: int,
    compression: bool = True,
) -> ForecastFile:
    if not products or len(products) != len(labels):
        raise ValueError("Products and labels must have the same non-zero length")
    if background_time.tzinfo is None or aida_valid_time.tzinfo is None:
        raise ValueError("Forecast times must include a UTC offset")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(descriptor)
    valid_time = aida_valid_time + timedelta(hours=lead_hours)
    analysis_time = aida_valid_time.astimezone(timezone.utc)
    hour_offset = analysis_time.hour % 6
    attributes = {
        "Conventions": "CF-1.10",
        "title": "WeatherMesh-5c ensemble forecast",
        "institution": "WindBorne Systems",
        "source": "WeatherMesh-5c with AIDA",
        "background_time": background_time.astimezone(timezone.utc).isoformat(),
        "aida_valid_time": analysis_time.isoformat(),
        "lead_hours": lead_hours,
        "forecast_reference_time": (analysis_time - timedelta(hours=hour_offset)).isoformat(),
        "forecast_hour": lead_hours + hour_offset,
        "ensemble_trajectory_count": trajectory_count,
    }
    if input_member is not None:
        attributes["open_ifs_member"] = input_member
    if pressure_supermember is not None:
        attributes["pressure_supermember"] = pressure_supermember
    try:
        if destination.suffix == ".zarr":
            from .zarr_output import write_zarr_products

            write_zarr_products(
                temporary_name,
                products,
                labels,
                mesh,
                valid_time,
                attributes,
                compression=compression,
            )
        elif destination.suffix == ".nc":
            with netCDF4.Dataset(temporary_name, "w", format="NETCDF4") as dataset:
                dataset.setncatts(attributes)
                _write_coordinate_fields(dataset, mesh, valid_time)
                _write_physical_products(dataset, products, labels, mesh, compression=compression)
        else:
            raise ValueError(f"Unsupported forecast file extension: {destination.suffix}")
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return ForecastFile(destination, valid_time, destination.stat().st_size)
