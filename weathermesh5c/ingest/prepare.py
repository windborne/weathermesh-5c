from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timedelta, timezone
from importlib.resources import files
from pathlib import Path

import numpy as np
import torch

from .storage import processed_path

SECONDS_PER_YEAR = 31_556_925
NORM_MAPPING = {
    "gp": "129_z",
    "temp": "130_t",
    "rh": "rhlol",
    "ucomp": "131_u",
    "vcomp": "132_v",
    "sh": "133_q",
    "mslp": "151_msl",
    "2t": "167_2t",
    "2d": "168_2d",
    "10u": "165_10u",
    "10v": "166_10v",
    "elev": "elev",
    "is_ship": "is_ship",
}
MESH_SOURCES = {
    "1bmhs": "1bmhs",
    "1bamua": "1bamua",
    "atms": "atms",
    "gpsro": "gpsro",
    "iasi_pca": "mtiasi",
    "radiosonde": "adpupa",
    "windborne": "windborne",
    "satwnd": "satwnd",
    "sfc": "adpsfc",
}


def _norms():
    path = files("weathermesh5c._core.norm") / "norm.json"
    with path.open() as stream:
        return json.load(stream)


def _solar_altitude(latitude, longitude, when):
    day = (when.astype("datetime64[D]") - when.astype("datetime64[Y]")).astype(int) + 1
    hour = (when.astype("datetime64[h]") - when.astype("datetime64[D]")).astype(int)
    minute = (when.astype("datetime64[m]") - when.astype("datetime64[h]")).astype(int)
    declination = np.deg2rad(23.45 * np.sin((2 * np.pi / 365) * (day - 81)))
    latitude_rad = np.deg2rad(latitude)
    b = 2 * np.pi / 364 * (day - 81)
    equation = 9.87 * np.sin(2 * b) - 7.53 * np.cos(b) - 1.5 * np.sin(b)
    solar_time = (hour * 60 + minute + 4 * longitude + equation) / 60
    hour_angle = np.deg2rad(15 * (solar_time - 12))
    elevation = np.cos(latitude_rad) * np.cos(declination) * np.cos(hour_angle) + np.sin(
        latitude_rad
    ) * np.sin(declination)
    return np.rad2deg(np.arcsin(np.clip(elevation, -1, 1)))


def _normalize(data: np.ndarray, mesh) -> tuple[np.ndarray, list[str]]:
    data = data.astype(np.float32, copy=True)
    norms = _norms()
    disk_columns = list(mesh.disk_varlist)
    has_pressure = "pres_pa" in disk_columns
    pressure = data[:, disk_columns.index("pres_pa")] / 100 if has_pressure else None
    norm_levels = np.asarray(norms["levels"])
    means = []
    standard_deviations = []
    for variable in mesh.vars:
        norm = norms[NORM_MAPPING[variable]]
        for field, target in (("mean", means), ("std", standard_deviations)):
            values = np.asarray(norm[field], dtype=np.float64)
            if values.ndim:
                valid = ~np.isnan(values)
                values = np.interp(norm_levels, norm_levels[valid], values[valid])
            target.append(values)

    output_columns = disk_columns.copy()
    for index, variable in enumerate(disk_columns):
        if not variable.startswith("unnorm_"):
            continue
        if "viz" in variable or "pr_mm" in variable:
            continue
        name = variable[7:].split("_")[0]
        variable_index = mesh.vars.index(name)
        if has_pressure:
            scale = np.interp(pressure, norm_levels, standard_deviations[variable_index])
            offset = np.interp(pressure, norm_levels, means[variable_index])
        else:
            scale = float(standard_deviations[variable_index])
            offset = float(means[variable_index])
        if variable.endswith("_c"):
            offset -= 273.15
        elif variable.endswith("_hpa"):
            scale /= 100
            offset /= 100
        data[:, index] = (data[:, index] - offset) / scale
        output_columns[index] = name
    return data, output_columns


def _reject_outliers(data: np.ndarray, columns: list[str], mesh, threshold: float = 10):
    for variable in mesh.vars:
        if variable in columns:
            index = columns.index(variable)
            data[np.abs(data[:, index]) > threshold, index] = np.nan
    pressure_name = "pres_pa" if "pres_pa" in columns else "pres_hpa"
    if pressure_name in columns:
        pressure = data[:, columns.index(pressure_name)]
        maximum = 105_000 if pressure_name == "pres_pa" else 1_050
        data = data[(pressure != 0) & (pressure < maximum)]
    return data


def _transform(data: np.ndarray, file_time: float, columns: list[str], mesh):
    output = np.full((len(data), len(mesh.full_varlist)), np.nan, dtype=np.float32)
    reltime = data[:, columns.index("reltime_hours")]
    absolute_time = file_time + (reltime * 3600).astype(np.int64)
    timestamps = absolute_time.astype("datetime64[s]")
    for index, variable in enumerate(mesh.full_varlist):
        if variable == "sin(time_of_year)":
            output[:, index] = np.sin(file_time / SECONDS_PER_YEAR * 2 * np.pi)
        elif variable == "cos(time_of_year)":
            output[:, index] = np.cos(file_time / SECONDS_PER_YEAR * 2 * np.pi)
        elif variable == "sin(time_of_day)":
            output[:, index] = np.sin(absolute_time / 3600 / 24 * 2 * np.pi)
        elif variable == "cos(time_of_day)":
            output[:, index] = np.cos(absolute_time / 3600 / 24 * 2 * np.pi)
        elif variable == "solar_elevation":
            output[:, index] = (
                _solar_altitude(
                    data[:, columns.index("lat_deg")],
                    data[:, columns.index("lon_deg")],
                    timestamps,
                )
                / 90
            )
        elif variable == "reltime_hours":
            output[:, index] = reltime
        elif variable in {"lat_deg", "lon_deg"}:
            if variable in columns:
                output[:, index] = data[:, columns.index(variable)]
            else:
                stem = variable.replace("_deg", "")
                output[:, index] = (
                    data[:, columns.index(f"{stem}int")] + data[:, columns.index(f"{stem}dec")]
                )
        elif variable == "pres_hpa":
            output[:, index] = data[:, columns.index("pres_pa")] / 100
        elif variable.endswith("_present?"):
            output[:, index] = ~np.isnan(data[:, columns.index(variable.removesuffix("_present?"))])
        elif variable in mesh.vars:
            output[:, index] = np.nan_to_num(data[:, columns.index(variable)])
            if variable.startswith("pca_"):
                output[:, index] = np.clip(output[:, index], -10, 10)
        elif variable.startswith("onehot_"):
            name, value = variable.removeprefix("onehot_").rsplit("_", 1)
            output[:, index] = data[:, columns.index(name)] == int(value)
        elif variable.startswith(("sin(", "cos(")):
            operation = np.sin if variable.startswith("sin(") else np.cos
            name = variable[variable.index("(") + 1 : -1]
            factor = np.pi / 180 if name.endswith("_deg") else 1
            source = (
                data[:, columns.index(name)]
                if name in columns
                else output[:, mesh.full_varlist.index(name)]
            )
            output[:, index] = operation(source * factor)
        else:
            raise ValueError(f"Unknown {mesh.string_id} model input: {variable}")
    if np.isnan(output).any():
        raise ValueError(f"NaN remained in prepared {mesh.string_id} observations")
    return output


def _transform_gpsro(data: np.ndarray, mesh):
    # Deep reconstructs coordinates in the stored dtype before casting the output.
    disk_columns = list(mesh.disk_varlist)
    present = data[:, :, disk_columns.index("all_present")].astype(bool)
    outliers = (np.abs(data[:, :, disk_columns.index("bint")]) > 10) | (
        np.abs(data[:, :, disk_columns.index("rint")]) > 10
    )
    present &= ~outliers
    output = np.full((*data.shape[:2], len(mesh.full_varlist)), np.nan, dtype=np.float32)
    for index, variable in enumerate(mesh.full_varlist):
        if variable == "reltime_hours":
            output[:, :, index] = data[:, :, disk_columns.index(variable)]
        elif variable in {"lat_deg", "lon_deg"}:
            stem = variable.replace("_deg", "")
            output[:, :, index] = (
                data[:, :, disk_columns.index(f"{stem}int")]
                + data[:, :, disk_columns.index(f"{stem}dec")]
            )
        elif variable in mesh.vars:
            values = data[:, :, disk_columns.index(variable)]
            output[:, :, index] = np.where(present, values, 0)
        elif variable.endswith("_present?"):
            output[:, :, index] = present
        elif variable.startswith(("sin(", "cos(")):
            operation = np.sin if variable.startswith("sin(") else np.cos
            name = variable[variable.index("(") + 1 : -1]
            output[:, :, index] = operation(np.deg2rad(output[:, :, mesh.full_varlist.index(name)]))
        else:
            raise ValueError(f"Unknown GPSRO model input: {variable}")
    if np.isnan(output).any():
        raise ValueError("NaN remained in prepared GPSRO observations")
    return output


def _prepare_hour(data: np.ndarray, mesh, file_time: float):
    if mesh.string_id == "gpsro":
        return _transform_gpsro(data, mesh)
    if mesh.string_id in {"1bmhs", "1bamua", "atms"}:
        data = data.astype(np.float32, copy=True)
        satellite = mesh.disk_varlist.index("said")
        known = np.isin(data[:, satellite], mesh.sats)
        data = data[known]
        order = np.argsort(mesh.sats, kind="stable")
        indices = np.searchsorted(mesh.sats, data[:, satellite], sorter=order)
        data[:, satellite] = order[indices]
        data = _reject_outliers(data, list(mesh.disk_varlist), mesh)
        return _transform(data, file_time, list(mesh.disk_varlist), mesh)
    if mesh.string_id == "iasi_pca":
        return _transform(data.astype(np.float32), file_time, list(mesh.disk_varlist), mesh)
    data, columns = _normalize(data, mesh)
    data = _reject_outliers(data, columns, mesh)
    return _transform(data, file_time, columns, mesh)


def _load_mesh(
    mesh,
    root: str | Path,
    valid_time: datetime,
    *,
    max_observations: int | None = None,
) -> torch.Tensor:
    minimum, maximum = mesh.da_window_size

    def hours() -> Iterator[tuple[datetime, np.ndarray]]:
        for dh in range(math.floor(minimum) + 1, math.ceil(maximum) + 1):
            hour = valid_time + timedelta(hours=dh)
            path = processed_path(root, MESH_SOURCES[mesh.string_id], hour)
            if path.is_file():
                with np.load(path) as stored:
                    yield hour, stored["x"]

    return prepare_observations(mesh, hours(), valid_time, max_observations=max_observations)


def prepare_observations(
    mesh,
    hours: Iterable[tuple[datetime, np.ndarray]],
    valid_time: datetime,
    *,
    max_observations: int | None = None,
) -> torch.Tensor:
    """Prepare hourly observation arrays with the native QC, windows, and sampling.

    ``hours`` must use the source's disk columns and dtype, in file-hour order.
    GPSRO arrays retain their profile axis. Times are UTC. This is also the
    boundary for external data adapters; it does not decode or encode a latent.
    """
    minimum, maximum = mesh.da_window_size
    prepared = []
    for hour, data in hours:
        dh = (hour - valid_time).total_seconds() / 3600
        data = _prepare_hour(data, mesh, hour.timestamp())
        time_index = mesh.full_varlist.index("reltime_hours")
        data[..., time_index] += dh
        in_window = (data[..., time_index] >= minimum) & (data[..., time_index] <= maximum)
        if data.ndim == 3:
            data = data[in_window.all(axis=1)]
        else:
            data = data[in_window]
        prepared.append(data)
    if not prepared:
        return torch.empty((1, 0, 0), dtype=torch.float32)
    observations = np.concatenate(prepared)
    limit = min(mesh.encoder_subset, len(observations))
    if max_observations is not None:
        if max_observations < 1:
            raise ValueError("max_observations must be positive")
        limit = min(limit, max_observations)
    if limit < len(observations):
        indices = torch.randperm(len(observations))[: int(limit)]
        observations = observations[indices.numpy()]
    return torch.from_numpy(observations).unsqueeze(0)


def load_observations(
    model,
    root: str | Path,
    valid_time: datetime,
    *,
    device=None,
    max_observations: int | None = None,
):
    if valid_time.tzinfo is None:
        valid_time = valid_time.replace(tzinfo=timezone.utc)
    valid_time = valid_time.astimezone(timezone.utc)
    if valid_time.minute or valid_time.second or valid_time.microsecond:
        raise ValueError("Observation valid time must be on an exact UTC hour")
    observations = {}
    for encoder in model.obs_encoders:
        for mesh in encoder.meshes:
            tensor = _load_mesh(
                mesh,
                root,
                valid_time,
                max_observations=max_observations,
            )
            observations[mesh.mesh_id] = tensor.to(device=device) if device is not None else tensor
    return observations


def build_batch(
    model,
    background: torch.Tensor,
    valid_time: datetime,
    *,
    observations_root: str | Path,
    analysis_offset_hours: int = -3,
    max_observations: int | None = None,
    observations: Mapping[str, torch.Tensor] | None = None,
    observation_cycles: Mapping[int, Mapping[str, torch.Tensor]] | None = None,
):
    if observation_cycles is not None:
        if observations is not None:
            raise ValueError("Use observations or observation_cycles, not both")
        hours = sorted(observation_cycles)
        if hours != list(range(analysis_offset_hours + 3, 1, 3)) or not hours:
            raise ValueError(
                "Observation cycles must follow the background every three hours through zero"
            )
        batch = {}
        for hour in hours:
            part = build_batch(
                model,
                background,
                valid_time + timedelta(hours=hour),
                observations_root=observations_root,
                analysis_offset_hours=analysis_offset_hours - hour,
                observations=observation_cycles[hour],
            )
            if not part[0]["has_observations"]:
                raise ValueError(f"No observations for assimilation hour {hour}")
            if not batch:
                batch[analysis_offset_hours] = part[analysis_offset_hours - hour]
            batch[hour] = part[0]
        return batch
    mesh = model.anl_encoders[0].mesh
    expected = tuple(mesh.shape())
    if tuple(background.shape) == expected:
        background = background.unsqueeze(0)
    if tuple(background.shape[1:]) != expected:
        raise ValueError(
            f"Background shape must be {expected} or {(1, *expected)}, "
            f"got {tuple(background.shape)}"
        )
    device = background.device
    if observations is None:
        observations = load_observations(
            model,
            observations_root,
            valid_time,
            device=device,
            max_observations=max_observations,
        )
    elif any(tensor.device != device for tensor in observations.values()):
        raise ValueError("Cached observations must use the background device")
    else:
        observations = {name: tensor.clone() for name, tensor in observations.items()}
    timestamp = torch.tensor([valid_time.timestamp()], dtype=torch.float64, device=device)
    return {
        analysis_offset_hours: {
            mesh.mesh_id: background,
            "timestamp": timestamp + analysis_offset_hours * 3600,
            "has_analysis": True,
            "has_observations": False,
        },
        0: {
            **observations,
            "timestamp": timestamp,
            "has_analysis": False,
            "has_observations": any(tensor.numel() for tensor in observations.values()),
        },
    }
