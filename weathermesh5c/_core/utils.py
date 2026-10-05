from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

PACKAGE_ROOT = Path(__file__).resolve().parent
NORM_PATH = PACKAGE_ROOT / "norm/norm.json"

levels_ecm1 = [10, 30, 50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
levels_joank = [
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
]
levels_medium = [
    10,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
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
    875,
    900,
    925,
    950,
    975,
    1000,
]
levels_full = [
    1,
    2,
    3,
    5,
    7,
    10,
    20,
    30,
    50,
    70,
    100,
    125,
    150,
    175,
    200,
    225,
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
    775,
    800,
    825,
    850,
    875,
    900,
    925,
    950,
    975,
    1000,
]
quantile_levels = [50, 250, 500, 750, 950]
core_pressure_vars = ["129_z", "130_t", "131_u", "132_v", "133_q"]
core_sfc_vars = ["165_10u", "166_10v", "167_2t", "151_msl"]
MICROWAVE_N_CHANNELS = {"1bmhs": 5, "1bamua": 15, "atms": 22}


def stretched_tau(
    n: int,
    k: float = 0.0,
    *,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    uniform = (torch.arange(n, device=device, dtype=dtype) + 0.5) / n
    if k == 0:
        return uniform
    exponent = 1 / (1 + k)
    return 0.5 * (1 + torch.sign(uniform - 0.5) * torch.abs(2 * uniform - 1) ** exponent)


def get_quantile_vars(variables: list[str], count: int) -> list[str]:
    result = []
    for variable in variables:
        if variable in core_pressure_vars:
            result.extend(
                f"{variable}_{level}_qt_{index}/{count}"
                for level in quantile_levels
                for index in range(count)
            )
        else:
            result.extend(f"{variable}_qt_{index}/{count}" for index in range(count))
    return result


def get_date(value):
    if isinstance(value, datetime):
        if value.tzinfo not in {None, timezone.utc}:
            raise ValueError("The date must use UTC")
        return value.replace(tzinfo=timezone.utc)
    if np.issubdtype(type(value), np.number):
        return datetime(1970, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=int(value))
    raise TypeError(f"Unsupported date value: {value!r}")


def load_state_norm(mesh):
    with Path(NORM_PATH).open() as stream:
        norms = json.load(stream)
    standard_deviations = []
    means = []
    for variable in mesh.pressure_vars:
        standard_deviations.append(np.asarray(norms[variable]["std"])[mesh.wh_lev])
        means.append(np.asarray(norms[variable]["mean"])[mesh.wh_lev])
    for variable in mesh.sfc_vars:
        if "_qt_" in variable:
            continue
        if variable == "zeropad":
            standard_deviations.append(np.asarray([1]))
            means.append(np.asarray([0]))
        else:
            standard_deviations.append(np.asarray([norms[variable]["std"]]))
            means.append(np.asarray([norms[variable]["mean"]]))
    std = torch.from_numpy(np.concatenate(standard_deviations).astype(np.float32))
    mean = torch.from_numpy(np.concatenate(means).astype(np.float32))
    return std, mean
