from __future__ import annotations

import json
from datetime import datetime, timezone
from importlib.resources import files

import numpy as np
import pytest
import requests
import torch

from weathermesh5c.background.download import (
    OPEN_LEVELS,
    PRESSURE_PARAMS,
    SURFACE_PARAMS,
    download_oisst,
    select_index_entries,
)
from weathermesh5c.background.prepare import (
    ENCODER_LEVELS,
    MODEL_LEVELS,
    OpenIFSFields,
    _expand_levels,
    build_background,
    build_sst,
    interpolate_missing_levels,
)


def test_background_expansion_matches_deep_half_arithmetic():
    values = np.random.default_rng(731).normal(size=(16, 8, 8)).astype(np.float32)
    actual = _expand_levels(dict(zip(MODEL_LEVELS, values, strict=True)))
    stored = torch.from_numpy(values).half()
    for level, output in zip(ENCODER_LEVELS, actual, strict=True):
        upper = int(np.searchsorted(MODEL_LEVELS, level))
        lower = max(upper - 1, 0)
        weight = (
            0
            if upper == 0
            else ((level - MODEL_LEVELS[lower]) / (MODEL_LEVELS[upper] - MODEL_LEVELS[lower]))
        )
        expected = stored[lower] * (1 - weight) + stored[upper] * weight
        np.testing.assert_array_equal(output, expected.numpy())


def _index_row(offset, *, param, member=1, levtype="sfc", level=None):
    row = {
        "_offset": offset,
        "_length": 8,
        "number": member,
        "step": 0,
        "levtype": levtype,
        "param": param,
    }
    if level is not None:
        row["levelist"] = str(level)
    return row


def _complete_index() -> str:
    rows = []
    offset = 0
    for aliases in PRESSURE_PARAMS.values():
        for level in OPEN_LEVELS:
            rows.append(_index_row(offset, param=aliases[0], levtype="pl", level=level))
            offset += 8
    for param in SURFACE_PARAMS.values():
        rows.append(_index_row(offset, param=param))
        offset += 8
    rows.append(_index_row(offset, param="tcc", member=2))
    return "\n".join(json.dumps(row) for row in rows)


def _norms():
    path = files("weathermesh5c._core.norm") / "norm.json"
    with path.open() as stream:
        return json.load(stream)


def _fields(shape=(2, 3)) -> OpenIFSFields:
    norms = _norms()
    norm_levels = list(norms["levels"])
    pressure = {}
    for variable_index, variable in enumerate(PRESSURE_PARAMS):
        pressure[variable] = {}
        for level in OPEN_LEVELS:
            index = norm_levels.index(level)
            normalized = variable_index / 10 + level / 1000
            value = norms[variable]["mean"][index] + norms[variable]["std"][index] * normalized
            pressure[variable][level] = np.full(shape, value, dtype=np.float32)
    surface = {}
    for variable_index, variable in enumerate(SURFACE_PARAMS):
        if variable == "lsm":
            surface[variable] = np.zeros(shape, dtype=np.float32)
        elif variable == "skt":
            surface[variable] = np.full(shape, 280, dtype=np.float32)
        else:
            normalized = np.float32(variable_index + 1)
            value = norms[variable]["mean"] + norms[variable]["std"] * normalized
            surface[variable] = np.full(shape, value, dtype=np.float32)
    return OpenIFSFields(pressure, surface)


def test_index_selects_the_complete_member_specific_tensor_contract() -> None:
    entries = select_index_entries(_complete_index(), member=1)

    assert {entry.param for entry in entries} >= {"gh", "t", "u", "v", "q", "skt", "lsm"}
    assert all(entry.offset < entries[-1].offset for entry in entries[:-1])

    rows = _complete_index().splitlines()
    incomplete = "\n".join(
        row for row in rows if not ('"param": "q"' in row and '"levelist": "10"' in row)
    )
    with pytest.raises(ValueError, match="133_q@10hPa"):
        select_index_entries(incomplete, member=1)


def test_missing_pressure_levels_use_log_pressure_and_positive_humidity() -> None:
    fields = _fields(shape=(1, 1))
    fields.pressure["133_q"][10][0, 0] = np.float32(1e-8)
    fields.pressure["133_q"][50][0, 0] = np.float32(1e-5)

    complete = interpolate_missing_levels(fields.pressure)

    assert 1e-8 < complete["133_q"][30][0, 0] < 1e-5
    q70 = complete["133_q"][70][0, 0]
    endpoints = (1e-5, fields.pressure["133_q"][100][0, 0])
    assert min(endpoints) < q70 < max(endpoints)
    assert all(np.isfinite(complete[variable][30]).all() for variable in PRESSURE_PARAMS)


def test_background_matches_the_5c_channel_order_and_padding() -> None:
    fields = _fields()
    fields.surface["lsm"][0, 0] = 1
    oisst = np.full((2, 3), 290, dtype=np.float32)
    oisst[0, 1] = np.nan

    background = build_background(fields, oisst=oisst)

    for variable_index in range(len(PRESSURE_PARAMS)):
        level_index = ENCODER_LEVELS.index(100)
        channel = variable_index * len(ENCODER_LEVELS) + level_index
        np.testing.assert_allclose(background[..., channel], variable_index / 10 + 0.1, atol=1e-3)
        level_index = ENCODER_LEVELS.index(350)
        channel = variable_index * len(ENCODER_LEVELS) + level_index
        np.testing.assert_allclose(background[..., channel], variable_index / 10 + 0.35, atol=1e-3)

    expected_surface = np.broadcast_to(np.arange(1, 8), background[..., 125:132].shape)
    np.testing.assert_allclose(background[..., 125:132], expected_surface, atol=1e-3)
    assert background[0, 0, 132] == 0
    expected_fallback = (280 - _norms()["034_sstk"]["mean"]) / _norms()["034_sstk"]["std"]
    assert np.isclose(background[0, 1, 132], expected_fallback, atol=1e-3)
    assert np.all(background[..., 133:] == 0)
    assert background.dtype == np.float16


def test_sst_uses_oisst_over_ocean_and_skin_temperature_only_for_gaps() -> None:
    fields = _fields()
    fields.surface["lsm"][0, 0] = 1
    oisst = np.full((2, 3), 291, dtype=np.float32)
    oisst[1, 2] = np.nan

    sst = build_sst(fields, oisst)

    assert np.isnan(sst[0, 0])
    assert sst[0, 1] == 291
    assert sst[1, 2] == fields.surface["skt"][1, 2]


def test_oisst_download_uses_the_latest_available_prior_day(tmp_path) -> None:
    payload = b"\x89HDF" + bytes(1020)

    class Response:
        def __init__(self, status_code):
            self.status_code = status_code

        def raise_for_status(self):
            if self.status_code >= 400:
                raise requests.HTTPError(response=self)

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            yield payload

        def close(self):
            return None

    class Session:
        def __init__(self):
            self.urls = []

        def get(self, url, **kwargs):
            self.urls.append(url)
            available = "20260820_preliminary.nc" in url
            return Response(200 if available else 404)

    session = Session()
    cycle = datetime(2026, 8, 22, 6, tzinfo=timezone.utc)
    path = download_oisst(tmp_path, cycle, session=session)

    assert path.name == "oisst-avhrr-v02r01.20260820_preliminary.nc"
    assert any("20260821_preliminary.nc" in url for url in session.urls)
    assert path.read_bytes() == payload
