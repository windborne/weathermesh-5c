from __future__ import annotations

import importlib
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from weathermesh5c._core.meshes import (
    GPSROData,
    IASIPcaData,
    MicrowaveData,
    RadiosondeData,
    SatwindData,
    SurfaceData,
    WindborneData,
)
from weathermesh5c.ingest.cli import _source_list
from weathermesh5c.ingest.download import download, latest_common_cycle, obsproc_url
from weathermesh5c.ingest.prepare import (
    NORM_MAPPING,
    _load_mesh,
    _normalize,
    _norms,
    _prepare_hour,
    build_batch,
)
from weathermesh5c.ingest.process import process_file
from weathermesh5c.ingest.sources import MICROWAVE_STATS, SOURCES
from weathermesh5c.ingest.storage import processed_path, write_hours
from weathermesh5c.ingest.windborne import fetch_windborne

UTC = timezone.utc


@pytest.mark.parametrize("mesh_type", [RadiosondeData, WindborneData])
def test_humidity_normalization_fills_missing_levels(mesh_type):
    mesh = mesh_type()
    norms = _norms()
    levels = np.asarray(norms["levels"])
    pressures = np.array([20, 210, 225, 240, 775, 825], dtype=np.float32)
    raw = np.zeros((len(pressures), len(mesh.disk_varlist)), dtype=np.float32)
    raw[:, mesh.disk_varlist.index("pres_pa")] = pressures * 100
    raw[:, mesh.disk_varlist.index("unnorm_rh")] = 0.5
    output, columns = _normalize(raw, mesh)
    norm = norms[NORM_MAPPING["rh"]]
    mean, std = [np.asarray(norm[key]) for key in ("mean", "std")]
    expected = 0.5 - np.interp(pressures, levels[~np.isnan(mean)], mean[~np.isnan(mean)])
    expected /= np.interp(pressures, levels[~np.isnan(std)], std[~np.isnan(std)])
    np.testing.assert_array_equal(output[:, columns.index("rh")], expected.astype(np.float32))


def test_gpsro_coordinates_use_stored_half_arithmetic():
    mesh = GPSROData()
    raw = np.zeros((1, 300, len(mesh.disk_varlist)), dtype=np.float16)
    for name, value in (
        ("latint", 47),
        ("latdec", 0.1234),
        ("lonint", 121),
        ("londec", 0.5678),
        ("all_present", 1),
    ):
        raw[..., mesh.disk_varlist.index(name)] = value
    prepared = _prepare_hour(raw, mesh, 0)
    for stem in ("lat", "lon"):
        expected = (
            raw[..., mesh.disk_varlist.index(stem + "int")]
            + raw[..., mesh.disk_varlist.index(stem + "dec")]
        )
        np.testing.assert_array_equal(
            prepared[..., mesh.full_varlist.index(stem + "_deg")], expected
        )


@pytest.mark.parametrize(
    "instrument, ids, rows, indices",
    [
        ("1bmhs", [3, 223, 3, 999, np.nan], [0, 1, 2], [4, 0, 4]),
        ("atms", [225, 999, 224, 226], [0, 2, 3], [1, 2, 0]),
        ("1bamua", [999, np.nan], [], []),
    ],
)
def test_microwave_preparation_preserves_satellite_order_and_known_rows(
    instrument, ids, rows, indices
) -> None:
    mesh = MicrowaveData(instrument)
    mesh.sats.reverse()
    raw = np.zeros((len(ids), len(mesh.disk_varlist)), dtype=np.float16)
    raw[:, mesh.disk_varlist.index("said")] = ids
    raw[:, mesh.disk_varlist.index("chan_0")] = np.arange(1, len(ids) + 1)
    before = raw.copy()

    prepared = _prepare_hour(raw, mesh, 0)

    onehot = np.zeros((len(rows), len(mesh.sats)), dtype=np.float32)
    onehot[np.arange(len(rows)), indices] = 1
    columns = [mesh.full_varlist.index(f"onehot_said_{index}") for index in range(len(mesh.sats))]
    np.testing.assert_array_equal(prepared[:, columns], onehot)
    np.testing.assert_array_equal(
        prepared[:, mesh.full_varlist.index("chan_0")], np.array(rows) + 1
    )
    np.testing.assert_array_equal(raw, before)


@pytest.mark.parametrize("cap", [None, 17, 600_000])
def test_observation_sampling_matches_deep_and_uncapped_keeps_every_row(monkeypatch, tmp_path, cap):
    when = datetime(2026, 9, 21, 6, tzinfo=UTC)
    mesh = SimpleNamespace(
        string_id="radiosonde",
        encoder_subset=float("inf"),
        da_window_size=(-1.5, 1.5),
        full_varlist=["reltime_hours", "id"],
    )
    # Exceed both former limits, with two rows outside the assimilation window.
    count = 500_001
    rows = np.column_stack((np.zeros(count + 2), np.arange(count + 2))).astype(np.float32)
    rows[-2:, 0] = [-2, 2]
    path = processed_path(tmp_path, "adpupa", when)
    path.parent.mkdir(parents=True)
    np.savez(path, x=rows)
    monkeypatch.setattr("weathermesh5c.ingest.prepare._prepare_hour", lambda data, *args: data)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(19)
        expected = (
            torch.randperm(count)[:cap].numpy()
            if cap is not None and cap < count
            else np.arange(count)
        )
        torch.random.default_generator.manual_seed(19)
        before = torch.random.get_rng_state()
        actual = _load_mesh(mesh, tmp_path, when, max_observations=cap)
        assert np.array_equal(actual[0, :, 1].numpy(), expected)
        assert actual.shape[1] == len(expected)
        if cap is None or cap >= count:
            assert torch.equal(before, torch.random.get_rng_state())
        else:
            assert len(np.unique(actual[0, :, 1])) == cap
            torch.random.default_generator.manual_seed(19)
            assert torch.equal(actual, _load_mesh(mesh, tmp_path, when, max_observations=cap))
            torch.random.default_generator.manual_seed(20)
            assert not torch.equal(actual, _load_mesh(mesh, tmp_path, when, max_observations=cap))


@pytest.mark.parametrize("rep", [False, True])
def test_numeric_bufr_reader_preserves_masks_shapes_and_buffer_reuse(monkeypatch, rep):
    module = importlib.import_module("weathermesh5c.ingest.process")
    missing = 1e11
    reports = [
        np.array(
            [
                [0, missing, missing + 999_999, missing + 1_000_001],
                [np.nan, np.inf, -np.inf, 273.15],
            ]
        ),
        np.array([[missing], [17.5]]),
        np.empty((2, 0)),
    ]
    calls = []

    def decode(unit, buffer, query, fields, levels):
        assert unit == 7 and query == "A B"
        assert buffer.flags.f_contiguous
        report = reports[len(calls)]
        calls.append(buffer)
        buffer[:, : report.shape[1]] = report
        return report.shape[1]

    def wrong_mode(*args):
        pytest.fail("Reader chose the wrong replication mode")

    native = SimpleNamespace(
        ufbint=wrong_mode if rep else decode,
        ufbrep=decode if rep else wrong_mode,
    )
    monkeypatch.setattr(module, "_bufr_module", lambda: SimpleNamespace(_bufrlib=native, _maxdim=8))
    reader = module._numeric_reader(SimpleNamespace(lunit=7, missing_value=missing))
    for report in reports:
        expected = np.ma.masked_values(report, missing).filled(np.nan)
        np.testing.assert_array_equal(reader("A B", rep=rep), expected, strict=True)
    assert all(buffer is calls[0] for buffer in calls)


def test_iasi_channel_scales_preserve_bin_order_missing_values_and_table_changes():
    module = importlib.import_module("weathermesh5c.ingest.process")
    # Overlapping bins use the first match; missing or unmatched scales use one.
    table = np.array(
        [[16, 16, 49, 51, 55], [38, 49, 49, 51, 55], [2, 3, 4, np.nan, 1e11]],
        dtype=np.float64,
    )
    indices = np.searchsorted(module.real_chans, [16, 38, 49, 51, 55, 57])
    key = table.tobytes()
    first = module._iasi_channel_scales(key)
    np.testing.assert_array_equal(first[indices], [0.01, 0.01, 0.001, 1, 1, 1])
    table[2, 0] = 1
    second = module._iasi_channel_scales(table.tobytes())
    np.testing.assert_array_equal(second[indices], [0.1, 0.1, 0.001, 1, 1, 1])
    np.testing.assert_array_equal(module._iasi_channel_scales(key), first)


def test_noaa_source_list_keeps_windborne_api_selection_explicit() -> None:
    assert "windborne" not in _source_list(["all"])
    with pytest.raises(ValueError, match="Unknown NOAA"):
        _source_list(["windborne"])


def test_nomads_url_uses_exact_operational_name() -> None:
    cycle = datetime(2026, 8, 21, 6, tzinfo=UTC)
    assert obsproc_url("adpsfc", cycle) == (
        "https://nomads.ncep.noaa.gov/pub/data/nccf/com/obsproc/prod/"
        "gdas.20260821/gdas.t06z.adpsfc.tm00.bufr_d.nr"
    )


def test_download_streams_to_the_expected_path(tmp_path) -> None:
    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def iter_content(self, chunk_size):
            assert chunk_size == 1024 * 1024
            yield b"BUFR"
            yield b"test"

        def close(self):
            return None

    class Session:
        def get(self, url, stream, timeout):
            assert url.endswith("gdas.t06z.adpupa.tm00.bufr_d")
            assert stream is True
            assert timeout == (10, 600)
            return Response()

    cycle = datetime(2026, 8, 21, 6, tzinfo=UTC)
    path = download("adpupa", tmp_path, cycle=cycle, session=Session())

    assert path.read_bytes() == b"BUFRtest"
    assert path == tmp_path / "adpupa/gdas/20260821/gdas.t06z.adpupa.tm00.bufr_d"
    assert download("adpupa", tmp_path, cycle=cycle, session=Session()) == path
    tomorrow = cycle.replace(day=22)
    assert download("adpupa", tmp_path, cycle=tomorrow, session=Session()) != path


def test_latest_common_cycle_waits_for_every_source(monkeypatch) -> None:
    def available(source, cycle, **kwargs):
        return cycle.hour == 6 or source == "1bamua"

    module = importlib.import_module("weathermesh5c.ingest.download")
    monkeypatch.setattr(module, "is_available", available)

    cycle = latest_common_cycle(
        ("1bamua", "mtiasi"),
        now=datetime(2026, 8, 21, 12, tzinfo=UTC),
    )

    assert cycle == datetime(2026, 8, 21, 6, tzinfo=UTC)


def test_source_columns_match_the_5c_mesh_contracts() -> None:
    meshes = {
        "1bmhs": MicrowaveData("1bmhs"),
        "1bamua": MicrowaveData("1bamua"),
        "atms": MicrowaveData("atms"),
        "gpsro": GPSROData(),
        "mtiasi": IASIPcaData(),
        "adpupa": RadiosondeData(),
        "windborne": WindborneData(),
        "satwnd": SatwindData(),
        "adpsfc": SurfaceData(),
    }
    for source, mesh in meshes.items():
        assert SOURCES[source].columns == tuple(mesh.disk_varlist)


@pytest.mark.parametrize("source", ["1bamua", "1bmhs", "atms"])
def test_microwave_masked_headers_keep_fallback_coordinates_and_reject_missing_angles(
    monkeypatch, source
):
    class Bufr:
        message = -1
        subset = -1
        closed = False

        def advance(self):
            self.message += 1
            return self.message

        def load_subset(self):
            self.subset += 1
            return int(self.subset == 2)

        def read_subset(self, query, **kwargs):
            if query == "TMBR":
                values = MICROWAVE_STATS[source][0]
            else:
                angle = 30 if self.subset == 0 else np.nan
                values = [1, 2026, 8, 21, 6, 30, 0, 10.5, 20.5, angle, 40, np.nan, np.nan]
            return np.ma.masked_invalid(np.asarray(values)[:, None])

        def close(self):
            self.closed = True

    bufr = Bufr()
    module = importlib.import_module("weathermesh5c.ingest.process")
    monkeypatch.setattr(module, "_bufr_module", lambda: SimpleNamespace(open=lambda _: bufr))
    hours = process_file(source, "microwave.bufr", available_at=123)
    payload = hours[datetime(2026, 8, 21, 7, tzinfo=UTC)]
    assert bufr.closed
    assert len(payload["x"]) == 1
    row = dict(zip(SOURCES[source].columns, payload["x"][0], strict=True))
    assert row["latint"] + row["latdec"] == 10.5
    assert row["lonint"] + row["londec"] == 20.5
    assert row["reltime_hours"] == -0.5
    assert row["chan_0"] == 0


def test_windborne_csv_processing_preserves_the_legacy_schema(tmp_path) -> None:
    path = tmp_path / "windborne.csv"
    path.write_text(
        "timestamp,id,latitude,longitude,pressure,speed_u,speed_v,temperature,"
        "humidity,specific_humidity,mission_name,mission_id\n"
        "1787293800,7,37.5,-122.2,850,4,-3,12,0.4,5.5,test,mission\n"
    )

    hours = process_file("windborne", path, available_at=123)
    payload = next(iter(hours.values()))
    row = np.asarray(payload["x"])[0]

    assert row.shape == (len(SOURCES["windborne"].columns),)
    assert row[SOURCES["windborne"].columns.index("pres_pa")] == 85_000
    assert np.isclose(row[SOURCES["windborne"].columns.index("unnorm_temp_k")], 285.15)
    assert payload["available_at"] == [123]


def test_windborne_api_fetches_super_observations_and_keeps_latest_ids() -> None:
    valid_time = datetime(2026, 8, 21, 3, tzinfo=UTC)
    window_start = int(valid_time.timestamp() - 1.5 * 3600)
    window_end = int(valid_time.timestamp() + 1.5 * 3600)
    calls = []

    def page_client(**kwargs):
        calls.append(kwargs)
        base = {
            "id": 7,
            "timestamp": window_start,
            "updated_at": 10,
            "latitude": 37.5,
            "longitude": -122.2,
            "pressure": 850,
            "speed_u": 4,
            "speed_v": -3,
            "temperature": 10,
            "humidity": 40,
            "specific_humidity": 5_500,
        }
        if len(calls) == 1:
            return {
                "observations": [base],
                "has_next_page": True,
                "next_since": window_start + 1,
            }
        return {
            "observations": [
                {**base, "updated_at": 11, "temperature": 12},
                {**base, "id": 8, "timestamp": window_end, "pressure": 500},
                {**base, "id": 9, "timestamp": window_end, "pressure": None},
            ],
            "has_next_page": False,
        }

    hours = fetch_windborne(
        valid_time,
        available_at=123,
        request_minutes=180,
        page_client=page_client,
    )
    rows = np.concatenate([payload["x"] for payload in hours.values()])
    columns = {
        name: SOURCES["windborne"].columns.index(name) for name in SOURCES["windborne"].columns
    }

    assert len(calls) == 2
    assert calls[0] == {
        "since": window_start,
        "min_time": window_start,
        "max_time": window_end,
        "include_ids": True,
        "include_mission_name": True,
        "include_updated_at": True,
    }
    assert calls[1]["since"] == window_start + 1
    assert rows.shape == (2, len(SOURCES["windborne"].columns))
    np.testing.assert_array_equal(rows[:, columns["id"]], [7, 8])
    np.testing.assert_allclose(rows[:, columns["pres_pa"]], [85_000, 50_000])
    assert rows[0, columns["unnorm_temp_k"]] == pytest.approx(285.15)
    assert rows[0, columns["unnorm_rh"]] == 40
    assert rows[0, columns["unnorm_sh_mgpkg"]] == 5_500
    assert all(payload["available_at"] == [123] for payload in hours.values())


def test_windborne_api_rejects_a_stalled_pagination_cursor() -> None:
    valid_time = datetime(2026, 8, 21, 3, tzinfo=UTC)

    def page_client(**kwargs):
        return {
            "observations": [],
            "has_next_page": True,
            "next_since": kwargs["since"],
        }

    with pytest.raises(RuntimeError, match="did not move forward"):
        fetch_windborne(valid_time, request_minutes=180, page_client=page_client)


def test_surface_bufr_processing_preserves_the_operational_input_contract(monkeypatch) -> None:
    identity = [2026, 8, 20, 21, 53, 10.126, -20.234, np.nan]
    basic = [100_000, 101_324, 102_000, 90.04, 5.04, 300.15, 290.15, 50, 1234.56, np.nan, 2.04]
    no_data = [np.nan] * len(basic)

    class Bufr:
        def __init__(self):
            self.messages = (
                ("NC000001", ((identity, basic), (identity, no_data))),
                ("NC000101", ((identity, basic),)),
                ("NC001001", (([*identity[:5], -10.126, 20.234, 12.9], basic),)),
            )
            self.message_index = -1
            self.subset_index = -1
            self.closed = False

        def advance(self):
            self.message_index += 1
            if self.message_index == len(self.messages):
                return 1
            self.msg_type = self.messages[self.message_index][0]
            self.subset_index = -1
            return 0

        def load_subset(self):
            self.subset_index += 1
            subsets = self.messages[self.message_index][1]
            return int(self.subset_index == len(subsets))

        def read_subset(self, query):
            values = self.messages[self.message_index][1][self.subset_index]
            index = int(query.startswith("PRES"))
            return np.ma.masked_invalid(np.asarray(values[index], dtype=np.float64)[:, None])

        def close(self):
            self.closed = True

    bufr = Bufr()
    module = importlib.import_module("weathermesh5c.ingest.process")
    monkeypatch.setattr(module, "_bufr_module", lambda: SimpleNamespace(open=lambda _: bufr))

    hours = process_file("adpsfc", "surface.bufr", available_at=123)
    payload = next(iter(hours.values()))
    rows = np.asarray(payload["x"])
    columns = {name: SOURCES["adpsfc"].columns.index(name) for name in SOURCES["adpsfc"].columns}

    assert bufr.closed
    assert rows.shape == (2, len(SOURCES["adpsfc"].columns))
    np.testing.assert_array_equal(rows[:, columns["is_ship"]], [0, 1])
    np.testing.assert_allclose(rows[:, columns["reltime_hours"]], [-7 / 60, -7 / 60])
    np.testing.assert_allclose(rows[:, columns["lat_deg"]], [10.13, -10.13])
    np.testing.assert_allclose(rows[:, columns["lon_deg"]], [-20.23, 20.23])
    np.testing.assert_array_equal(rows[:, columns["unnorm_elev_m"]], [99999, 12])
    np.testing.assert_allclose(rows[:, columns["unnorm_mslp_hpa"]], 1013.2)
    np.testing.assert_allclose(rows[:, columns["unnorm_2t_c"]], 27.0)
    np.testing.assert_allclose(rows[:, columns["unnorm_2d_c"]], 17.0)
    np.testing.assert_allclose(rows[:, columns["unnorm_10u_mps"]], -5.0, atol=1e-6)
    np.testing.assert_allclose(rows[:, columns["unnorm_10v_mps"]], 0.0, atol=1e-6)
    np.testing.assert_allclose(rows[:, columns["unnorm_viz_m"]], 1234.6)
    assert np.isnan(rows[:, columns["unnorm_3hpr_mm"]]).all()
    np.testing.assert_allclose(rows[:, columns["unnorm_24hpr_mm"]], 2.0)
    assert payload["available_at"] == [123, 123]


def test_storage_is_atomic_and_deduplicates_rows(tmp_path) -> None:
    hour = datetime(2026, 8, 21, 7, tzinfo=UTC)
    payload = {hour: {"x": [np.array([1, 2], dtype=np.float32)], "available_at": [3]}}

    write_hours(tmp_path, "adpupa", payload)
    write_hours(tmp_path, "adpupa", payload)

    with np.load(processed_path(tmp_path, "adpupa", hour)) as stored:
        assert stored["x"].shape == (1, 2)
        np.testing.assert_array_equal(stored["available_at"], [3])


def test_radiosonde_rows_become_finite_model_inputs() -> None:
    mesh = RadiosondeData()
    raw = np.array([[0, -0.5, 40, -105, 85_000, 3, np.nan, 280, 0.5]], dtype=np.float32)

    prepared = _prepare_hour(raw, mesh, datetime(2026, 8, 21, 7, tzinfo=UTC).timestamp())

    assert prepared.shape == (1, len(mesh.full_varlist))
    assert np.isfinite(prepared).all()
    assert prepared[0, mesh.full_varlist.index("ucomp_present?")] == 1
    assert prepared[0, mesh.full_varlist.index("vcomp_present?")] == 0


def test_build_batch_loads_processed_observations(tmp_path) -> None:
    valid_time = datetime(2026, 8, 21, 7, tzinfo=UTC)
    mesh = RadiosondeData(da_window_size=(-1.5, 1.5))
    raw = np.array([[0, -0.5, 40, -105, 85_000, 3, -2, 280, 0.5]], dtype=np.float32)
    write_hours(
        tmp_path,
        "adpupa",
        {valid_time: {"x": raw, "available_at": [valid_time.timestamp()]}},
    )
    analysis_mesh = SimpleNamespace(mesh_id="analysis", shape=lambda: [2, 3])
    model = SimpleNamespace(
        anl_encoders=[SimpleNamespace(mesh=analysis_mesh)],
        obs_encoders=[SimpleNamespace(meshes=[mesh])],
    )

    batch = build_batch(
        model,
        torch.zeros((1, 2, 3)),
        valid_time,
        observations_root=tmp_path,
    )

    assert batch[-3]["analysis"].shape == (1, 2, 3)
    assert batch[0][mesh.mesh_id].shape == (1, 1, len(mesh.full_varlist))
    assert batch[0]["has_observations"] is True


def test_build_batch_reuses_prepared_observations_across_members(tmp_path) -> None:
    valid_time = datetime(2026, 8, 21, 7, tzinfo=UTC)
    analysis_mesh = SimpleNamespace(mesh_id="analysis", shape=lambda: [2, 3])
    model = SimpleNamespace(anl_encoders=[SimpleNamespace(mesh=analysis_mesh)])
    cached = {"observations": torch.ones((1, 4, 2))}

    first = build_batch(
        model,
        torch.zeros((1, 2, 3)),
        valid_time,
        observations_root=tmp_path / "missing",
        observations=cached,
    )
    second = build_batch(
        model,
        torch.ones((1, 2, 3)),
        valid_time,
        observations_root=tmp_path / "missing",
        observations=cached,
    )

    assert first[0]["observations"] is not cached["observations"]
    assert second[0]["observations"] is not first[0]["observations"]
    assert torch.equal(first[0]["observations"], cached["observations"])
    assert torch.equal(second[0]["observations"], cached["observations"])
    assert not torch.equal(first[-3]["analysis"], second[-3]["analysis"])


@pytest.mark.parametrize("steps", [2, 3])
def test_cycling_batch_preserves_times_and_cached_observations(tmp_path, steps):
    valid_time = datetime(2026, 8, 22, 0, tzinfo=UTC)
    model = SimpleNamespace(
        anl_encoders=[
            SimpleNamespace(mesh=SimpleNamespace(mesh_id="analysis", shape=lambda: [2, 3]))
        ]
    )
    windows = {hour: {"obs": torch.ones(1, 2, 3)} for hour in range(-3 * (steps - 1), 1, 3)}
    batch = build_batch(
        model,
        torch.zeros(1, 2, 3),
        valid_time,
        observations_root=tmp_path,
        analysis_offset_hours=-3 * steps,
        observation_cycles=windows,
    )
    assert list(batch) == list(range(-3 * steps, 1, 3))
    for hour, entry in batch.items():
        assert entry["timestamp"].item() == valid_time.timestamp() + hour * 3600
        assert entry["has_analysis"] == (hour == -3 * steps)
    batch[0]["obs"].zero_()
    assert windows[0]["obs"].count_nonzero() == 6
    windows[0] = {"obs": torch.empty(1, 0, 0)}
    with pytest.raises(ValueError, match="No observations"):
        build_batch(
            model,
            torch.zeros(1, 2, 3),
            valid_time,
            observations_root=tmp_path,
            analysis_offset_hours=-3 * steps,
            observation_cycles=windows,
        )
