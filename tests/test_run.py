from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import netCDF4
import numpy as np
import pytest
import torch

from weathermesh5c._core.utils import stretched_tau
from weathermesh5c.ensemble import (
    DiskStatistics,
    EnsemblePlan,
    rollout_member,
)
from weathermesh5c.output import (
    DecodedProducts,
    OutputGrid,
    PhysicalProducts,
    ProductTransform,
    aggregate_decoded_products,
    physicalize_statistics,
    select_global_products,
    write_product_file,
)
from weathermesh5c.run import (
    ForecastRequest,
    _assimilation_steps,
    _ensemble_plan,
    _lead_hours,
    _output_executor,
    _prepare_observations,
    _prepare_run_directory,
    _request_from_args,
    _sources,
    build_parser,
)
from weathermesh5c.transfer import AsyncProductConsumer
from weathermesh5c.writer import ProductFileJob, _load_snapshot, write_statistics_file

UTC = timezone.utc


@pytest.mark.parametrize("offset", [0, 3])
def test_pressure_schedule_rollout_matches_deep(offset):
    plan = _ensemble_plan(ForecastRequest(members=(1,), fork_count=1), pressure_offset=offset)
    emitted = []
    rollout_member(
        _RolloutModel(),
        torch.zeros(1),
        plan,
        1,
        lambda lead, branch, state: emitted.append(lead) if lead in plan.pressure_leads else None,
    )
    assert emitted == list(range(offset, 121, 6))


def test_pressure_schedule_rejects_mixed_offsets():
    with pytest.raises(ValueError, match="six-hour"):
        EnsemblePlan(surface_leads=(0, 3, 6), pressure_leads=(0, 3))


def test_assimilation_schedule_uses_background_age_only_for_live_runs():
    cycle = datetime(2026, 8, 21, 18, tzinfo=UTC)
    boundary = cycle + timedelta(hours=7.75)
    assert _assimilation_steps(ForecastRequest(), cycle, boundary) == 2
    assert _assimilation_steps(ForecastRequest(), cycle, boundary + timedelta(seconds=1)) == 3
    assert (
        _assimilation_steps(ForecastRequest(cycle=cycle), cycle, boundary + timedelta(days=5)) == 2
    )
    assert (
        _assimilation_steps(ForecastRequest(cycle=cycle, assimilation_steps=3), cycle, boundary)
        == 3
    )
    with pytest.raises(ValueError, match="two or three"):
        _assimilation_steps(ForecastRequest(assimilation_steps=1), cycle, boundary)


def test_live_selection_requires_later_observation_cycles(monkeypatch):
    import weathermesh5c.run as run

    latest = datetime(2026, 8, 21, 18, tzinfo=UTC)
    checked = []
    monkeypatch.setattr(run, "latest_open_cycle", lambda: latest)

    def available(sources, now, lookback):
        checked.append(now)
        if now > latest:
            raise FileNotFoundError("Not published yet")
        return now

    monkeypatch.setattr(run, "latest_common_cycle", available)
    assert run._latest_compatible_cycle(("adpupa",), 2) == latest - timedelta(hours=6)
    assert checked == [latest, latest + timedelta(hours=6), latest - timedelta(hours=6), latest]


@pytest.mark.parametrize("steps, offsets", [(2, (0, 6)), (3, (0, 6, 12))])
def test_cycling_ingestion_covers_all_windows(monkeypatch, tmp_path, steps, offsets):
    import weathermesh5c.run as run

    calls = []
    cycle = datetime(2026, 8, 21, 18, tzinfo=UTC)
    monkeypatch.setattr(run, "_require_observation_decoder", lambda sources: None)
    monkeypatch.setattr(
        run, "ProcessPoolExecutor", lambda max_workers, mp_context: ThreadPoolExecutor(max_workers)
    )

    def ingest(source, raw, processed, when):
        calls.append((source, when))
        return source, 1, 1

    monkeypatch.setattr(run, "_ingest_source", ingest)
    _prepare_observations(ForecastRequest(assimilation_steps=steps), cycle, ("adpupa",))
    assert sorted(calls) == [("adpupa", cycle + timedelta(hours=h)) for h in offsets]


@pytest.mark.parametrize(
    "change", ["none", "raw", "output", "missing", "receipt", "schema", "malformed"]
)
def test_ingestion_reuses_only_unchanged_inputs_and_outputs(monkeypatch, tmp_path, change):
    import weathermesh5c.run as run

    raw = tmp_path / "raw.bufr"
    raw.write_bytes(b"BUFR")
    output = tmp_path / "hour.npz"
    calls = []
    cycle = datetime(2026, 8, 21, 6, tzinfo=UTC)
    monkeypatch.setattr(run, "download", lambda *args, **kwargs: raw)

    def process(*args):
        calls.append(args)
        return {cycle: {"x": [1, 2]}}

    def write(*args):
        output.write_bytes(b"processed")
        return [output]

    monkeypatch.setattr(run, "process_file", process)
    monkeypatch.setattr(run, "write_hours", write)
    assert run._ingest_source("atms", tmp_path, tmp_path, cycle) == ("atms", 2, 1)
    if change == "raw":
        raw.write_bytes(b"new raw data")
    elif change == "output":
        output.write_bytes(b"changed")
    elif change == "missing":
        output.unlink()
    elif change == "receipt":
        (tmp_path / ".ingested/atms/2026082106.json").write_text("invalid json")
    elif change in {"schema", "malformed"}:
        receipt = tmp_path / ".ingested/atms/2026082106.json"
        cached = json.loads(receipt.read_text())
        cached["schema_version" if change == "schema" else "files"] = None
        receipt.write_text(json.dumps(cached))
    assert run._ingest_source("atms", tmp_path, tmp_path, cycle) == ("atms", 2, 1)
    assert len(calls) == (1 if change == "none" else 2)


def test_failed_processing_does_not_cache_a_success(monkeypatch, tmp_path):
    import weathermesh5c.run as run

    raw = tmp_path / "raw.bufr"
    raw.write_bytes(b"BUFR")
    monkeypatch.setattr(run, "download", lambda *args, **kwargs: raw)

    def fail(*args):
        raise RuntimeError("bad input")

    monkeypatch.setattr(run, "process_file", fail)
    with pytest.raises(RuntimeError, match="bad input"):
        run._ingest_source("atms", tmp_path, tmp_path, datetime(2026, 8, 21, 6, tzinfo=UTC))
    assert not (tmp_path / ".ingested").exists()


def test_background_and_observation_preparation_overlap_and_propagate_failure(
    monkeypatch, tmp_path
):
    import weathermesh5c.run as run

    background_started = Event()
    observations_started = Event()
    monkeypatch.setattr(run, "RuntimeBundle", lambda root: None)

    def backgrounds(*args):
        background_started.set()
        assert observations_started.wait(5)
        return []

    def observations(*args):
        observations_started.set()
        assert background_started.wait(5)
        raise RuntimeError("observation preparation failed")

    monkeypatch.setattr(run, "_prepare_backgrounds", backgrounds)
    monkeypatch.setattr(run, "_prepare_observations", observations)
    request = ForecastRequest(
        bundle_root=tmp_path,
        cycle=datetime(2026, 8, 21, 6, tzinfo=UTC),
        output_root=tmp_path / "output",
    )
    with pytest.raises(RuntimeError, match="observation preparation failed"):
        run.run_forecast(request)


LEVELS = [
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


def _mesh():
    means = torch.full((137,), 10, dtype=torch.float32)
    standard_deviations = torch.full((137,), 2, dtype=torch.float32)
    precipitation = 125 + 9
    means[precipitation] = 0
    standard_deviations[precipitation] = 1
    return SimpleNamespace(
        levels=LEVELS,
        lats=np.array([0.25, 0], dtype=np.float32),
        lons=np.array([0, 0.25, 0.5], dtype=np.float32),
        n_pr=125,
        num_rms_vars=137,
        state_norm_means=means,
        state_norm_stds=standard_deviations,
        pr_quantile_vars=[None] * 750,
        sfc_quantile_vars=[None] * 360,
    )


def test_run_arguments_reject_invalid_leads_and_non_noaa_sources(tmp_path) -> None:
    assert _lead_hours([6, 3, 3]) == (3, 6)
    assert "windborne" not in _sources(["all"])
    with pytest.raises(ValueError, match="multiples of three"):
        _lead_hours([4])
    with pytest.raises(ValueError, match="Unknown NOAA"):
        _sources(["windborne"])

    run_directory = tmp_path / "forecasts" / "cycle"
    run_directory.mkdir(parents=True)
    stale = run_directory / "stale.nc"
    stale.touch()
    with pytest.raises(FileExistsError, match="overwrite"):
        _prepare_run_directory(run_directory, output_root=tmp_path / "forecasts", overwrite=False)
    _prepare_run_directory(run_directory, output_root=tmp_path / "forecasts", overwrite=True)
    assert run_directory.is_dir()
    assert not stale.exists()


def test_cli_arguments_map_to_forecast_request(tmp_path) -> None:
    request = _request_from_args(
        build_parser().parse_args(
            [
                "--bundle-root",
                str(tmp_path / "bundle"),
                "--members",
                "2",
                "4",
                "--lead-hours",
                "0",
                "3",
                "--pressure-lead-hours",
                "0",
                "--windborne",
                "--skip-member-output",
            ]
        )
    )
    assert request == ForecastRequest(
        bundle_root=tmp_path / "bundle",
        members=(2, 4),
        surface_leads=(0, 3),
        pressure_leads=(0,),
        ingest_windborne=True,
        member_output=False,
    )


def test_forecast_ingests_windborne_for_the_aida_valid_time(monkeypatch, tmp_path) -> None:
    cycle = datetime(2026, 8, 21, 0, tzinfo=UTC)
    valid_times = []
    written = []

    def fetch(valid_time):
        valid_times.append(valid_time)
        return {valid_time: {"x": [np.zeros(10)], "available_at": [123]}}

    def write(root, source, hours):
        written.append((root, source, hours))
        return [root / source / "test.npz"]

    monkeypatch.setattr("weathermesh5c.run.fetch_windborne", fetch)
    monkeypatch.setattr("weathermesh5c.run.write_hours", write)

    request = ForecastRequest(
        bundle_root=tmp_path / "bundle",
        observations_root=tmp_path / "observations",
        sources=(),
        ingest_windborne=True,
    )
    _prepare_observations(request, cycle, ())

    assert valid_times == [datetime(2026, 8, 21, hour, tzinfo=UTC) for hour in (3, 6)]
    assert written[0][:2] == (request.observations_root, "windborne")


def test_single_output_worker_does_not_require_multiprocessing() -> None:
    with _output_executor(1) as executor:
        assert isinstance(executor, ThreadPoolExecutor)
        assert executor.submit(abs, -4).result() == 4


class _IncrementProcessor:
    def __call__(self, state):
        return state + 3


class _NoisyForker:
    def __init__(self):
        self.calls = []

    def __call__(self, state, count, hour):
        self.calls.append((count, hour))
        return state + torch.randn_like(state)


class _RolloutModel:
    def __init__(self):
        self.processors = {"3": _IncrementProcessor()}
        self.forker = _NoisyForker()


def _streamed_rollout():
    model = _RolloutModel()
    plan = EnsemblePlan(
        members=(4,),
        surface_leads=(0, 3, 12, 15),
        pressure_leads=(0, 12),
        fork_hour=12,
        fork_count=2,
        seed=27,
    )
    outputs = []
    rollout_member(
        model,
        torch.zeros(1),
        plan,
        4,
        lambda lead, branch, state: outputs.append((lead, branch, state.item())),
    )
    return model, outputs


def test_rollout_streams_learned_branches_with_coherent_seeded_trajectories() -> None:
    model, outputs = _streamed_rollout()
    _, repeated = _streamed_rollout()

    assert outputs == repeated
    assert model.forker.calls == [(1, 12), (1, 12)]
    assert [(lead, branch) for lead, branch, _ in outputs] == [
        (0, None),
        (3, None),
        (12, 0),
        (15, 0),
        (12, 1),
        (15, 1),
    ]
    assert outputs[3][2] == pytest.approx(outputs[2][2] + 3)
    assert outputs[5][2] == pytest.approx(outputs[4][2] + 3)


def test_surface_only_products_use_operational_precipitation_mean_and_wind_speed() -> None:
    mesh = _mesh()
    decoded = torch.zeros((1, 2, 3, 372), dtype=torch.float32)
    surface, pressure = select_global_products(decoded, mesh, surface_only=True)
    transform = ProductTransform(
        global_means=mesh.state_norm_means.numpy(),
        global_standard_deviations=mesh.state_norm_stds.numpy(),
    )
    products = physicalize_statistics(
        aggregate_decoded_products(DecodedProducts(surface=surface), transform),
        transform,
        "mean",
    )

    expected_precipitation = np.trapezoid(
        np.full(30, 3, dtype=np.float32), stretched_tau(30, k=1).numpy()
    )
    assert pressure is None
    np.testing.assert_allclose(products.surface[9], expected_precipitation)
    np.testing.assert_allclose(products.surface[12], np.sqrt(200))
    np.testing.assert_allclose(products.surface[13], np.sqrt(200))


def test_compact_operational_transfer_preserves_pressure_normalization() -> None:
    mesh = _mesh()
    decoded = torch.zeros((1, 2, 3, 1247), dtype=torch.float16)
    surface, pressure = select_global_products(decoded, mesh, surface_only=False)
    transform = ProductTransform(
        global_means=mesh.state_norm_means.numpy(),
        global_standard_deviations=mesh.state_norm_stds.numpy(),
    )

    aggregate = aggregate_decoded_products(
        DecodedProducts(surface=surface.cpu(), pressure=pressure.cpu()),
        transform,
    )
    physical = physicalize_statistics(aggregate, transform, "mean")

    np.testing.assert_allclose(physical.pressure, 10)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda:0",
            marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
        ),
    ],
)
def test_async_product_consumer_preserves_submission_order(device) -> None:
    received = []

    def consume(index, products):
        received.append((index, products.surface.item()))

    with AsyncProductConsumer(consume, capacity=1) as consumer:
        for index in range(3):
            consumer.submit(index, DecodedProducts(surface=torch.tensor([index], device=device)))

    assert received == [(0, 0), (1, 1), (2, 2)]


def test_disk_statistics_and_ensemble_netcdf_preserve_global_products(tmp_path) -> None:
    mesh = _mesh()
    first = PhysicalProducts(
        surface=np.ones((14, 2, 3), np.float32),
        pressure=np.ones((5, 25, 2, 3), np.float32),
    )
    third = PhysicalProducts(
        surface=np.full((14, 2, 3), 3, np.float32),
        pressure=np.full((5, 25, 2, 3), 3, np.float32),
    )
    statistics = DiskStatistics(tmp_path / "statistics")
    statistics.add(12, first)
    statistics.add(12, third)
    mean = statistics.snapshot(12, "mean")
    statistics.flush()
    result = write_statistics_file(
        ProductFileJob(
            path=tmp_path / "ensemble.nc",
            snapshots=(
                statistics.reference(12, "mean"),
                statistics.reference(12, "standard_deviation"),
            ),
            labels=("mean", "standard_deviation"),
            product_kind="all",
            grid=OutputGrid.from_mesh(mesh),
            transform=ProductTransform(
                global_means=np.zeros(137, np.float32),
                global_standard_deviations=np.ones(137, np.float32),
            ),
            background_time=datetime(2026, 8, 21, 0, tzinfo=UTC),
            aida_valid_time=datetime(2026, 8, 21, 3, tzinfo=UTC),
            lead_hours=12,
            trajectory_count=2,
        )
    )

    with netCDF4.Dataset(result.path) as dataset:
        assert dataset.forecast_reference_time == "2026-08-21T00:00:00+00:00"
        assert dataset.forecast_hour == 15
        assert dataset.lead_hours == 12
        assert result.valid_time == datetime(2026, 8, 21, 15, tzinfo=UTC)
        assert dataset.ensemble_trajectory_count == 2
        assert list(dataset.variables["ensemble_statistic"][:]) == [
            "mean",
            "standard_deviation",
        ]
        np.testing.assert_allclose(dataset.variables["air_temperature_2m"][0], 2)
        np.testing.assert_allclose(dataset.variables["air_temperature_2m"][1], np.sqrt(2))
        np.testing.assert_allclose(dataset.variables["geopotential"][0], 2)

    member_result = write_product_file(
        tmp_path / "member.nc",
        (mean,),
        ("member_mean",),
        mesh,
        background_time=datetime(2026, 8, 21, 0, tzinfo=UTC),
        aida_valid_time=datetime(2026, 8, 21, 3, tzinfo=UTC),
        lead_hours=12,
        input_member=1,
        trajectory_count=2,
    )
    with netCDF4.Dataset(member_result.path) as dataset:
        assert dataset.open_ifs_member == 1
        np.testing.assert_allclose(dataset.variables["air_temperature_2m"][:], 2)
        np.testing.assert_allclose(dataset.variables["geopotential"][:], 2)
    statistics.close()


def test_mean_only_disk_statistics_avoid_unused_variance_state(tmp_path) -> None:
    statistics = DiskStatistics(tmp_path / "statistics", mean_only=True)
    statistics.add(3, PhysicalProducts(surface=np.ones((2, 3), np.float32)))
    statistics.add(3, PhysicalProducts(surface=np.full((2, 3), 3, np.float32)))

    np.testing.assert_allclose(statistics.snapshot(3, "mean").surface, 2)
    with pytest.raises(ValueError, match="mean only"):
        statistics.snapshot(3, "standard_deviation")
    assert not (tmp_path / "statistics" / "f003" / "surface-m2.npy").exists()
    statistics.close()


@pytest.mark.parametrize("shape", [(14, 129, 147), (5, 25, 47, 47)])
@pytest.mark.parametrize("layout", ["contiguous", "channels_last", "reversed"])
@pytest.mark.parametrize("mean_only", [False, True])
def test_disk_statistics_preserve_products_across_layouts_and_blocks(
    tmp_path, shape, layout, mean_only
) -> None:
    name = "surface" if len(shape) == 3 else "pressure"
    base = (np.arange(np.prod(shape), dtype=np.float32) % 97 / 4).reshape(shape)
    axes = tuple(range(2, len(shape))) + (0, 1) if len(shape) == 4 else (1, 2, 0)
    inverse = tuple(axes.index(i) for i in range(len(shape)))
    statistics = DiskStatistics(tmp_path / "statistics", mean_only=mean_only)
    for index, offset in enumerate((-3, 1, 5)):
        value = base + offset
        if layout != "contiguous" and index != 1:
            value = np.ascontiguousarray(value.transpose(axes)).transpose(inverse)
            if layout == "reversed":
                value = value[..., ::-1].copy()[..., ::-1]
        before = value.copy()
        statistics.add(12, PhysicalProducts(**{name: value}))
        np.testing.assert_array_equal(value, before)
    statistics.flush()
    for statistic, expected in (
        ("mean", base + 1),
        ("standard_deviation", np.full(shape, 4, np.float32)),
    ):
        if mean_only and statistic == "standard_deviation":
            continue
        direct = getattr(statistics.snapshot(12, statistic), name)
        reopened = getattr(_load_snapshot(statistics.reference(12, statistic)), name)
        np.testing.assert_array_equal(direct, expected)
        np.testing.assert_array_equal(reopened, expected)
    statistics.close()


def test_first_statistics_update_preserves_float32_values_and_zero_spread(tmp_path) -> None:
    bits = np.array([0, 0x80000000, 1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF], np.uint32)
    value = bits.view(np.float32).reshape(1, -1)
    statistics = DiskStatistics(tmp_path / "statistics")
    statistics.add(0, PhysicalProducts(surface=value))
    expected = bits.copy()
    expected[1] = 0
    mean = statistics.snapshot(0, "mean").surface
    np.testing.assert_array_equal(mean.view(np.uint32).reshape(-1), expected)
    np.testing.assert_array_equal(
        statistics.snapshot(0, "standard_deviation").surface, np.zeros_like(value)
    )
    with pytest.raises(ValueError, match="Product shape"):
        statistics.add(0, PhysicalProducts(surface=value[:, :-1]))
    np.testing.assert_array_equal(mean.view(np.uint32).reshape(-1), expected)
    statistics.close()


def test_statistics_keep_welford_rounding(tmp_path) -> None:
    statistics = DiskStatistics(tmp_path / "statistics")
    expected = (
        (1065353216, 0),
        (1069547520, 1060439283),
        (1075139925, 1069778419),
        (3196059648, 1084888339),
        (1077097267, 1090758116),
    )
    for value, (mean_bits, spread_bits) in zip((1, 2, 4, -8, 15), expected, strict=True):
        statistics.add(0, PhysicalProducts(surface=np.array([value], np.float32)))
        assert statistics.snapshot(0, "mean").surface.view(np.uint32).item() == mean_bits
        assert (
            statistics.snapshot(0, "standard_deviation").surface.view(np.uint32).item()
            == spread_bits
        )
    statistics.close()


def test_statistics_update_readonly_inputs_across_blocks(tmp_path) -> None:
    shape = (1, 1, 4_194_305)
    statistics = DiskStatistics(tmp_path / "statistics")
    for value in (1, 3):
        array = np.broadcast_to(np.float32(value), shape)
        statistics.add(0, PhysicalProducts(surface=array))
        np.testing.assert_array_equal(array, value)
    np.testing.assert_array_equal(statistics.snapshot(0, "mean").surface, 2)
    np.testing.assert_array_equal(
        statistics.snapshot(0, "standard_deviation").surface, np.float32(np.sqrt(2))
    )
    statistics.close()


def test_statistics_initialize_variance_with_original_signed_zero_rounding(tmp_path) -> None:
    statistics = DiskStatistics(tmp_path / "statistics")
    higher = np.nextafter(np.float32(1), np.float32(2))
    samples = (
        np.array([0, -0.0, higher, 1, 1e-38, -1e-38], np.float32),
        np.array([-0.0, 0, 1, higher, -1e-38, 1e-38], np.float32),
        np.array([1, -1, 0, 0, 0, 0], np.float32),
    )
    mean, m2 = np.zeros_like(samples[0]), np.zeros_like(samples[0])
    for count, value in enumerate(samples, 1):
        difference = value - mean
        mean += difference / count
        m2 += difference * (value - mean)
        statistics.add(0, PhysicalProducts(surface=value))
        spread = np.sqrt(m2 / (count - 1)) if count > 1 else np.zeros_like(mean)
        for statistic, expected in (("mean", mean), ("standard_deviation", spread)):
            actual = statistics.snapshot(0, statistic).surface
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
    statistics.close()


@pytest.mark.parametrize("members", [(1,), tuple(range(1, 8))])
@pytest.mark.parametrize("member_output", [False, True])
def test_forecast_member_and_group_means_preserve_ensemble_state(
    monkeypatch, tmp_path, members, member_output
):
    import weathermesh5c.run as run

    cycle = datetime(2026, 9, 21, tzinfo=UTC)
    backgrounds = [
        (
            tmp_path / str(member),
            SimpleNamespace(cycle=cycle, member=member, sst_date=cycle.date()),
        )
        for member in members
    ]
    model = _RolloutModel()
    model.config = SimpleNamespace()
    model.decoders = [SimpleNamespace(mesh=_mesh())]
    model.move_assets = lambda device: None
    model.assimilate = lambda batch: (batch.reshape(-1)[:1], -6)
    monkeypatch.setattr(
        run,
        "RuntimeBundle",
        lambda root: SimpleNamespace(
            root=root, checkpoint=tmp_path / "checkpoint", checkpoint_sha256="test"
        ),
    )
    monkeypatch.setattr(run, "_prepare_backgrounds", lambda *args: backgrounds)
    monkeypatch.setattr(run, "_prepare_observations", lambda *args: None)
    monkeypatch.setattr(
        run,
        "load_background",
        lambda path: SimpleNamespace(
            cycle=cycle, member=int(path.name), values=np.array([int(path.name)], np.float32)
        ),
    )
    monkeypatch.setattr(run, "load_model", lambda *args, **kwargs: model)
    monkeypatch.setattr(run, "load_observations", lambda *args, **kwargs: {"x": torch.ones(1)})
    monkeypatch.setattr(run, "_observation_counts", lambda window: {"public": 1})
    samples = {}

    def decode(model, state, *, include_pressure):
        return DecodedProducts(
            torch.full((1, 2, 3, 12), state.item()),
            torch.full((1, 2, 3, 125), state.item()) if include_pressure else None,
        )

    original_rollout = run.rollout_member

    def rollout(model, state, plan, member, emit):
        def record(lead, branch, forecast):
            samples.setdefault((member, lead), []).append(float(forecast.item()))
            emit(lead, branch, forecast)

        original_rollout(model, state, plan, member, record)

    monkeypatch.setattr(run, "build_batch", lambda model, background, *args, **kwargs: background)
    monkeypatch.setattr(run, "decode_products_for_transfer", decode)
    monkeypatch.setattr(run, "rollout_member", rollout)
    request = ForecastRequest(
        bundle_root=tmp_path,
        cycle=cycle,
        assimilation_steps=2,
        members=members,
        surface_leads=(0, 3, 6, 9, 12, 15),
        pressure_leads=(0, 6, 12),
        fork_hour=12,
        fork_count=2,
        device="cpu",
        ingest_observations=False,
        output_workers=1,
        output_format="netcdf",
        output_root=tmp_path / "outputs",
        member_output=member_output,
    )
    manifest = json.loads(run.run_forecast(request).read_text())
    root = Path(manifest["forecasts"][0]["path"]).parent
    for lead in request.surface_leads:
        values = np.array([value for member in members for value in samples[member, lead]])
        with netCDF4.Dataset(root / f"forecast-f{lead:03d}.nc") as file:
            np.testing.assert_allclose(
                file["air_temperature_2m"][0], values.mean() * 2 + 10, rtol=2e-6
            )
            spread = values.std(ddof=1) * 2 if len(values) > 1 else 0
            np.testing.assert_allclose(file["air_temperature_2m"][1], spread, rtol=2e-6)
        for member in members if member_output else ():
            with netCDF4.Dataset(
                root / "members" / f"member{member:02d}" / f"forecast-f{lead:03d}.nc"
            ) as file:
                np.testing.assert_allclose(
                    file["air_temperature_2m"][:],
                    np.mean(samples[member, lead]) * 2 + 10,
                    rtol=2e-6,
                )
                assert "geopotential" not in file.variables
        if member_output and lead in request.pressure_leads:
            for group, start in enumerate(range(0, len(members), 6)):
                values = [
                    value
                    for member in members[start : start + 6]
                    for value in samples[member, lead]
                ]
                with netCDF4.Dataset(
                    root
                    / "pressure-supermembers"
                    / f"supermember{group:02d}"
                    / f"forecast-f{lead:03d}.nc"
                ) as file:
                    np.testing.assert_allclose(
                        file["geopotential"][:], np.mean(values) * 2 + 10, rtol=2e-6
                    )
                    assert "air_temperature_2m" not in file.variables
    if not member_output:
        assert not (root / "members").exists()
        assert not (root / "pressure-supermembers").exists()
