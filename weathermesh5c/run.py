from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing
import shutil
import tempfile
import time
from collections.abc import Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from typing import Literal

import torch

from .assets import RuntimeBundle
from .background import (
    BackgroundInfo,
    background_info,
    background_path,
    download_oisst,
    download_open_ifs,
    latest_open_cycle,
    load_background,
    prepare_background,
)
from .checkpoint import EXPECTED_SHA256, load_model
from .ensemble import (
    OPERATIONAL_FORK_COUNT,
    OPERATIONAL_FORK_HOUR,
    OPERATIONAL_MEMBERS,
    OPERATIONAL_SURFACE_LEADS,
    DiskStatistics,
    EnsemblePlan,
    decode_products_for_transfer,
    rollout_member,
)
from .ingest import (
    PUBLIC_SOURCES,
    build_batch,
    download,
    fetch_windborne,
    latest_common_cycle,
    load_observations,
    process_file,
)
from .ingest.storage import write_hours
from .output import (
    DecodedProducts,
    ForecastFile,
    OutputGrid,
    PhysicalProducts,
    ProductTransform,
    aggregate_decoded_products,
    write_manifest,
)
from .transfer import AsyncProductConsumer
from .writer import ProductFileJob, write_statistics_file

UTC = timezone.utc
MINIMUM_FREE_CUDA_GIB = 16


@dataclass(frozen=True)
class ForecastRequest:
    """Inputs for one complete AIDA and WeatherMesh-5c ensemble forecast."""

    bundle_root: Path | None = None
    cycle: datetime | None = None
    assimilation_steps: int | None = None
    members: tuple[int, ...] | None = None
    surface_leads: tuple[int, ...] | None = None
    pressure_leads: tuple[int, ...] | None = None
    fork_hour: int = OPERATIONAL_FORK_HOUR
    fork_count: int = OPERATIONAL_FORK_COUNT
    seed: int = 0
    pressure_group_size: int = 6
    device: str = "cuda:0"
    verify_checksum: bool = True
    raw_root: Path = Path("data/raw")
    background_root: Path = Path("data/backgrounds")
    background: Path | None = None
    observations_root: Path = Path("data/processed")
    sources: tuple[str, ...] = PUBLIC_SOURCES
    ingest_observations: bool = True
    ingest_windborne: bool = False
    max_observations: int | None = None
    download_workers: int = 8
    ingest_workers: int = 4
    output_workers: int = 4
    output_format: Literal["netcdf", "zarr"] = "zarr"
    output_compression: bool = True
    output_root: Path = Path("data/forecasts")
    member_output: bool = True
    overwrite: bool = False


@dataclass(frozen=True)
class _AggregationRequest:
    lead: int
    overall: DiskStatistics
    member: DiskStatistics | None
    pressure_group: DiskStatistics | None


def _cycle(value: str) -> datetime:
    try:
        cycle = datetime.strptime(value, "%Y%m%d%H").replace(tzinfo=UTC)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Cycle must use YYYYMMDDHH in UTC") from error
    if cycle.hour % 6:
        raise argparse.ArgumentTypeError("Cycle hour must be 00, 06, 12, or 18 UTC")
    return cycle


def _lead_hours(values: list[int]) -> tuple[int, ...]:
    leads = tuple(sorted(set(values)))
    if not leads or leads[0] < 0 or any(lead % 3 for lead in leads):
        raise ValueError("Lead hours must be unique non-negative multiples of three")
    return leads


def _members(values: list[int]) -> tuple[int, ...]:
    members = tuple(values)
    if not members or len(set(members)) != len(members):
        raise ValueError("Members must be non-empty and unique")
    if any(not 1 <= member <= 50 for member in members):
        raise ValueError("Open IFS perturbed members are numbered 1 through 50")
    return members


def _ensemble_plan(
    request: ForecastRequest,
    background_member: int | None = None,
    *,
    pressure_offset: int = 0,
) -> EnsemblePlan:
    requested_members = request.members
    if background_member is not None:
        members = _members(list(requested_members or (background_member,)))
        if members != (background_member,):
            raise ValueError(
                f"Prepared background contains member {background_member}; request only that member"
            )
    else:
        members = _members(list(requested_members or OPERATIONAL_MEMBERS))

    surface_leads = (
        _lead_hours(list(request.surface_leads))
        if request.surface_leads is not None
        else OPERATIONAL_SURFACE_LEADS
    )
    if request.pressure_leads is None:
        pressure_leads = tuple(
            lead for lead in surface_leads if lead <= 120 and lead % 6 == pressure_offset
        )
    else:
        pressure_leads = _lead_hours(list(request.pressure_leads)) if request.pressure_leads else ()
    return EnsemblePlan(
        members=members,
        surface_leads=surface_leads,
        pressure_leads=pressure_leads,
        fork_hour=request.fork_hour,
        fork_count=request.fork_count,
        seed=request.seed,
        pressure_group_size=request.pressure_group_size,
    )


def _sources(values: list[str]) -> tuple[str, ...]:
    if values == ["all"]:
        return PUBLIC_SOURCES
    unknown = set(values) - set(PUBLIC_SOURCES)
    if unknown:
        raise ValueError(f"Unknown NOAA observation sources: {', '.join(sorted(unknown))}")
    return tuple(dict.fromkeys(values))


def _latest_compatible_cycle(
    sources: tuple[str, ...], assimilation_steps: int | None = None
) -> datetime:
    cycle = latest_open_cycle()
    now = datetime.now(UTC)
    for _ in range(12):
        try:
            steps = _assimilation_steps(
                ForecastRequest(assimilation_steps=assimilation_steps), cycle, now
            )
            for offset in range(0, steps * 3 + 6, 6):
                latest_common_cycle(sources, now=cycle + timedelta(hours=offset), lookback=1)
            return cycle
        except FileNotFoundError:
            cycle -= timedelta(hours=6)
    raise FileNotFoundError("No common Open IFS and NOAA observation cycle is available")


def _assimilation_steps(request: ForecastRequest, cycle: datetime, now: datetime) -> int:
    if request.assimilation_steps is not None:
        if request.assimilation_steps not in (2, 3):
            raise ValueError("assimilation_steps must be two or three")
        return request.assimilation_steps
    if request.cycle is not None or request.background is not None:
        return 2
    return 3 if (now - cycle).total_seconds() > 7.75 * 3600 else 2


def _require_observation_decoder(sources: tuple[str, ...]) -> None:
    if sources and importlib.util.find_spec("ncepbufr") is None:
        raise RuntimeError(
            "NOAA BUFR processing needs ncepbufr. Run ./scripts/install-ncepbufr.sh first."
        )


def _prepare_background(request: ForecastRequest, cycle: datetime, member: int):
    if request.background:
        info = background_info(request.background)
        if info.cycle != cycle:
            raise ValueError(f"Background cycle is {info.cycle:%Y%m%d%H}, not {cycle:%Y%m%d%H}")
        return request.background

    prepared_path = background_path(request.background_root, cycle, member)
    if prepared_path.is_file():
        info = background_info(prepared_path)
        if info.cycle == cycle and info.member == member:
            return prepared_path

    grib = download_open_ifs(
        request.raw_root,
        cycle,
        member=member,
        workers=request.download_workers,
    )
    oisst = download_oisst(request.raw_root, cycle)
    return prepare_background(
        grib,
        oisst,
        request.background_root,
        cycle=cycle,
        member=member,
    )


def _ingest_source(
    source: str,
    raw_root: Path,
    observations_root: Path,
    cycle: datetime,
) -> tuple[str, int, int]:
    started = time.monotonic()
    print(f"observation_source={source} status=downloading", flush=True)
    raw = download(source, raw_root, cycle=cycle, feed="gdas")
    receipt = observations_root / ".ingested" / source / f"{cycle:%Y%m%d%H}.json"
    raw_stamp = _file_stamp(raw)
    try:
        cached = json.loads(receipt.read_text())
        if (
            cached["schema_version"] == 1
            and cached["raw"] == raw_stamp
            and isinstance(cached["files"], dict)
            and isinstance(cached["count"], int)
            and all(
                _file_stamp(observations_root / name) == stamp
                for name, stamp in cached["files"].items()
            )
        ):
            print(f"observation_source={source} status=cached", flush=True)
            return source, cached["count"], len(cached["files"])
    except (OSError, ValueError, KeyError, TypeError):
        pass
    print(f"observation_source={source} status=processing", flush=True)
    hours = process_file(source, raw)
    written = write_hours(observations_root, source, hours)
    count = sum(len(payload["x"]) for payload in hours.values())
    write_manifest(
        receipt,
        {
            "schema_version": 1,
            "raw": raw_stamp,
            "count": count,
            "files": {
                str(path.relative_to(observations_root)): _file_stamp(path) for path in written
            },
        },
    )
    print(
        f"observation_source={source} elapsed_seconds={time.monotonic() - started:.1f}", flush=True
    )
    return source, count, len(written)


def _file_stamp(path: Path) -> list[int]:
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _prepare_backgrounds(
    request: ForecastRequest, cycle: datetime, members: tuple[int, ...]
) -> list[tuple[Path, BackgroundInfo]]:
    files = []
    for member in members:
        path = _prepare_background(request, cycle, member)
        info = background_info(path)
        if info.cycle != cycle or info.member != member:
            raise ValueError(
                f"Prepared background {path} contains {info.cycle:%Y%m%d%H} member {info.member}, "
                f"expected {cycle:%Y%m%d%H} member {member}"
            )
        files.append((path, info))
        print(f"background_member={member} path={path}", flush=True)
    return files


def _prepare_observations(
    request: ForecastRequest,
    cycle: datetime,
    sources: tuple[str, ...],
) -> None:
    if not request.ingest_observations:
        return
    steps = request.assimilation_steps or 2
    if sources:
        _require_observation_decoder(sources)
        # Start the high-volume feeds first to keep long tasks off the queue's tail.
        sources = tuple(
            sorted(sources, key=lambda source: source not in ("atms", "mtiasi", "satwnd", "1bmhs"))
        )
        worker_count = min(request.ingest_workers, len(sources))
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=context) as executor:
            futures = [
                executor.submit(
                    _ingest_source,
                    source,
                    request.raw_root,
                    request.observations_root,
                    cycle + timedelta(hours=offset),
                )
                # GDAS files cover six-hour windows centered on their cycle.
                # Adjacent cycles supply the edges of the three-hour DA windows.
                for offset in range(0, steps * 3 + 6, 6)
                for source in sources
            ]
            for future in as_completed(futures):
                source, count, file_count = future.result()
                print(
                    f"observation_source={source} observations={count} files={file_count}",
                    flush=True,
                )
    for step in range(1, steps + 1) if request.ingest_windborne else ():
        hours = fetch_windborne(cycle + timedelta(hours=3 * step))
        written = write_hours(request.observations_root, "windborne", hours)
        count = sum(len(payload["x"]) for payload in hours.values())
        print(
            f"observation_source=windborne observations={count} files={len(written)}",
            flush=True,
        )


def _observation_counts(observations) -> dict[str, int]:
    return {
        key: int(value.shape[1])
        for key, value in observations.items()
        if isinstance(value, torch.Tensor) and value.ndim >= 3
    }


def _gpu_memory(device: torch.device) -> str:
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    return f"allocated_gib={allocated:.2f} reserved_gib={reserved:.2f}"


def _expected_outputs(
    run_directory: Path, leads: tuple[int, ...], output_format: str
) -> tuple[Path, ...]:
    suffix = "zarr" if output_format == "zarr" else "nc"
    return tuple(run_directory / f"forecast-f{lead:03d}.{suffix}" for lead in leads)


def _prepare_run_directory(path: Path, *, output_root: Path, overwrite: bool) -> None:
    if path.is_symlink():
        raise ValueError(f"Refusing symlinked forecast directory: {path}")
    if path.exists() and not path.is_dir():
        raise FileExistsError(f"Forecast output path is not a directory: {path}")
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(
                f"Forecast directory already contains output; use --overwrite: {path}"
            )
        resolved_path = path.resolve()
        resolved_root = output_root.resolve()
        if resolved_path.parent != resolved_root:
            raise ValueError(f"Refusing to replace forecast directory outside {resolved_root}")
        shutil.rmtree(resolved_path)
    path.mkdir(parents=True, exist_ok=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the operational WeatherMesh-5c ensemble from portable inputs"
    )
    parser.add_argument("--cycle", type=_cycle)
    parser.add_argument(
        "--assimilation-steps",
        type=int,
        choices=(2, 3),
        help="AIDA updates: historical runs default to 2; live runs use background age",
    )
    parser.add_argument("--member", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--members", type=int, nargs="+")
    parser.add_argument("--lead-hours", type=int, nargs="+")
    parser.add_argument("--pressure-lead-hours", type=int, nargs="*")
    parser.add_argument("--fork-hour", type=int, default=OPERATIONAL_FORK_HOUR)
    parser.add_argument("--fork-count", type=int, default=OPERATIONAL_FORK_COUNT)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pressure-group-size", type=int, default=6)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--raw-root", type=Path, default=Path("data/raw"))
    parser.add_argument("--background-root", type=Path, default=Path("data/backgrounds"))
    parser.add_argument("--background", type=Path, help="Use an existing prepared background")
    parser.add_argument("--observations-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--sources", nargs="+", default=["all"])
    parser.add_argument(
        "--windborne",
        action="store_true",
        help="Also ingest official WindBorne super-observations",
    )
    parser.add_argument("--skip-observation-ingest", action="store_true")
    parser.add_argument(
        "--max-observations",
        type=int,
        help="Random subset per source and AIDA window; default: uncapped",
    )
    parser.add_argument("--download-workers", type=int, default=8)
    parser.add_argument("--ingest-workers", type=int, default=4)
    parser.add_argument("--output-workers", type=int, default=4)
    parser.add_argument(
        "--output-format",
        choices=("netcdf", "zarr"),
        default=ForecastRequest.output_format,
        help="Forecast storage format (default: Zarr); Zarr uses packed arrays in a ZIP store",
    )
    parser.add_argument(
        "--no-output-compression",
        action="store_true",
        help="Disable lossless output compression; increases output size",
    )
    parser.add_argument("--output-root", type=Path, default=Path("data/forecasts"))
    parser.add_argument("--skip-member-output", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _request_from_args(args: argparse.Namespace) -> ForecastRequest:
    if args.member is not None and args.members is not None:
        raise ValueError("Use either --member or --members, not both")
    members = args.members
    if args.member is not None:
        members = [args.member]
    return ForecastRequest(
        bundle_root=args.bundle_root,
        cycle=args.cycle,
        assimilation_steps=args.assimilation_steps,
        members=tuple(members) if members is not None else None,
        surface_leads=tuple(args.lead_hours) if args.lead_hours is not None else None,
        pressure_leads=(
            tuple(args.pressure_lead_hours) if args.pressure_lead_hours is not None else None
        ),
        fork_hour=args.fork_hour,
        fork_count=args.fork_count,
        seed=args.seed,
        pressure_group_size=args.pressure_group_size,
        device=args.device,
        verify_checksum=not args.skip_checksum,
        raw_root=args.raw_root,
        background_root=args.background_root,
        background=args.background,
        observations_root=args.observations_root,
        sources=_sources(args.sources),
        ingest_observations=not args.skip_observation_ingest,
        ingest_windborne=args.windborne,
        max_observations=args.max_observations,
        download_workers=args.download_workers,
        ingest_workers=args.ingest_workers,
        output_workers=args.output_workers,
        output_format=args.output_format,
        output_compression=not args.no_output_compression,
        output_root=args.output_root,
        member_output=not args.skip_member_output,
        overwrite=args.overwrite,
    )


def _write_jobs(
    executor: Executor,
    jobs: list[ProductFileJob],
) -> list[ForecastFile]:
    futures = [executor.submit(write_statistics_file, job) for job in jobs]
    return [future.result() for future in futures]


def _output_executor(worker_count: int) -> Executor:
    if worker_count == 1:
        return ThreadPoolExecutor(max_workers=1)
    return ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=multiprocessing.get_context("spawn"),
    )


def run_forecast(request: ForecastRequest) -> Path:
    """Run the portable operational workflow and return its manifest path."""

    command_started = time.monotonic()
    if request.bundle_root is None:
        raise ValueError("ForecastRequest.bundle_root must identify a runtime bundle")
    bundle = RuntimeBundle(request.bundle_root)
    sources = _sources(list(request.sources))
    if request.max_observations is not None and request.max_observations < 1:
        raise ValueError("max_observations must be positive")
    if request.ingest_workers < 1:
        raise ValueError("ingest_workers must be positive")
    if request.output_workers < 1:
        raise ValueError("output_workers must be positive")
    if request.output_format not in {"netcdf", "zarr"}:
        raise ValueError(f"Unknown output format: {request.output_format}")
    if request.background:
        info = background_info(request.background)
        cycle = request.cycle or info.cycle
    else:
        cycle = request.cycle or _latest_compatible_cycle(sources, request.assimilation_steps)
    steps = _assimilation_steps(request, cycle, datetime.now(UTC))
    request = replace(request, assimilation_steps=steps)
    valid_time = cycle + timedelta(hours=3 * steps)
    plan = _ensemble_plan(
        request, info.member if request.background else None, pressure_offset=valid_time.hour % 6
    )
    print(
        f"cycle={cycle:%Y%m%d%H}Z members={plan.members[0]}-{plan.members[-1]} "
        f"member_count={len(plan.members)} fork={plan.fork_hour}h:{plan.fork_count}",
        flush=True,
    )

    run_directory = request.output_root / f"{valid_time:%Y%m%d%H}-ensemble"
    _prepare_run_directory(
        run_directory,
        output_root=request.output_root,
        overwrite=request.overwrite,
    )
    output_paths = _expected_outputs(run_directory, plan.surface_leads, request.output_format)
    output_suffix = output_paths[0].suffix
    manifest_path = run_directory / "manifest.json"
    print(f"aida_valid_time={valid_time:%Y%m%d%H}Z")
    with ThreadPoolExecutor(max_workers=1) as preparation:
        observations_ready = preparation.submit(_prepare_observations, request, cycle, sources)
        background_files = _prepare_backgrounds(request, cycle, plan.members)
        observations_ready.result()

    device = torch.device(request.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        torch.cuda.set_device(device)
        free_bytes, _ = torch.cuda.mem_get_info(device)
        free_gib = free_bytes / 1024**3
        if free_gib < MINIMUM_FREE_CUDA_GIB:
            raise RuntimeError(
                f"{device} has {free_gib:.2f} GiB free; WeatherMesh-5c needs at least "
                f"{MINIMUM_FREE_CUDA_GIB} GiB free for this 4090-safe run"
            )
        torch.cuda.reset_peak_memory_stats(device)

    inference_started = time.monotonic()
    model = load_model(
        bundle,
        device=device,
        verify_checksum=request.verify_checksum,
    )
    model.move_assets(device)
    assimilation_hours = tuple(range(-3 * (steps - 1), 1, 3))
    model.config.assimilation_hours = [assimilation_hours[0]]
    output_mesh = model.decoders[0].mesh
    output_grid = OutputGrid.from_mesh(output_mesh)
    # Reuse each sampled window across ensemble members without changing fork noise.
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(request.seed)
        observations = {
            hour: load_observations(
                model,
                request.observations_root,
                valid_time + timedelta(hours=hour),
                device=device,
                max_observations=request.max_observations,
            )
            for hour in assimilation_hours
        }
    observation_counts = {
        (valid_time + timedelta(hours=hour)).isoformat(): _observation_counts(window)
        for hour, window in observations.items()
    }
    for hour, window in observations.items():
        if not any(tensor.numel() for tensor in window.values()):
            raise RuntimeError(
                f"No processed observations cover {valid_time + timedelta(hours=hour)} "
                f"in {request.observations_root}"
            )
    transform = ProductTransform.from_model(model)
    file_job = partial(
        ProductFileJob,
        grid=output_grid,
        transform=transform,
        background_time=cycle,
        aida_valid_time=valid_time,
        compression=request.output_compression,
    )
    counts_by_member = {}
    outputs = []
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ),
        tempfile.TemporaryDirectory(prefix=".ensemble-work-", dir=run_directory) as work,
        _output_executor(request.output_workers) as output_executor,
    ):
        work = Path(work)
        overall = DiskStatistics(work / "overall")

        def mean_statistics(root: Path) -> DiskStatistics:
            return overall if len(plan.members) == 1 else DiskStatistics(root, mean_only=True)

        pressure_group = (
            mean_statistics(work / "pressure-group-00")
            if plan.pressure_leads and request.member_output
            else None
        )
        pressure_group_index = 0
        pressure_group_members = 0

        def aggregate(request: _AggregationRequest, decoded: DecodedProducts) -> None:
            products = aggregate_decoded_products(decoded, transform)
            request.overall.add(request.lead, products)
            if request.member is not None and request.member is not request.overall:
                request.member.add(request.lead, PhysicalProducts(surface=products.surface))
            if (
                request.pressure_group is not None
                and request.pressure_group is not request.overall
                and products.pressure is not None
            ):
                request.pressure_group.add(
                    request.lead, PhysicalProducts(pressure=products.pressure)
                )

        with AsyncProductConsumer(aggregate, capacity=3) as consumer:
            for member_index, (background_file, info) in enumerate(background_files):
                prepared = load_background(background_file)
                assert prepared.cycle == info.cycle and prepared.member == info.member
                background_tensor = torch.from_numpy(prepared.values).unsqueeze(0).to(device)
                batch = build_batch(
                    model,
                    background_tensor,
                    valid_time,
                    observations_root=request.observations_root,
                    max_observations=request.max_observations,
                    analysis_offset_hours=-3 * steps,
                    observation_cycles=observations,
                )
                counts_by_member[str(prepared.member)] = observation_counts.copy()
                print(
                    f"member={prepared.member} observation_counts={observation_counts}",
                    flush=True,
                )
                state, analysis_dh = model.assimilate(batch)
                if analysis_dh != -3 * steps or not torch.isfinite(state).all():
                    raise RuntimeError(
                        f"AIDA returned an invalid state for member {prepared.member}"
                    )
                del batch, background_tensor, prepared

                member_statistics = (
                    None
                    if not request.member_output
                    else mean_statistics(work / f"member-{info.member:02d}")
                )

                def emit(
                    lead: int,
                    branch: int | None,
                    forecast_state: torch.Tensor,
                    member_store=member_statistics,
                    pressure_store=pressure_group,
                    prepared_member=info.member,
                ) -> None:
                    decoded = decode_products_for_transfer(
                        model,
                        forecast_state,
                        include_pressure=lead in plan.pressure_leads,
                    )
                    consumer.submit(
                        _AggregationRequest(
                            lead,
                            overall,
                            member_store,
                            pressure_store,
                        ),
                        decoded,
                    )
                    branch_label = "deterministic" if branch is None else str(branch)
                    print(
                        f"member={prepared_member} branch={branch_label} lead={lead}h",
                        flush=True,
                    )

                rollout_member(model, state, plan, info.member, emit)
                del state
                consumer.flush()

                if member_statistics is not None:
                    member_statistics.flush()
                    jobs = []
                    for lead in plan.surface_leads:
                        jobs.append(
                            file_job(
                                path=(
                                    run_directory
                                    / "members"
                                    / f"member{info.member:02d}"
                                    / f"forecast-f{lead:03d}{output_suffix}"
                                ),
                                snapshots=(member_statistics.reference(lead, "mean"),),
                                labels=("member_mean",),
                                product_kind="surface",
                                lead_hours=lead,
                                input_member=info.member,
                                trajectory_count=member_statistics.count(lead),
                            )
                        )
                    for result in _write_jobs(output_executor, jobs):
                        print(f"member_forecast={result.path}", flush=True)
                    if member_statistics is not overall:
                        member_statistics.close()

                if pressure_group is not None:
                    pressure_group_members += 1
                    group_complete = pressure_group_members == plan.pressure_group_size
                    ensemble_complete = member_index == len(background_files) - 1
                    if group_complete or ensemble_complete:
                        pressure_group.flush()
                        jobs = []
                        for lead in plan.pressure_leads:
                            jobs.append(
                                file_job(
                                    path=(
                                        run_directory
                                        / "pressure-supermembers"
                                        / f"supermember{pressure_group_index:02d}"
                                        / f"forecast-f{lead:03d}{output_suffix}"
                                    ),
                                    snapshots=(pressure_group.reference(lead, "mean"),),
                                    labels=("supermember_mean",),
                                    product_kind="pressure",
                                    lead_hours=lead,
                                    pressure_supermember=pressure_group_index,
                                    trajectory_count=pressure_group.count(lead, "pressure"),
                                )
                            )
                        for result in _write_jobs(output_executor, jobs):
                            print(f"pressure_supermember={result.path}", flush=True)
                        if pressure_group is not overall:
                            pressure_group.close()
                        pressure_group_index += 1
                        pressure_group_members = 0
                        if not ensemble_complete:
                            pressure_group = mean_statistics(
                                work / f"pressure-group-{pressure_group_index:02d}"
                            )

        overall.flush()
        jobs = [
            file_job(
                path=output_path,
                snapshots=(
                    overall.reference(lead, "mean"),
                    overall.reference(lead, "standard_deviation"),
                ),
                labels=("mean", "standard_deviation"),
                product_kind="all",
                lead_hours=lead,
                trajectory_count=overall.count(lead),
            )
            for lead, output_path in zip(plan.surface_leads, output_paths, strict=True)
        ]
        for lead, result in zip(
            plan.surface_leads,
            _write_jobs(output_executor, jobs),
            strict=True,
        ):
            outputs.append(result)
            print(
                f"ensemble_forecast={result.path} trajectories={overall.count(lead)} "
                f"bytes={result.size_bytes}",
                flush=True,
            )
        overall.close()

    inference_elapsed = time.monotonic() - inference_started
    peak_gib = None
    device_name = str(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_gib = torch.cuda.max_memory_allocated(device) / 1024**3
        device_name = torch.cuda.get_device_name(device)
        print(_gpu_memory(device))
        print(f"peak_allocated_gib={peak_gib:.2f}")

    forecast_records = [
        {
            "lead_hours": lead,
            "valid_time": result.valid_time.isoformat(),
            "path": str(result.path.resolve()),
            "size_bytes": result.size_bytes,
            "trajectory_count": plan.trajectory_count(lead),
        }
        for lead, result in zip(plan.surface_leads, outputs, strict=True)
    ]
    total_elapsed = time.monotonic() - command_started
    manifest = {
        "schema_version": 1,
        "model": "WeatherMesh-5c with AIDA",
        "created_at": datetime.now(UTC).isoformat(),
        "background": {
            "cycle": cycle.isoformat(),
            "members": [
                {
                    "path": str(path.resolve()),
                    "open_ifs_member": info.member,
                    "oisst_date": info.sst_date.isoformat() if info.sst_date else None,
                }
                for path, info in background_files
            ],
        },
        "aida_valid_time": valid_time.isoformat(),
        "forecast_reference_time": (valid_time - timedelta(hours=valid_time.hour % 6)).isoformat(),
        "assimilation_times": list(observation_counts),
        "ensemble": {
            "input_members": list(plan.members),
            "fork_schedule": {str(plan.fork_hour): plan.fork_count},
            "seed": plan.seed,
            "surface_leads": list(plan.surface_leads),
            "pressure_leads": list(plan.pressure_leads),
            "pressure_group_size": plan.pressure_group_size,
            "member_outputs_written": request.member_output,
        },
        "observations": {
            "root": str(request.observations_root.resolve()),
            "sources": [*sources, *(["windborne"] if request.ingest_windborne else [])],
            "counts_by_member": counts_by_member,
            "maximum_per_source": request.max_observations,
            "ingest_skipped": not request.ingest_observations,
        },
        "checkpoint": {
            "path": str(bundle.checkpoint),
            "sha256": EXPECTED_SHA256 if request.verify_checksum else None,
        },
        "runtime_bundle": str(bundle.root),
        "runtime": {
            "device": device_name,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "peak_allocated_gib": peak_gib,
            "aggregation": "bounded_pinned_cpu",
            "inference_elapsed_seconds": inference_elapsed,
            "total_elapsed_seconds": total_elapsed,
        },
        "forecasts": forecast_records,
        "output": {"format": request.output_format, "compression": request.output_compression},
    }
    write_manifest(manifest_path, manifest)
    print(f"manifest={manifest_path}")
    print(f"inference_elapsed_seconds={inference_elapsed:.1f}")
    print(f"total_elapsed_seconds={total_elapsed:.1f}")
    return manifest_path


def main(argv: Sequence[str] | None = None) -> None:
    request = _request_from_args(build_parser().parse_args(argv))
    run_forecast(request)


if __name__ == "__main__":
    main()
