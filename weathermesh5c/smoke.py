from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch

from . import load_model
from .background import load_background
from .ingest.prepare import build_batch


def gpu_memory(device: torch.device) -> str:
    allocated = torch.cuda.memory_allocated(device) / 1024**3
    reserved = torch.cuda.memory_reserved(device) / 1024**3
    return f"allocated_gib={allocated:.2f} reserved_gib={reserved:.2f}"


def synthetic_background(model, device: torch.device):
    mesh = model.anl_encoders[0].mesh
    shape = tuple(mesh.shape())
    background = torch.full((1, *shape), 0.01, dtype=torch.float16, device=device)
    pad_start = mesh.n_pr + mesh.n_sfc - mesh.extra_sfc_pad
    background[..., pad_start:] = 0
    return background


def empty_observation_inputs(model, background, analysis_time: datetime):
    mesh = model.anl_encoders[0].mesh
    expected = tuple(mesh.shape())
    if tuple(background.shape) == expected:
        background = background.unsqueeze(0)
    if tuple(background.shape[1:]) != expected:
        raise ValueError(f"Background shape must be {expected}, got {tuple(background.shape[1:])}")
    device = background.device
    observations = {}
    for encoder in model.obs_encoders:
        for obs_mesh in encoder.meshes:
            observations[obs_mesh.mesh_id] = torch.empty(
                (1, 0, 0), dtype=torch.float32, device=device
            )
    analysis_t0 = torch.tensor([analysis_time.timestamp()], dtype=torch.float64, device=device)
    batch = {
        -3: {
            mesh.mesh_id: background,
            "timestamp": analysis_t0,
            "has_analysis": True,
            "has_observations": False,
        },
        0: {
            **observations,
            "timestamp": analysis_t0 + 3 * 3600,
            "has_analysis": False,
            "has_observations": True,
        },
    }
    return batch


def _finite(value) -> bool:
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, (list, tuple)):
        return all(_finite(item) for item in value)
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    return True


def _shape(value):
    if isinstance(value, torch.Tensor):
        return tuple(value.shape)
    if isinstance(value, (list, tuple)):
        return [_shape(item) for item in value]
    if isinstance(value, dict):
        return {key: _shape(item) for key, item in value.items()}
    return type(value).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description="WeatherMesh-5c single-GPU smoke test")
    parser.add_argument("--bundle-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--load-only", action="store_true")
    parser.add_argument("--skip-checksum", action="store_true")
    parser.add_argument("--background", type=Path)
    parser.add_argument("--observations-root", type=Path)
    parser.add_argument("--valid-time", help="AIDA UTC time as YYYYMMDDHH")
    parser.add_argument("--max-observations", type=int)
    parser.add_argument("--decode", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "CUDA is not available"
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    model = load_model(
        args.bundle_root,
        device=device,
        verify_checksum=not args.skip_checksum,
    )
    print(f"model_loaded keys={len(model.state_dict())}")
    if device.type == "cuda":
        print(gpu_memory(device))
    if args.load_only:
        return

    model.move_assets(device)
    model.config.assimilation_hours = [0]
    if args.background:
        prepared = load_background(args.background)
        background = torch.from_numpy(prepared.values).unsqueeze(0).to(device)
        default_valid_time = prepared.cycle + timedelta(hours=3)
        print(f"background_cycle={prepared.cycle:%Y%m%d%H}Z member={prepared.member}")
        if prepared.sst_date:
            print(f"oisst_date={prepared.sst_date:%Y%m%d}")
    else:
        background = synthetic_background(model, device)
        default_valid_time = datetime(2025, 10, 14, 12, tzinfo=timezone.utc)
    valid_time = (
        datetime.strptime(args.valid_time, "%Y%m%d%H").replace(tzinfo=timezone.utc)
        if args.valid_time
        else default_valid_time
    )
    if args.observations_root:
        batch = build_batch(
            model,
            background,
            valid_time,
            observations_root=args.observations_root,
            max_observations=args.max_observations,
        )
        counts = {
            key: value.shape[1]
            for key, value in batch[0].items()
            if isinstance(value, torch.Tensor) and value.ndim >= 3
        }
        print(f"observation_counts={counts}")
    else:
        batch = empty_observation_inputs(model, background, valid_time - timedelta(hours=3))
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=device.type == "cuda",
        ),
    ):
        analysis, analysis_dh = model.assimilate(batch)
        forecast = model.processors["3"](analysis)
        decoded = None
        if args.decode:
            normal = model.predecoder(forecast)
            decoder = model.decoders[0]
            decoded = decoder(normal)

    assert analysis_dh == -3
    assert analysis.shape == forecast.shape
    assert torch.isfinite(analysis).all()
    assert torch.isfinite(forecast).all()
    assert decoded is None or _finite(decoded)
    print(f"analysis_shape={tuple(analysis.shape)} analysis_dh={analysis_dh}")
    print(f"forecast_shape={tuple(forecast.shape)} forecast_hour=3")
    if decoded is not None:
        print(f"global_forecast_shape={_shape(decoded)} forecast_hour=3")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = torch.cuda.max_memory_allocated(device) / 1024**3
        print(f"peak_allocated_gib={peak:.2f}")


if __name__ == "__main__":
    main()
