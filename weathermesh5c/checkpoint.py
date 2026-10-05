from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import torch

from .assets import RuntimeBundle, sha256_file
from .model import build_model

EXPECTED_BYTES = 2_841_543_169
EXPECTED_ELEMENTS = 762_997_823
EXPECTED_KEYS = 984
EXPECTED_SHA256 = "20d6a06cf75f6fa42c8796bc39fa7e426ea08be390a895b6236a6f03aabe73d2"


@dataclass(frozen=True)
class CheckpointInfo:
    path: Path
    size_bytes: int
    sha256: str | None
    keys: int | None = None
    tensor_elements: int | None = None


def checkpoint_info(
    checkpoint: str | Path,
    *,
    verify_checksum: bool = False,
    inspect_tensors: bool = False,
) -> CheckpointInfo:
    path = Path(checkpoint).expanduser().resolve()
    size = path.stat().st_size
    digest = sha256_file(path) if verify_checksum else None
    keys = None
    elements = None
    if inspect_tensors:
        state = load_checkpoint_state(path)
        keys = len(state)
        elements = sum(tensor.numel() for tensor in state.values())
    return CheckpointInfo(path, size, digest, keys, elements)


def load_checkpoint_state(
    checkpoint: str | Path,
) -> Mapping[str, torch.Tensor]:
    path = Path(checkpoint).expanduser().resolve()
    state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if not isinstance(state, Mapping) or not state:
        raise TypeError("WeatherMesh-5c checkpoint must be a non-empty state dictionary")
    if not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise TypeError("WeatherMesh-5c checkpoint contains non-tensor state")
    return state


def validate_checkpoint(checkpoint: str | Path, *, checksum: bool = True) -> None:
    path = Path(checkpoint).expanduser().resolve()
    size = path.stat().st_size
    if size != EXPECTED_BYTES:
        raise ValueError(f"Checkpoint size is {size}; expected {EXPECTED_BYTES}")
    if checksum:
        digest = sha256_file(path)
        if digest != EXPECTED_SHA256:
            raise ValueError(f"Checkpoint SHA-256 is {digest}; expected {EXPECTED_SHA256}")


def load_model(
    bundle: str | Path | RuntimeBundle,
    *,
    device: str | torch.device = "cpu",
    verify_checksum: bool = True,
):
    bundle = bundle if isinstance(bundle, RuntimeBundle) else RuntimeBundle(bundle)
    bundle.validate(checksums=False)
    checkpoint = bundle.checkpoint
    validate_checkpoint(checkpoint, checksum=verify_checksum)
    state = load_checkpoint_state(checkpoint)
    if len(state) != EXPECTED_KEYS:
        raise ValueError(f"Checkpoint has {len(state)} keys; expected {EXPECTED_KEYS}")
    elements = sum(tensor.numel() for tensor in state.values())
    if elements != EXPECTED_ELEMENTS:
        raise ValueError(f"Checkpoint has {elements} tensor elements; expected {EXPECTED_ELEMENTS}")

    model = build_model(bundle=bundle)
    model.load_state_dict(state, strict=True)
    del state
    model.eval()
    return model.to(device)
