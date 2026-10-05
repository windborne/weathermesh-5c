from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

import numpy as np
import pytest

import weathermesh5c.assets as assets
from weathermesh5c.assets import BUNDLE_SCHEMA_VERSION, RuntimeBundle, sha256_file


def test_replace_file_can_reuse_package_cache_storage(tmp_path: Path) -> None:
    source = tmp_path / "source"
    target = tmp_path / "nested" / "target"
    source.write_bytes(b"runtime asset")

    assets._replace_file(source, target, hardlink=True)

    assert source.samefile(target)


def test_resolved_bundle_uses_flat_package_cache(tmp_path: Path, monkeypatch) -> None:
    from weathermesh5c._core.model import data

    names = ("land_mask.npy", "soil_type.npy", "topography.npy")
    monkeypatch.setattr(
        assets,
        "global_bundle_files",
        lambda: tuple(PurePosixPath("assets/global") / name for name in names),
    )
    mapping = {}
    entries = []
    files = [
        ("weights/weather-mesh-5c-aida.pt", "checkpoint"),
        *((f"assets/global/{name}", "global") for name in names),
    ]
    for index, (relative, role) in enumerate(files):
        path = tmp_path / f"cached-{index}"
        path.write_bytes(role.encode())
        mapping[relative] = path
        entries.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
            }
        )
    manifest = tmp_path / "cached-manifest"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "model": "WeatherMesh-5c with AIDA",
                "files": entries,
            }
        )
    )
    mapping["manifest.json"] = manifest
    bundle = RuntimeBundle(tmp_path, resolve=mapping.__getitem__)
    bundle.validate()

    def load(path):
        assert path in [mapping[f"assets/global/{name}"] for name in names]
        return np.ones((3, 3))

    monkeypatch.setattr(data, "load_array", load)
    constants, variables = data.get_constant_vars(
        SimpleNamespace(xpos=np.zeros((2, 3, 2))),
        bundle.global_root,
        resolve=bundle.global_path,
    )
    assert constants.shape == (2, 3, len(variables))
    assert constants[..., variables.index("land_mask")].all()
    assert not (tmp_path / "assets").exists()


def test_bundle_integrity(tmp_path: Path, monkeypatch) -> None:
    checkpoint = tmp_path / "weights" / "weather-mesh-5c-aida.pt"
    global_file = tmp_path / "assets" / "global" / "constant.npy"
    for path, content in (
        (checkpoint, b"checkpoint"),
        (global_file, b"global"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    monkeypatch.setattr(
        assets,
        "global_bundle_files",
        lambda: (PurePosixPath("assets/global/constant.npy"),),
    )
    entries = []
    for path, role in ((checkpoint, "checkpoint"), (global_file, "global")):
        entries.append(
            {
                "path": path.relative_to(tmp_path).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
            }
        )
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "model": "WeatherMesh-5c with AIDA",
                "files": entries,
            }
        )
    )
    bundle = RuntimeBundle(tmp_path)

    bundle.validate(checksums=True)

    global_file.write_bytes(b"GLOBAL")
    with pytest.raises(ValueError, match="SHA-256"):
        bundle.validate(checksums=True)


def test_materialize_runtime_bundle_reuses_valid_files(tmp_path: Path, monkeypatch) -> None:
    source = tmp_path / "source"
    destination = tmp_path / "destination"
    checkpoint = source / "weights" / "weather-mesh-5c-aida.pt"
    global_file = source / "assets" / "global" / "constant.npy"
    for path, content in (
        (checkpoint, b"checkpoint"),
        (global_file, b"global"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    monkeypatch.setattr(
        assets,
        "global_bundle_files",
        lambda: (PurePosixPath("assets/global/constant.npy"),),
    )
    entries = []
    for path, role in (
        (checkpoint, "checkpoint"),
        (global_file, "global"),
    ):
        entries.append(
            {
                "path": path.relative_to(source).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
                "role": role,
            }
        )
    (source / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": BUNDLE_SCHEMA_VERSION,
                "model": "WeatherMesh-5c with AIDA",
                "files": entries,
            }
        )
    )
    resolved = []

    def resolve(relative: str) -> Path:
        resolved.append(relative)
        return source / relative

    bundle = RuntimeBundle.materialize(resolve, destination)
    assert resolved == [
        "manifest.json",
        "assets/global/constant.npy",
        "weights/weather-mesh-5c-aida.pt",
    ]
    bundle.validate()

    resolved.clear()
    RuntimeBundle.materialize(resolve, destination)
    assert resolved == ["manifest.json"]

    (destination / "assets/global/constant.npy").write_bytes(b"GLOBAL")
    resolved.clear()
    RuntimeBundle.materialize(resolve, destination)
    assert resolved == ["manifest.json", "assets/global/constant.npy"]
