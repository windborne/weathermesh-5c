#!/usr/bin/env python3
"""Convert trusted production inputs into a portable WeatherMesh-5c runtime bundle."""

from __future__ import annotations

import argparse
import errno
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import torch

from weathermesh5c.assets import (
    BUNDLE_SCHEMA_VERSION,
    CHECKPOINT_FILE,
    GLOBAL_ROOT,
    RuntimeBundle,
    global_bundle_files,
    sha256_file,
)
from weathermesh5c.checkpoint import EXPECTED_BYTES, EXPECTED_SHA256

_GLOBAL_ARRAYS = ("land_mask.npy", "soil_type.npy", "topography.npy")
_HGRIDS = (
    "hgrid_nu32_D6_KL21_KD5_B16",
    "hgrid_da_442adbe5eb4a688c",
    "hgrid_da_791b788601d0fc08",
)


def _source_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    return path


def _source_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def _destination(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise argparse.ArgumentTypeError(f"Destination already exists: {path}")
    return path


def _transfer(source: Path, destination: Path, *, link: bool) -> None:
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"Source file is missing or symlinked: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if link:
        try:
            os.link(source, destination)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(source, destination)
    else:
        shutil.copyfile(source, destination)


def _numpy(value) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _trusted_pickle(path: Path):
    # This conversion tool accepts only trusted WindBorne production inputs.
    # Runtime code never opens pickle files.
    with path.open("rb") as stream:
        return pickle.load(stream)


def _convert_hgrid(source: Path, destination: Path) -> None:
    grid = _trusted_pickle(source)
    if not isinstance(grid, dict):
        raise TypeError(f"Expected an attention-grid dictionary: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        vertices=_numpy(grid["vertices"]),
        which=_numpy(grid["which"]),
        mask2=_numpy(grid["mask2"]),
        lonlat=_numpy(grid["lonlat"]),
    )


def _convert_encoder_mesh(source: Path, destination: Path) -> None:
    vertices, (g2m_indices, g2m_weights), _ = _trusted_pickle(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        vertices=_numpy(vertices),
        g2m_idx=_numpy(g2m_indices),
        g2m_weights=_numpy(g2m_weights),
    )


def _convert_encoder_edges(source: Path, destination: Path) -> None:
    indices, edge_features = _trusted_pickle(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        g2m_idx=_numpy(indices),
        edge_features=_numpy(edge_features),
    )


def _convert_decoder_mesh(source: Path, destination: Path) -> None:
    gathers, subgathers, edge_features = _trusted_pickle(source)
    if not (len(gathers) == len(subgathers) == len(edge_features) == 40):
        raise ValueError(f"Expected 40 global-decoder chunks: {source}")
    arrays = {}
    for index, (gather, subgather, edges) in enumerate(
        zip(gathers, subgathers, edge_features, strict=True)
    ):
        arrays[f"gather_{index}"] = _numpy(gather)
        arrays[f"subgather_{index}"] = _numpy(subgather)
        arrays[f"edge_features_{index}"] = _numpy(edges)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(destination, **arrays)


def _entry(root: Path, relative: Path, role: str) -> dict[str, object]:
    path = root / relative
    return {
        "path": relative.as_posix(),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "role": role,
    }


def _write_manifest(root: Path) -> None:
    files = [(Path(CHECKPOINT_FILE), "checkpoint")]
    files.extend((Path(path), "global") for path in global_bundle_files())
    entries = []
    for index, (relative, role) in enumerate(files, start=1):
        entries.append(_entry(root, relative, role))
        if index % 500 == 0:
            print(f"hashed_files={index}/{len(files)}", flush=True)
    checkpoint = entries[0]
    if checkpoint["size_bytes"] != EXPECTED_BYTES or checkpoint["sha256"] != EXPECTED_SHA256:
        raise ValueError("The source checkpoint is not the production WeatherMesh-5c state")
    manifest = {
        "schema_version": BUNDLE_SCHEMA_VERSION,
        "model": "WeatherMesh-5c with AIDA",
        "files": entries,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def build_bundle(args) -> None:
    destination = args.output
    working = destination.with_name(f".{destination.name}.building-{os.getpid()}")
    if working.exists():
        raise FileExistsError(f"Temporary bundle path already exists: {working}")
    working.mkdir(parents=True)

    _transfer(
        args.checkpoint,
        working.joinpath(*CHECKPOINT_FILE.parts),
        link=args.link_existing_files,
    )
    global_root = working.joinpath(*GLOBAL_ROOT.parts)
    for name in _GLOBAL_ARRAYS:
        _transfer(args.consts / name, global_root / name, link=args.link_existing_files)

    mesh_source = args.consts / "meshes"
    mesh_destination = global_root / "meshes"
    for name in _HGRIDS:
        _convert_hgrid(mesh_source / f"{name}.pickle", mesh_destination / f"{name}.npz")
    _convert_encoder_mesh(mesh_source / "healpix_8.pickle", mesh_destination / "healpix_8.npz")
    _convert_encoder_edges(
        mesh_source / "healpix_g2m_7.pickle",
        mesh_destination / "healpix_g2m_7.npz",
    )
    _convert_decoder_mesh(
        mesh_source / "healpix_m2g_7_c40.pickle",
        mesh_destination / "healpix_m2g_7_c40.npz",
    )

    _write_manifest(working)
    RuntimeBundle(working).validate(checksums=False)
    working.rename(destination)
    print(f"runtime_bundle={destination}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=_source_file, required=True)
    parser.add_argument("--consts", type=_source_directory, required=True)
    parser.add_argument("--output", type=_destination, required=True)
    parser.add_argument(
        "--link-existing-files",
        action="store_true",
        help="Hard-link fixed inputs for local validation; shared bundles should copy them",
    )
    return parser


if __name__ == "__main__":
    build_bundle(build_parser().parse_args())
