from __future__ import annotations

import hashlib
import json
import os
import shutil
import string
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from uuid import uuid4

import numpy as np

BUNDLE_SCHEMA_VERSION = 3
CHECKPOINT_FILE = PurePosixPath("weights/weather-mesh-5c-aida.pt")
GLOBAL_ROOT = PurePosixPath("assets/global")
_FICLONE = 0x40049409

_GLOBAL_FILES = (
    "land_mask.npy",
    "soil_type.npy",
    "topography.npy",
    "meshes/hgrid_nu32_D6_KL21_KD5_B16.npz",
    "meshes/hgrid_da_442adbe5eb4a688c.npz",
    "meshes/hgrid_da_791b788601d0fc08.npz",
    "meshes/healpix_8.npz",
    "meshes/healpix_g2m_7.npz",
    "meshes/healpix_m2g_7_c40.npz",
)


def sha256_file(path: Path, chunk_bytes: int = 16 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _replace_file(source: Path, target: Path, *, hardlink: bool = False) -> None:
    source = source.resolve()
    if source == target.resolve(strict=False):
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{uuid4().hex}.tmp")
    try:
        linked = False
        if hardlink:
            try:
                os.link(source, temporary)
                linked = True
            except OSError:
                pass
        if not linked and sys.platform == "linux":
            import fcntl

            try:
                with source.open("rb") as source_stream, temporary.open("xb") as target_stream:
                    fcntl.ioctl(target_stream.fileno(), _FICLONE, source_stream.fileno())
            except OSError:
                temporary.unlink(missing_ok=True)
                shutil.copyfile(source, temporary)
        elif not linked:
            shutil.copyfile(source, temporary)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)


def global_bundle_files() -> tuple[PurePosixPath, ...]:
    return tuple(GLOBAL_ROOT / name for name in _GLOBAL_FILES)


@dataclass(frozen=True)
class ManifestFile:
    path: PurePosixPath
    size_bytes: int
    sha256: str
    role: str

    @classmethod
    def parse(cls, value: object) -> ManifestFile:
        if not isinstance(value, dict):
            raise TypeError("Each runtime-bundle file entry must be an object")
        try:
            path = PurePosixPath(value["path"])
            size_bytes = value["size_bytes"]
            digest = value["sha256"]
            role = value["role"]
        except KeyError as error:
            raise ValueError(f"Runtime-bundle file entry is missing {error.args[0]!r}") from error
        if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
            raise ValueError(f"Unsafe runtime-bundle path: {path}")
        if not isinstance(size_bytes, int) or size_bytes < 0:
            raise TypeError(f"Invalid size for runtime-bundle file {path}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in string.hexdigits for character in digest)
        ):
            raise TypeError(f"Invalid SHA-256 for runtime-bundle file {path}")
        if role not in {"checkpoint", "global"}:
            raise ValueError(f"Invalid role for runtime-bundle file {path}: {role!r}")
        return cls(path, size_bytes, digest.lower(), role)


class RuntimeBundle:
    """A complete, versioned set of WeatherMesh-5c runtime inputs."""

    def __init__(self, root: str | Path, *, resolve: Callable[[str], str | Path] | None = None):
        self.root = Path(root).expanduser().resolve()
        self._resolve = resolve
        manifest_path = self.path("manifest.json")
        with manifest_path.open() as stream:
            manifest = json.load(stream)
        if (
            not isinstance(manifest, dict)
            or manifest.get("schema_version") != BUNDLE_SCHEMA_VERSION
        ):
            raise ValueError(
                f"Runtime bundle must use schema version {BUNDLE_SCHEMA_VERSION}: {manifest_path}"
            )
        if manifest.get("model") != "WeatherMesh-5c with AIDA":
            raise ValueError(f"Runtime bundle identifies an unexpected model: {manifest_path}")
        raw_files = manifest.get("files")
        if not isinstance(raw_files, list):
            raise TypeError(f"Runtime-bundle manifest needs a file list: {manifest_path}")
        files = tuple(ManifestFile.parse(value) for value in raw_files)
        paths = [entry.path for entry in files]
        if len(paths) != len(set(paths)):
            raise ValueError(f"Runtime-bundle manifest contains duplicate paths: {manifest_path}")
        self._files = {entry.path: entry for entry in files}
        global_files = set(global_bundle_files())
        expected_roles = {
            CHECKPOINT_FILE: "checkpoint",
            **dict.fromkeys(global_files, "global"),
        }
        unexpected = self._files.keys() - expected_roles.keys()
        if unexpected:
            formatted = ", ".join(str(path) for path in sorted(unexpected))
            raise ValueError(f"Runtime-bundle manifest contains unexpected files: {formatted}")
        for path, entry in self._files.items():
            if entry.role != expected_roles[path]:
                raise ValueError(
                    f"Runtime-bundle file {path} has role {entry.role!r}; "
                    f"expected {expected_roles[path]!r}"
                )

    @property
    def checkpoint(self) -> Path:
        return self.path(CHECKPOINT_FILE)

    @property
    def global_root(self) -> Path:
        return self.root.joinpath(*GLOBAL_ROOT.parts)

    def path(self, relative: str | PurePosixPath) -> Path:
        if self._resolve is not None:
            return Path(self._resolve(str(relative)))
        return self.root.joinpath(*PurePosixPath(relative).parts)

    def global_path(self, relative: str) -> Path:
        """Resolve a global asset without requiring a second directory tree."""
        return self.path(GLOBAL_ROOT / relative)

    def required_files(self) -> tuple[PurePosixPath, ...]:
        """Return every required runtime file."""

        required = {CHECKPOINT_FILE, *global_bundle_files()}
        missing_entries = required - self._files.keys()
        if missing_entries:
            formatted = "\n".join(f"- {path}" for path in sorted(missing_entries))
            raise FileNotFoundError(f"Runtime-bundle manifest is incomplete:\n{formatted}")
        return tuple(sorted(required))

    @classmethod
    def materialize(
        cls,
        resolve: Callable[[str], str | Path],
        destination: str | Path,
        *,
        checksums: bool = True,
        hardlinks: bool = False,
    ) -> RuntimeBundle:
        """Materialize a runtime bundle from a remote package resolver.

        The resolver must return a local path for a requested package-relative file.
        Existing valid files are reused, and each new file is installed atomically.
        Same-filesystem hard links can avoid a second copy in a package cache.
        """

        root = Path(destination).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        _replace_file(
            Path(resolve("manifest.json")),
            root / "manifest.json",
            hardlink=hardlinks,
        )
        bundle = cls(root)

        for relative in bundle.required_files():
            target = bundle.path(relative)
            if bundle._file_is_valid(relative, checksums=checksums):
                continue
            _replace_file(
                Path(resolve(relative.as_posix())),
                target,
                hardlink=hardlinks,
            )
            bundle._validate_file(relative, checksums=checksums)

        return bundle

    def _file_is_valid(self, relative: PurePosixPath, *, checksums: bool) -> bool:
        entry = self._files[relative]
        path = self.path(relative)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != entry.size_bytes:
            return False
        return not checksums or sha256_file(path) == entry.sha256

    def _validate_file(self, relative: PurePosixPath, *, checksums: bool) -> None:
        entry = self._files[relative]
        path = self.path(relative)
        if path.is_symlink() or not path.is_file():
            raise FileNotFoundError(f"Runtime-bundle file is missing or symlinked: {path}")
        size = path.stat().st_size
        if size != entry.size_bytes:
            raise ValueError(
                f"Runtime-bundle file {relative} has {size} bytes; expected {entry.size_bytes}"
            )
        if checksums:
            digest = sha256_file(path)
            if digest != entry.sha256:
                raise ValueError(
                    f"Runtime-bundle file {relative} has SHA-256 {digest}; expected {entry.sha256}"
                )

    def validate(self, *, checksums: bool = True) -> None:
        for relative in self.required_files():
            self._validate_file(relative, checksums=checksums)


def load_array(path: Path, *, mmap_mode: str | None = None) -> np.ndarray:
    return np.load(path, mmap_mode=mmap_mode, allow_pickle=False)


def load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}
