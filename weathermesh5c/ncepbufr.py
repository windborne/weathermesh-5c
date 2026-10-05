from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY = "https://github.com/NOAA-EMC/NCEPLIBS-bufr.git"
VERSION = "v12.3.0"
COMMIT = "328f2294082ad1fac27525859b8422882295671a"
BUILD_TOOLS = ("cmake", "gfortran", "git", "meson", "ninja")


def _environment() -> Path:
    prefix = Path(sys.prefix).resolve()
    if prefix == Path(sys.base_prefix).resolve():
        raise RuntimeError("Refusing to install NCEPLIBS-bufr into a system Python prefix")
    return prefix


def _run(*arguments: str) -> None:
    subprocess.run(arguments, check=True)


def install() -> None:
    """Build the pinned NOAA BUFR decoder in the active virtual environment."""
    prefix = _environment()
    missing = [tool for tool in BUILD_TOOLS if shutil.which(tool) is None]
    if missing:
        raise RuntimeError(f"Missing build tools: {', '.join(missing)}")

    with tempfile.TemporaryDirectory(prefix="wm5c-ncepbufr-") as directory:
        root = Path(directory)
        source = root / "source"
        build = root / "build"
        _run(
            "git", "clone", "--quiet", "--depth", "1", "--branch", VERSION, REPOSITORY, str(source)
        )
        commit = subprocess.check_output(
            ("git", "-C", str(source), "rev-parse", "HEAD"), text=True
        ).strip()
        if commit != COMMIT:
            raise RuntimeError(f"Expected NCEPLIBS-bufr {COMMIT}, got {commit}")

        _run(
            "cmake",
            "-S",
            str(source),
            "-B",
            str(build),
            "-DENABLE_PYTHON=ON",
            "-DBUILD_TESTING=OFF",
            "-DBUILD_UTILS=OFF",
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DCMAKE_INSTALL_PREFIX={prefix}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        )
        _run("cmake", "--build", str(build), "--parallel")
        _run("cmake", "--install", str(build))

    _run(sys.executable, "-c", "import ncepbufr; print(f'ncepbufr={ncepbufr.__version__}')")


def main() -> None:
    try:
        install()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        raise SystemExit(f"NCEPLIBS-bufr installation failed: {error}") from error


if __name__ == "__main__":
    main()
