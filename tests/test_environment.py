from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from weathermesh5c.ncepbufr import install

ROOT = Path(__file__).parents[1]
LAUNCHER = ROOT / "scripts/wm5c-uv.sh"
INSTALLER = ROOT / "scripts/install-ncepbufr.sh"


def _fake_uv(directory: Path) -> Path:
    executable = directory / "uv"
    executable.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "print(json.dumps({\n"
        "    'arguments': sys.argv[1:],\n"
        "    'cwd': os.getcwd(),\n"
        "    'path': os.environ.get('PATH'),\n"
        "    'python_home': os.environ.get('PYTHONHOME'),\n"
        "    'python_path': os.environ.get('PYTHONPATH'),\n"
        "    'project_environment': os.environ.get('UV_PROJECT_ENVIRONMENT'),\n"
        "    'virtual_environment': os.environ.get('VIRTUAL_ENV'),\n"
        "}))\n"
    )
    executable.chmod(0o755)
    return executable


@pytest.mark.parametrize(
    "inherited_variables",
    (
        ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT"),
        ("VIRTUAL_ENV",),
        ("UV_PROJECT_ENVIRONMENT",),
        (),
    ),
)
def test_launcher_forces_the_repository_environment_and_frozen_lock(
    tmp_path, inherited_variables
) -> None:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    _fake_uv(bin_directory)
    environment = os.environ.copy()
    inherited_environments = {
        "VIRTUAL_ENV": tmp_path / "active",
        "UV_PROJECT_ENVIRONMENT": tmp_path / "shared",
    }
    for variable, directory in inherited_environments.items():
        environment.pop(variable, None)
        if variable in inherited_variables:
            environment[variable] = str(directory)
    environment["PATH"] = os.pathsep.join(
        (
            str(bin_directory),
            *(str(path / "bin") for path in inherited_environments.values()),
            environment["PATH"],
        )
    )
    environment["PYTHONHOME"] = str(tmp_path / "python-home")
    environment["PYTHONPATH"] = str(tmp_path / "python-path")

    result = subprocess.run(
        [LAUNCHER, "run", "python", "-V"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    report = json.loads(result.stdout)

    expected = str(ROOT / ".venv")
    assert report["arguments"] == ["run", "--frozen", "python", "-V"]
    assert report["cwd"] == str(ROOT)
    assert report["project_environment"] == expected
    assert report["virtual_environment"] == expected
    assert report["python_home"] is None
    assert report["python_path"] is None
    assert report["path"].split(os.pathsep)[0] == str(ROOT / ".venv/bin")
    for variable, directory in inherited_environments.items():
        assert (str(directory / "bin") in report["path"].split(os.pathsep)) == (
            variable not in inherited_variables
        )

    result = subprocess.run(
        [LAUNCHER, "build"],
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    assert json.loads(result.stdout)["arguments"] == ["build", "--no-build-isolation"]


def test_launcher_rejects_a_symlinked_environment(tmp_path) -> None:
    root = tmp_path / "checkout"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    launcher = scripts / LAUNCHER.name
    shutil.copyfile(LAUNCHER, launcher)
    launcher.chmod(0o755)
    (root / ".venv").symlink_to(tmp_path / "external", target_is_directory=True)

    result = subprocess.run(
        [launcher, "sync"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "symlinked" in result.stderr


def test_launcher_rejects_an_explicit_environment_override(tmp_path) -> None:
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    _fake_uv(bin_directory)
    environment = os.environ.copy()
    environment["PATH"] = f"{bin_directory}{os.pathsep}{environment['PATH']}"

    result = subprocess.run(
        [LAUNCHER, "run", "--active", "python", "-V"],
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.returncode == 2
    assert "environment override" in result.stderr


def test_ncepbufr_installer_requires_the_repository_environment(tmp_path) -> None:
    scripts = tmp_path / "checkout/scripts"
    scripts.mkdir(parents=True)
    installer = scripts / INSTALLER.name
    shutil.copyfile(INSTALLER, installer)
    installer.chmod(0o755)

    result = subprocess.run(
        [installer],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "repository-local environment" in result.stderr


def test_ncepbufr_installer_rejects_system_python(monkeypatch) -> None:
    monkeypatch.setattr(sys, "prefix", sys.base_prefix)

    with pytest.raises(RuntimeError, match="system Python prefix"):
        install()
