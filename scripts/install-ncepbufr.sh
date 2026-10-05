#!/usr/bin/env bash
set -euo pipefail

wm5c_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
wm5c_environment="$wm5c_root/.venv"
wm5c_python="$wm5c_environment/bin/python"

if [[ -L "$wm5c_environment" || ! -x "$wm5c_python" ]]; then
  echo "Create the repository-local environment with ./scripts/wm5c-uv.sh sync first." >&2
  exit 2
fi

wm5c_prefix="$($wm5c_python -c 'from pathlib import Path; import sys; print(Path(sys.prefix).resolve())')"

if [[ "$wm5c_prefix" != "$wm5c_environment" ]]; then
  echo "Refusing to install NCEPLIBS-bufr outside $wm5c_environment." >&2
  exit 2
fi

exec "$wm5c_python" -m weathermesh5c.ncepbufr
