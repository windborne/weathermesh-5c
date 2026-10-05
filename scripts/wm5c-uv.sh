#!/usr/bin/env bash
set -euo pipefail

wm5c_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
wm5c_environment="$wm5c_root/.venv"
wm5c_uv="$(command -v uv)"

if [[ -L "$wm5c_environment" ]]; then
  echo "Refusing to use a symlinked WeatherMesh-5c environment: $wm5c_environment" >&2
  exit 2
fi
if [[ -e "$wm5c_environment" && ! -d "$wm5c_environment" ]]; then
  echo "WeatherMesh-5c environment path is not a directory: $wm5c_environment" >&2
  exit 2
fi
if [[ $# -eq 0 ]]; then
  echo "Usage: $0 {sync|run|build|lock} [arguments...]" >&2
  exit 2
fi

wm5c_command="$1"
shift
case "$wm5c_command" in
  sync|run)
    wm5c_arguments=("$wm5c_command" --frozen "$@")
    ;;
  build)
    wm5c_arguments=(build --no-build-isolation "$@")
    ;;
  lock)
    wm5c_arguments=(
      lock
      --index https://download.pytorch.org/whl/cu128
      --default-index https://pypi.org/simple/
      --index-strategy unsafe-best-match
      "$@"
    )
    ;;
  *)
    echo "Unsupported WeatherMesh-5c uv command: $wm5c_command" >&2
    exit 2
    ;;
esac

for wm5c_argument in "${wm5c_arguments[@]}"; do
  case "$wm5c_argument" in
    --active|--system|--no-project|--project|--project=*|--directory|--directory=*)
      echo "Refusing uv environment override: $wm5c_argument" >&2
      exit 2
      ;;
  esac
done

wm5c_clean_path=""
IFS=: read -r -a wm5c_path_entries <<< "${PATH:-}"
for wm5c_path_entry in "${wm5c_path_entries[@]}"; do
  for wm5c_inherited_environment in "${VIRTUAL_ENV:-}" "${UV_PROJECT_ENVIRONMENT:-}"; do
    if [[ -n "$wm5c_inherited_environment" &&
          "${wm5c_path_entry%/}" == "${wm5c_inherited_environment%/}/bin" ]]; then
      continue 2
    fi
  done
  wm5c_clean_path+="${wm5c_clean_path:+:}$wm5c_path_entry"
done

unset PYTHONHOME PYTHONPATH UV_PROJECT UV_WORKING_DIR UV_NO_PROJECT
export PATH="$wm5c_environment/bin${wm5c_clean_path:+:$wm5c_clean_path}"
export UV_PROJECT_ENVIRONMENT="$wm5c_environment"
export VIRTUAL_ENV="$wm5c_environment"
cd -- "$wm5c_root"
exec "$wm5c_uv" "${wm5c_arguments[@]}"
