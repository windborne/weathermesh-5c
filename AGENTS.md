# Project instructions

- Run project `uv` commands only through `./scripts/wm5c-uv.sh`.
- Never install or synchronize this project into a shared or external Python environment.
- The launcher owns the repository-local `.venv`. Sync and run commands use the
  committed lock. Build commands use the synchronized environment without installing
  isolated build dependencies.
- Use a separate temporary environment for clean-install validation. Set and validate
  `UV_PROJECT_ENVIRONMENT` explicitly before every such `uv` command.
- Run `./scripts/wm5c-uv.sh run --no-sync ruff check` at the repository root before
  each push.
