# Repository Guidelines

## Project Structure & Module Organization
`app.py` is the Gradio entrypoint and UI layer. Core workflow logic lives in `services/` (`asr_service.py`, `summary_service.py`, `embedding_service.py`, `quiz_service.py`, `pipeline_service.py`). Shared configuration, schemas, storage helpers, and model registry code live in `utils/`. Tests are in `tests/` and mirror the runtime modules with files such as `tests/test_pipeline_service.py`. Runtime inputs and outputs stay inside the repo: uploads go to `data/uploads/`, run artifacts to `artifacts/runs/`, and server logs to `artifacts/server_logs/`.

## Build, Test, and Development Commands
Install exact dependencies with `pip install -r requirements.lock.txt`. Use `requirements.txt` only when intentionally updating the dependency set and regenerating the lock file.

Run the app locally with `python app.py`. To override the bind address, use `APP_HOST=0.0.0.0 APP_PORT=7860 python app.py`.

Run the full test suite with `python -m unittest discover -s tests`. Run a focused test file with `python -m unittest tests.test_pipeline_service`.

For local UI work without auto-starting vLLM-backed services, use `AUTO_START_MODEL_SERVERS=0 python app.py`.

## Coding Style & Naming Conventions
Use 4-space indentation, type hints, and `from __future__ import annotations` for new Python modules to match the existing codebase. Prefer small service methods with explicit data flow through `utils.schemas` models. Use `snake_case` for functions, variables, and modules; use `PascalCase` for classes and dataclasses. Keep user-facing configuration in `utils/config.py` or `model_info.json` instead of scattering constants across services.

## Testing Guidelines
This repository uses `unittest` with file names in the `test_*.py` pattern. Add tests beside the nearest affected module and prefer lightweight fakes over real model calls, following the existing pipeline and app tests. Cover both happy paths and failure handling for pipeline steps, schema validation, and UI rendering helpers.

## Commit & Pull Request Guidelines
Recent history uses short, imperative commit subjects such as `Add model selection presets` and `Improve quiz UI and variant generation`. Keep commits focused and descriptive; avoid mixing refactors with behavior changes. PRs should explain the user-visible impact, list any required environment variables or model/server assumptions, and include screenshots when `app.py` changes the Gradio UI.

## Configuration Tips
Model selection is driven by `model_info.json`; update that file before hard-coding model changes. Keep generated artifacts out of source control unless they are required fixtures for tests.
