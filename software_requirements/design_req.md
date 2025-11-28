# Testing and pytest – Implementation Guide

This document describes the pytest requirements and recommended project/testing structure for the project. It also specifies packaging and versioning conventions: Python >= 3.10, using a `pyproject.toml`-based build, and storing the package version in `src/brainmaze_sigcoreg/_version.py` (exposed via the package and read dynamically by the build metadata).

## High-level rules

- Python runtime: `>=3.10`.
- Packaging: use `pyproject.toml` + `setuptools` (PEP 517/621). The `pyproject.toml` declares a dynamic `version` attribute read from `brainmaze_sigcoreg._version.__version__`.
- The canonical package version is stored in `src/brainmaze_sigcoreg/_version.py` as `__version__ = "x.y.z"`.

## pytest requirements (dev)

Dev dependencies are declared in `pyproject.toml` under the `dev` optional-dependencies group. To install them in editable mode, run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[dev]'
```

Recommended minimal dev requirements (declared in `pyproject.toml`):

- pytest >= 7.0q
- pytest-cov
- pytest-mock (or `mock` / `unittest.mock` as appropriate)

## pytest configuration

A minimal `pytest.ini` is provided in the repo. Key points:

- `testpaths = tests`
- `python_files = test_*.py`
- sensible defaults for marks and minimum pytest version

You can alternatively move pytest configuration into `pyproject.toml` under `[tool.pytest.ini_options]` (this template already includes it).

## How to run tests locally

1. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

2. Install package and dev dependencies

```bash
pip install -e '.[dev]'
```

3. Run pytest

```bash
pytest -q
# or with coverage
pytest --cov=brainmaze_sigcoreg --cov-report=term-missing
```

## Versioning: `_version.py` and `pyproject.toml`

Store the package version in `src/brainmaze_sigcoreg/_version.py` as:

```python
__version__ = "0.0.0"
```

The `pyproject.toml` in this template is configured to use `setuptools` dynamic metadata to read the `__version__` attribute at build time. This keeps the package version in a single, importable location (useful at runtime and for tests).

Notes:
- Keep `_version.py` minimal and free of side effects to ensure it is import-safe in build environments.
- Update `_version.py` when making releases; CI/CD can replace or tag versions automatically.

## Test structure and conventions

- tests should live under `tests/` and follow `test_*.py` naming.
- Use `tests/conftest.py` for fixtures that are shared across tests (e.g., mocked `MefReader` instances).
- Unit tests should be fast and deterministic; mock heavy I/O and large data reads.
- Add integration tests separately (e.g., `tests/integration/`) and mark them with `@pytest.mark.integration` or `@pytest.mark.slow`.

## CI (GitHub Actions) example

A template workflow `.github/workflows/pytest.yml` is provided. It:

- Runs on push and pull_request
- Uses Python 3.10
- Installs dev extras via `pip install -e '.[dev]'`
- Runs `pytest` (with optional coverage)

Adjust the workflow to your preferred Python matrix or add caching for pip.

## Example test

`tests/test_version.py` in this template verifies basic import and that `__version__` is defined and non-empty. This is a helpful smoke test to ensure packaging and imports work in CI.

## Troubleshooting

- If CI fails due to missing system dependencies (rare for pure-Python), add an apt-get step in the workflow to install them.
- If `pyproject.toml` fails to pick up the `__version__` attr during build, ensure `src/brainmaze_sigcoreg/_version.py` is import-safe (no heavy imports at top-level) and that `setuptools` in the build-system supports dynamic attr resolution.

---

## Non-Functional Requirements (cross-platform)

- **Cross-platform Compatibility & Architecture Independence**: The project must be portable and runnable across major operating systems and CPU architectures. Specifically:
  - Support Linux, macOS, and Windows as target platforms.
  - Support common CPU architectures where Python is available (for example x86_64 and ARM64).
  - Avoid OS-specific system calls, shell-dependent behavior, or hard-coded path separators in the library code. Use `pathlib`, `os`, and other cross-platform abstractions.
  - If native extensions are required, provide pre-built wheels for supported platforms or keep native code optional and fall back to pure-Python implementations where practical.
  - The CI pipeline should exercise at least one runner per supported OS (see `.github/workflows/pytest.yml`) and we should add ARM/alternative-architecture runners where available to validate architecture compatibility.

---

End of testing document.
