# Repository Guidelines

## Project Structure & Module Organization

Core Python APIs live in `pufferlib/`. Simulator variants under `pufferlib/ocean/`, `pufferlib/pacific/`, and `pufferlib/atlantic/` combine Python wrappers, C bindings, and simulator sources. Shared C++/CUDA extensions are in `pufferlib/extensions/`; INI files and map resources are under `pufferlib/config/` and `pufferlib/resources/`. Put Python tests in `tests/test_*.py` and native parser fixtures in `tests/ini_parser/`. Examples, helpers, and mdBook sources belong in `examples/`, `scripts/`, and `docs/src/`.

## Build, Test, and Development Commands

- `uv venv && source .venv/bin/activate` creates the recommended environment.
- `uv pip install -e .` installs PufferDrive in editable mode.
- `python setup.py build_ext --inplace --force` rebuilds C/C++/CUDA extensions against the active interpreter.
- `pytest tests/test_atlantic_fault.py` runs a focused file; use `pytest tests/` only with required optional dependencies and datasets.
- `bash tests/ini_parser/build_n_test.sh` configures, builds, and runs the C parser tests with CMake/CTest.
- `pre-commit run --all-files` applies repository formatting and hygiene checks.
- `puffer train puffer_drive` performs a training smoke run after compilation.

## Coding Style & Naming Conventions

Use four-space indentation and `snake_case` for Python functions, modules, and variables; use `PascalCase` for classes and `UPPER_CASE` for constants. Ruff targets Python 3.10, formats to 120 columns, and sorts imports with two blank lines after them. Native C/C++/CUDA files are formatted by clang-format. Keep configuration keys consistent between INI, Python, and C layers when a setting crosses those boundaries.

## Testing Guidelines

Tests use pytest and `unittest`, with files and test functions named `test_*`. Add focused regression coverage beside the affected subsystem. There is no declared coverage threshold. Note skipped dataset-, GPU-, or platform-dependent checks in the pull request. Run heavier training and performance checks when changing rollout, policy, or simulator hot paths.

## Commit & Pull Request Guidelines

Recent commits favor brief, imperative or topic-style subjects such as `add sequential map sample`. Keep each commit scoped. Pull requests should explain motivation and user-visible effects, link issues, list commands run, and identify required data or hardware. Include screenshots or recordings for renderer, docs, or visualization changes, and target the active development branch used by CI.

**No fallbacks.** Do not write defensive code that hides errors or invalid states. Avoid fallbacks, silent recovery, and default-value access such as `dict.get(key, default)`. Do not use `try/except` or defensive `if/else` to mask unexpected failures; let errors propagate and fail loudly. Handle only genuinely expected business conditions explicitly.

## Generated Files & Data

Do not commit virtual environments, build products, checkpoints, downloaded datasets, videos, or generated binaries. The root `.gitignore` already excludes common outputs such as `build/`, `experiments/`, `data/`, `*.so`, and `*.mp4`.
