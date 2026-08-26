# Repository Guidelines

## Project Structure & Module Organization

Environments live in `pufferlib/`, with INI files in `pufferlib/config/`. Extensions are in `pufferlib/extensions/`, tests in `tests/`, helpers in `scripts/`, and docs in `docs/src/`.

## Simulator Variants

- **Ocean (`puffer_drive`)** is the baseline: fixed `-0.5` collision/offroad rewards, no condition inputs, continued simulation after those events, and goal respawning.
- **Pacific (`sut_drive`, `cond_drive`)** observes per-agent collision cost, offroad cost, and lane width. `sut_drive` fixes them at `0.5`, `0.5`, and `4`. `cond_drive` samples both costs from `0–3`, continues after collisions/offroad, removes agents at goals, and penalizes overspeed. Random mode also samples lane width from the implementation default `1–5`; evaluation fixes values at `2`, `2`, and `4`. `selfplay_drive` is retired; add no new references.
- **Atlantic (`adv_drive`)** observes self-fault cost, counterparty-fault preference `f`, offroad cost, and lane width. It rewards counterparty-caused collisions, penalizes self-fault, scales goal reward by `1-f`, stops after collision/offroad, and removes agents at goals.

Run `puffer train <env_name>`; never mix checkpoints across families.

## Build, Test, and Development Commands

- `uv venv && source .venv/bin/activate && uv pip install -e .`: create an editable environment.
- `python setup.py build_ext --inplace --force`: rebuild native extensions.
- `pytest tests/test_atlantic_fault.py`: run a focused test; use `pytest tests/` when optional dependencies/data are available.
- `bash tests/ini_parser/build_n_test.sh`: build and run C parser tests.
- `pre-commit run --all-files`: run formatting and hygiene checks.

## Coding Style & Naming Conventions

Use four spaces, `snake_case` for Python names, `PascalCase` for classes, and `UPPER_CASE` for constants. Ruff targets Python 3.10 and 120 columns; clang-format handles native code. Keep INI, Python, and C configuration keys aligned.

## Testing Guidelines

Use pytest or `unittest`; name tests `test_*`. Add regressions and report skipped data-, GPU-, or platform-dependent checks. Run training/performance tests for policy or simulator hot-path changes.

## Commit & Pull Request Guidelines

Keep commits brief and scoped. PRs must explain effects, link issues, list validation and required hardware/data, and include visuals for renderer or documentation changes.

## Agent-Specific Instructions

**No fallbacks.** Do not write defensive code that hides errors or invalid states. Avoid fallbacks, silent recovery, and default-value access such as `dict.get(key, default)`. Do not use `try/except` or defensive `if/else` to mask unexpected failures; let errors propagate and fail loudly. Handle only genuinely expected business conditions explicitly.

## Generated Files & Data

Do not commit environments, builds, checkpoints, datasets, videos, or generated binaries; `.gitignore` covers common outputs.
