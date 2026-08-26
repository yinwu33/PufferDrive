# Repository Guidelines

## Project Structure & Module Organization

Environments live in `pufferlib/`, with INI files in `pufferlib/config/`. Extensions are in `pufferlib/extensions/`, tests in `tests/`, helpers in `scripts/`, and docs in `docs/src/`.

## Simulator Variants

- **Ocean (`puffer_drive`)** is the baseline: fixed `-0.5` collision/offroad rewards, no condition inputs, continued simulation after those events, and goal respawning.
- **Pacific (`sut_drive`, `cond_drive`)** observes per-agent collision cost, offroad cost, and lane width. `sut_drive` fixes them at `0.5`, `0.5`, and `4`. `cond_drive` samples both costs from `0–3`, continues after collisions/offroad, removes agents at goals, and penalizes overspeed. Random mode also with fixed lane width `4`; evaluation fixes values at `2`, `2`, and `4`. `selfplay_drive` is retired; add no new references.
- **Atlantic (`adv_drive`)** observes self-fault cost, counterparty-fault preference `f`, offroad cost, and lane width. It rewards counterparty-caused collisions, penalizes self-fault, scales goal reward by `1-f`, stops after collision/offroad, and removes agents at goals.

Run `puffer train <env_name>`; never mix checkpoints across families.

## Build, Test, and Development Commands

- `uv venv && source .venv/bin/activate && uv pip install -e .`: create an editable environment.
- `python setup.py build_ext --inplace --force`: rebuild native extensions.


## Testing Guidelines

Use pytest or `unittest`; name tests `test_*`. Add regressions and report skipped data-, GPU-, or platform-dependent checks. Run training/performance tests for policy or simulator hot-path changes.

## Commit & Pull Request Guidelines

Keep commits brief and scoped. PRs must explain effects, link issues, list validation and required hardware/data, and include visuals for renderer or documentation changes.

## Conventions

**No fallbacks.** Do not write defensive code that hides errors or invalid states. Avoid fallbacks, silent recovery, and default-value access such as `dict.get(key, default)`. Do not use `try/except` or defensive `if/else` to mask unexpected failures; let errors propagate and fail loudly. Handle only genuinely expected business conditions explicitly.

Do not commit environments, builds, checkpoints, datasets, videos, or generated binaries; `.gitignore` covers common outputs.


## Evaluations

Researching related files saved in `research/`. Evaluation results in `research/results/`.

Results for `cond_drive` are saved in [eval_cond_drive.md](research/results/eval_cond_drive.md).