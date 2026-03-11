# AutoResearch-RL Scaffold

Minimal production-oriented scaffold for autonomous training-script research loops.

## Goals
- Deterministic, auditable experiment loop
- Safe code-edit sandboxing
- Fixed-budget comparable evaluations
- Trace-first observability

## Layout
- `src/autoresearch_rl/controller` – run orchestration and state machine
- `src/autoresearch_rl/sandbox` – patch validation + constrained execution
- `src/autoresearch_rl/policy` – proposal interface (LLM/RL plug points)
- `src/autoresearch_rl/eval` – scoring, early-stop forecasting hooks
- `src/autoresearch_rl/telemetry` – trace/event emitters
- `configs/` – runtime and benchmark configs
- `scripts/` – CLI entry helpers
- `tests/` – unit/integration tests

## Quickstart
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
cp configs/example.yaml configs/local.yaml
python scripts/run_once.py --config configs/local.yaml
```

## Safety defaults
- mutable scope limited to target file list
- no network in runner (expected to be enforced by runtime/container)
- hard wall-clock timeout per run
- every run emits structured JSONL trace
