# Architecture (v0 scaffold)

## Control-plane
- Proposal policy (LLM or RL)
- Loop state (best config, history window, penalties)
- Gating (compile check, policy constraints)

## Data-plane
- Sandboxed trial runner
- Metric collector
- Early-stop evaluator

## Observability
- Structured event stream (JSONL)
- Run artifacts with immutable IDs
- Replayable run manifest

## Next steps
1. Implement patch validator (AST + forbidden APIs)
2. Add subprocess runner with cgroup/timeouts
3. Add metric stream parser + early-stop forecaster
4. Add baseline policies (random, greedy LLM)
5. Integrate llmtrace spans/events
