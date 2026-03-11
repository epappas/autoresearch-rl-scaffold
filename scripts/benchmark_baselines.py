from __future__ import annotations

import typer

from autoresearch_rl.eval.metrics import parse_metrics
from autoresearch_rl.policy.baselines import GreedyLLMPolicy, RandomPolicy
from autoresearch_rl.sandbox.runner import run_trial

app = typer.Typer()


def _run(policy, iterations: int) -> dict:
    best = float("inf")
    for i in range(iterations):
        diff = policy.propose_diff({"iter": i, "best_score": best if best < 999 else None})
        r = run_trial(diff=diff, timeout_s=10, command=["bash", "-lc", "echo 'step=1 loss=2.1'; echo 'val_bpb=1.22'"])
        parsed = parse_metrics(r.stdout)
        score = parsed.val_bpb if parsed.val_bpb is not None else 999.0
        best = min(best, score)
    return {"best_val_bpb": best, "iterations": iterations}


@app.command()
def main(iterations: int = 5) -> None:
    rp = _run(RandomPolicy(), iterations)
    gp = _run(GreedyLLMPolicy(), iterations)
    print({"random": rp, "greedy": gp})


if __name__ == "__main__":
    app()
