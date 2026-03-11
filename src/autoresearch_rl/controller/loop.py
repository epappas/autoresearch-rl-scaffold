from dataclasses import dataclass


@dataclass
class LoopResult:
    best_score: float
    iterations: int


def run_loop(max_iterations: int = 1) -> LoopResult:
    """Placeholder control loop for scaffold bootstrapping."""
    best = float("inf")
    for _ in range(max_iterations):
        pass
    return LoopResult(best_score=best, iterations=max_iterations)
