from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass
class RandomPolicy:
    seed: int = 7

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    def propose_diff(self, state: dict) -> str:
        lr = self._rng.choice(["2e-3", "2.5e-3", "2.8e-3", "3e-3"])
        return f"diff --git a/train.py b/train.py\n+ learning_rate = {lr}"


@dataclass
class GreedyLLMPolicy:
    """Stub greedy policy to be wired to real LLM backend."""

    def propose_diff(self, state: dict) -> str:
        best = state.get("best_score")
        if best is None or best > 1.3:
            return "diff --git a/train.py b/train.py\n+ use_qk_norm = True"
        return "diff --git a/train.py b/train.py\n+ grad_clip = 0.8"
