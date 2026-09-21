"""Mock LLM models with defined accuracy/latency/cost profiles (routing ext).

A "model" is a behavior profile: for each subtask TYPE it has a success
probability, a latency, and a per-call cost. Because these profiles are
DEFINED, we get a ground-truth oracle: the best model for a given task
type is knowable without running anything -- exactly like the failure
oracle in the base testbed. This is what makes routing quality measurable.

Subtasks are GRADEABLE: each carries a gold answer, so success = (answer
== gold), no LLM judge needed.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

SUBTASK_TYPES = ("extract", "classify", "summarize")


@dataclass(frozen=True)
class MockModel:
    name: str
    # success prob per subtask type
    acc: dict
    latency_ms: int
    cost: float          # per call, arbitrary units

    def run(self, task_type: str, item: dict, rng: random.Random):
        """Return (answer, latency_ms, cost, ok). Deterministic given rng."""
        p = self.acc.get(task_type, 0.5)
        ok = rng.random() < p
        answer = item["gold"] if ok else item.get("wrong", "__WRONG__")
        return answer, self.latency_ms, self.cost, ok


# Two models with a clear speed/accuracy/cost tradeoff.
# small: cheap+fast, good at easy (classify), weak at hard (extract)
# large: expensive+slow, strong everywhere
MODELS = {
    # small wins on classify (easy) and is far cheaper; large wins on the
    # hard tasks. So the BEST model differs by task type -> routing is
    # non-trivial: a good team routes easy tasks to small, hard to large.
    "small": MockModel("small",
                       acc={"extract": 0.25, "classify": 0.95, "summarize": 0.5},
                       latency_ms=200, cost=1.0),
    "large": MockModel("large",
                       acc={"extract": 0.9, "classify": 0.8, "summarize": 0.9},
                       latency_ms=1200, cost=6.0),
}


def best_model_oracle(task_type: str) -> str:
    """Ground truth: which model has the highest success prob for this type."""
    return max(MODELS, key=lambda m: MODELS[m].acc.get(task_type, 0))


def make_batch(n: int, seed: int = 0) -> list:
    """A gradeable batch of subtasks. Each item has type + gold answer."""
    rng = random.Random(seed)
    batch = []
    for i in range(n):
        t = rng.choice(SUBTASK_TYPES)
        # a few 'input flavors' -> lets granularity matter later
        flavor = rng.choice(["plain", "table", "scanned"])
        batch.append({"id": i, "task_type": t, "flavor": flavor,
                      "gold": f"ans{i}", "wrong": f"bad{i}"})
    return batch
