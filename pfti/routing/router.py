"""Predicate-guided model router (routing extension of PFTI).

Same mechanism as the failure store, richer record. When an agent runs a
subtask, it writes a PERFORMANCE signature:

  (task_type=extract, flavor=table, model=small) -> {ok, latency, cost}

A DIFFERENT agent facing a matching subtask reads peers' records and
routes to the model with the best observed outcome -- learning the
routing policy from teammates' trials instead of repeating them.

Granularity still applies: coarse signature (task_type only) generalizes
fast; fine signature (task_type+flavor) transfers less but more precisely.
"""
from __future__ import annotations

from dataclasses import dataclass

# reuse the core Signature machinery
from ..core.signature import Signature


def perf_signature(task_type: str, flavor: str, model: str,
                   level: str = "mid") -> Signature:
    """Signature of a (subtask, model) trial.
    coarse: task_type only ; mid: +model ; fine: +flavor."""
    preds = {("task_type", task_type)}
    if level in ("mid", "fine"):
        preds.add(("model", model))
    if level == "fine":
        preds.add(("flavor", flavor))
    return Signature(frozenset(preds))


def task_key(task_type: str, flavor: str, level: str) -> frozenset:
    """The part of the signature that describes the TASK (no model),
    used to look up 'records about this kind of task'."""
    preds = {("task_type", task_type)}
    if level == "fine":
        preds.add(("flavor", flavor))
    return frozenset(preds)


@dataclass
class PerfRecord:
    agent: str
    task_type: str
    flavor: str
    model: str
    ok: bool
    latency_ms: int
    cost: float


class PerfStore:
    def __init__(self):
        self.records: list[PerfRecord] = []

    def add(self, rec: PerfRecord):
        self.records.append(rec)


class Router:
    """Chooses a model for a subtask from peer performance records.

    Policy: among peer records whose task-key matches the current subtask,
    pick the model with the best observed success rate (ties -> cheaper).
    If no peer record exists, use `default` (cold start)."""

    def __init__(self, store: PerfStore, models: dict, level: str = "mid",
                 default: str = "small"):
        self.store = store
        self.models = models
        self.level = level
        self.default = default

    def choose(self, task_type: str, flavor: str, exclude_agent: str) -> tuple:
        key = task_key(task_type, flavor, self.level)
        # gather peer records about this kind of task
        stats = {}  # model -> [n_ok, n_total, total_cost]
        for r in self.store.records:
            if r.agent == exclude_agent:
                continue
            rkey = task_key(r.task_type, r.flavor, self.level)
            if rkey == key:
                s = stats.setdefault(r.model, [0, 0, 0.0])
                s[0] += int(r.ok)
                s[1] += 1
                s[2] += r.cost
        if not stats:
            return self.default, "cold_start"
        # explore first: any model never tried on this task gets priority
        # (optimism under uncertainty -- you can't learn a model is better
        #  until a peer has tried it at least once)
        untried = [m for m in self.models if m not in stats]
        if untried:
            # try the untried model once (prefer the cheaper unknown)
            pick = min(untried, key=lambda m: self.models[m].cost)
            return pick, "explore_untried"
        # exploit: best success rate, break ties by lower avg cost
        def score(m):
            ok, tot, cost = stats[m]
            return (ok / tot, -cost / tot)
        best = max(stats, key=score)
        return best, f"learned_from_{stats[best][1]}_peer_trials"
