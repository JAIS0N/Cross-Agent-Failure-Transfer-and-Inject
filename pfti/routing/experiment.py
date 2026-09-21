"""Routing experiment (routing extension).

Three conditions over the SAME gradeable batch:
  always-small  : cheap, low accuracy  (baseline floor)
  always-large  : accurate, expensive  (baseline ceiling)
  PFTI-routed   : starts naive, learns the routing policy from peer
                  performance records as the shared store fills

Metrics per condition: success rate (vs gold), total cost, total latency,
and (routed only) how often it matched the oracle's best model.

The headline: PFTI-routed reaches accuracy near always-large at cost near
always-small -- the 'good corner' of the cost/accuracy plot.
"""
from __future__ import annotations

import random

from .models import MODELS, make_batch, best_model_oracle
from .router import PerfStore, PerfRecord, Router


def _agents_round_robin(n_items):
    # 3 agents take turns on the batch, so records from one are seen by others
    agents = ["A", "B", "C"]
    return [agents[i % 3] for i in range(n_items)]


def run_condition(batch, policy: str, level: str = "mid", seed: int = 0):
    rng = random.Random(seed)
    store = PerfStore()
    router = Router(store, MODELS, level=level, default="small")
    owners = _agents_round_robin(len(batch))

    ok_n = cost = latency = optimal_n = 0
    trace = []
    for item, agent in zip(batch, owners):
        tt, fl = item["task_type"], item["flavor"]
        if policy == "small":
            model, why = "small", "fixed"
        elif policy == "large":
            model, why = "large", "fixed"
        else:  # routed
            model, why = router.choose(tt, fl, exclude_agent=agent)

        ans, lat, c, ran_ok = MODELS[model].run(tt, item, rng)
        graded_ok = (ans == item["gold"])
        ok_n += int(graded_ok)
        cost += c
        latency += lat
        if model == best_model_oracle(tt):
            optimal_n += 1
        # record the trial for peers to learn from
        store.add(PerfRecord(agent, tt, fl, model, graded_ok, lat, c))
        trace.append((agent, tt, fl, model, why, graded_ok))

    n = len(batch)
    return {"policy": policy, "level": level,
            "success_rate": ok_n / n, "total_cost": cost,
            "total_latency_s": latency / 1000,
            "optimal_route_rate": optimal_n / n, "trace": trace, "n": n}


def run_all(n_items=30, seed=0, level="mid"):
    batch = make_batch(n_items, seed=seed)
    return {p: run_condition(batch, p, level=level, seed=seed)
            for p in ("small", "large", "routed")}
