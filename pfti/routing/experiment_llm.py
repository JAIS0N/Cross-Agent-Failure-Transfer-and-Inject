"""Routing experiment with REAL LLM models (Ollama-backed).

Same three conditions as experiment.py (always-small, always-large,
PFTI-routed) but each subtask is solved by an actual local LLM and graded
against the gold answer, with real measured latency.

Run:  python -m pfti.routing.experiment_llm
Needs Ollama running + models pulled (see llm_models.py).
"""
from __future__ import annotations

from .router import PerfStore, PerfRecord, Router
from .llm_models import build_llm_models


# A tiny gradeable benchmark of real subtasks (extend as you like).
BENCH = [
    {"task_type": "extract", "flavor": "table",
     "input": "name=Ada; role=engineer; city=Paris", "question": "city",
     "gold": "Paris"},
    {"task_type": "extract", "flavor": "plain",
     "input": "Invoice total is 250 USD, tax 20 USD.", "question": "tax",
     "gold": "20"},
    {"task_type": "classify", "flavor": "plain",
     "input": "The app crashes when I click save.", "labels": "[bug, feature]",
     "gold": "bug"},
    {"task_type": "classify", "flavor": "plain",
     "input": "Please add a dark mode option.", "labels": "[bug, feature]",
     "gold": "feature"},
    {"task_type": "extract", "flavor": "table",
     "input": "id=42; status=open; owner=Sam", "question": "owner",
     "gold": "Sam"},
    {"task_type": "classify", "flavor": "plain",
     "input": "Login button does nothing on Safari.", "labels": "[bug, feature]",
     "gold": "bug"},
]


def run_condition(models, batch, policy, level="mid"):
    store = PerfStore()
    router = Router(store, models, level=level, default="small")
    agents = ["A", "B", "C"]
    ok_n = cost = latency = 0
    for i, item in enumerate(batch):
        agent = agents[i % 3]
        tt, fl = item["task_type"], item["flavor"]
        if policy in ("small", "large"):
            model = policy
        else:
            model, _ = router.choose(tt, fl, exclude_agent=agent)
        ans, lat, c, ok = models[model].run(tt, item)
        ok_n += int(ok); cost += c; latency += lat
        store.add(PerfRecord(agent, tt, fl, model, ok, lat, c))
        print(f"  [{policy:<6}] {agent} {tt:<9} -> {model:<5} "
              f"ans={ans!r:<12} ok={ok} {lat}ms")
    n = len(batch)
    return {"policy": policy, "success_rate": ok_n / n,
            "total_cost": cost, "total_latency_s": latency / 1000}


def main():
    models = build_llm_models()
    print("Models:", {k: v.ollama_tag for k, v in models.items()})
    results = {}
    for p in ("small", "large", "routed"):
        print(f"\n== {p} ==")
        results[p] = run_condition(models, BENCH, p)
    print("\n" + "=" * 50)
    print(f"  {'policy':<8}{'success':>9}{'cost':>7}{'latency':>10}")
    for p in ("small", "large", "routed"):
        d = results[p]
        print(f"  {p:<8}{d['success_rate']:>8.0%}{d['total_cost']:>7.0f}"
              f"{d['total_latency_s']:>9.1f}s")


if __name__ == "__main__":
    main()
