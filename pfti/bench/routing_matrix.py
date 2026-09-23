"""Routing before/after: baselines (always-X) vs PFTI-routed, real models.

Generalizes routing/experiment.py to the full model matrix. Every subtask
is gradeable (gold answer -> ok = answer == gold), so no LLM judge is
needed. Conditions:

  * always-<model>  for each model in the matrix  (the baselines / ceilings)
  * routed          picks a model per subtask from PEER performance records
                    (PFTI mechanism, self excluded), cold-starting on the
                    cheapest model

Metrics per condition: success rate, total cost (size proxy), total latency.
The headline: routed sits at the 'good corner' -- near the best model's
accuracy at a fraction of its cost.

Provider-agnostic: pass a dict {name: runnable} where runnable.run(
task_type, item) -> (answer, latency_ms, cost, ok). ``build_real_models``
wires real Ollama models; ``build_fake_models`` gives deterministic
profiles for offline validation.
"""
from __future__ import annotations

from .providers import MODEL_MATRIX
from ..routing.router import PerfStore, PerfRecord, Router

BENCH = [
    {"task_type": "extract", "flavor": "table",
     "input": "name=Ada; role=engineer; city=Paris", "question": "city",
     "gold": "paris"},
    {"task_type": "extract", "flavor": "plain",
     "input": "Invoice total is 250 USD, tax 20 USD.", "question": "tax",
     "gold": "20"},
    {"task_type": "extract", "flavor": "table",
     "input": "id=42; status=open; owner=Sam", "question": "owner",
     "gold": "sam"},
    {"task_type": "extract", "flavor": "plain",
     "input": "The meeting is on Tuesday at 3pm in room 4.",
     "question": "day", "gold": "tuesday"},
    {"task_type": "classify", "flavor": "plain",
     "input": "The app crashes when I click save.", "labels": "[bug, feature]",
     "gold": "bug"},
    {"task_type": "classify", "flavor": "plain",
     "input": "Please add a dark mode option.", "labels": "[bug, feature]",
     "gold": "feature"},
    {"task_type": "classify", "flavor": "plain",
     "input": "Login button does nothing on Safari.",
     "labels": "[bug, feature]", "gold": "bug"},
    {"task_type": "classify", "flavor": "plain",
     "input": "Could we support CSV export too?", "labels": "[bug, feature]",
     "gold": "feature"},
    {"task_type": "extract", "flavor": "table",
     "input": "sku=A1; price=9; qty=3", "question": "price", "gold": "9"},
    {"task_type": "classify", "flavor": "plain",
     "input": "Page 500s intermittently under load.",
     "labels": "[bug, feature]", "gold": "bug"},
    {"task_type": "extract", "flavor": "plain",
     "input": "Contact: jane@acme.com, phone 555-1234.", "question": "email",
     "gold": "janeacmecom"},
    {"task_type": "classify", "flavor": "plain",
     "input": "Add keyboard shortcuts please.", "labels": "[bug, feature]",
     "gold": "feature"},
]


def run_condition(models, batch, policy, level="mid", default="small"):
    store = PerfStore()
    router = Router(store, models, level=level, default=default)
    agents = ["A", "B", "C"]
    ok_n = cost = latency = 0
    for i, item in enumerate(batch):
        agent = agents[i % 3]
        tt, fl = item["task_type"], item["flavor"]
        if policy == "routed":
            model, _ = router.choose(tt, fl, exclude_agent=agent)
        else:
            model = policy  # always-<model>
        ans, lat, c, ok = models[model].run(tt, item)
        ok_n += int(ok)
        cost += c
        latency += lat
        store.add(PerfRecord(agent, tt, fl, model, ok, lat, c))
    n = len(batch)
    return {"policy": policy, "success_rate": ok_n / n, "total_cost": cost,
            "total_latency_s": latency / 1000.0, "n": n}


def run_routing(models, batch=None, default=None):
    batch = batch or BENCH
    default = default or min(models, key=lambda m: getattr(models[m], "cost",
                                                           1.0))
    conditions = list(models.keys()) + ["routed"]
    rows = []
    for p in conditions:
        rows.append(run_condition(models, batch, p, default=default))
    return rows


# ---- real models (Ollama) -------------------------------------------------

def build_real_models(tags=None):
    """Wire the matrix tags to real Ollama-backed LLMModel objects."""
    from openai import OpenAI
    import os
    from ..routing.llm_models import LLMModel
    tags = tags or list(MODEL_MATRIX.keys())
    client = OpenAI(base_url=os.environ.get("PFTI_BASE_URL",
                                            "http://localhost:11434/v1"),
                    api_key=os.environ.get("PFTI_API_KEY", "ollama"))
    models = {}
    for t in tags:
        cost = MODEL_MATRIX.get(t, {}).get("cost", 1.0)
        models[t] = LLMModel(t, t, cost, client)
    return models


# ---- fake models (offline) ------------------------------------------------

class _FakeModel:
    """Deterministic profile: bigger model = more accurate, costlier, slower.
    Grades against gold so routed can learn which model wins per task type."""

    def __init__(self, name, size_b, cost):
        self.name = name
        self.cost = cost
        self.latency = int(120 * size_b)
        # accuracy rises with size; classify is 'easy' (small already good)
        self.acc = {
            "extract": min(0.98, 0.30 + 0.05 * size_b),
            "classify": min(0.98, 0.80 + 0.012 * size_b),
            "summarize": min(0.98, 0.40 + 0.04 * size_b),
        }

    def run(self, task_type, item, rng=None):
        # deterministic competence: a model "handles" a task type if its
        # profile accuracy clears a bar. This gives a clean per-(model,type)
        # oracle so routed can converge to the good corner offline.
        ok = self.acc.get(task_type, 0.5) >= 0.60
        ans = item["gold"] if ok else "__wrong__"
        return ans, self.latency, self.cost, ok


def build_fake_models(tags=None):
    tags = tags or list(MODEL_MATRIX.keys())
    return {t: _FakeModel(t, MODEL_MATRIX[t]["size_b"], MODEL_MATRIX[t]["cost"])
            for t in tags}
