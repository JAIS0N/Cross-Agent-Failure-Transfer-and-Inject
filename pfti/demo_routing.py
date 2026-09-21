"""Routing-extension demo: predicate-guided model routing from shared trials.

Story: a team of agents solves gradeable subtasks (extract / classify /
summarize). Each subtask can run on a cheap-weak model or an expensive-
strong model. When an agent runs a subtask it records a PERFORMANCE
signature (task_type, flavor, model) -> (ok, latency, cost). Peers read
those records and route THEIR subtasks to the model observed to work best
-- learning the routing policy from teammates instead of rediscovering it.

Run:  python -m pfti.demo_routing
"""
from __future__ import annotations

import statistics as st

from .routing.models import best_model_oracle, SUBTASK_TYPES, MODELS
from .routing.experiment import run_all


def banner(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


def main():
    banner("SETUP: two models with a per-task tradeoff (ground truth)")
    for name, m in MODELS.items():
        print(f"  {name:<6} acc={m.acc}  latency={m.latency_ms}ms  cost={m.cost}")
    print("  oracle best model per task:",
          {t: best_model_oracle(t) for t in SUBTASK_TYPES})
    print("  -> a good router sends easy tasks to 'small', hard to 'large'")

    banner("A. ONE RUN (40 subtasks): three conditions")
    r = run_all(40, seed=0)
    print(f"  {'policy':<8}{'success':>9}{'cost':>7}{'latency':>10}"
          f"{'optimal_route':>16}")
    for p in ("small", "large", "routed"):
        d = r[p]
        print(f"  {p:<8}{d['success_rate']:>8.0%}{d['total_cost']:>7.0f}"
              f"{d['total_latency_s']:>9.1f}s{d['optimal_route_rate']:>15.0%}")

    banner("B. HOW ROUTED LEARNS: first vs last third of the batch")
    tr = r["routed"]["trace"]
    third = len(tr) // 3
    for label, seg in (("first third", tr[:third]), ("last third", tr[-third:])):
        opt = sum(1 for (_a, tt, _f, model, _w, _ok) in seg
                  if model == best_model_oracle(tt)) / len(seg)
        explore = sum(1 for x in seg if x[4].startswith("explore")) / len(seg)
        learned = sum(1 for x in seg if x[4].startswith("learned")) / len(seg)
        print(f"  {label:<12} optimal-route={opt:.0%}  "
              f"exploring={explore:.0%}  using-peer-knowledge={learned:.0%}")
    print("  -> early on it explores; later it routes from peer records")

    banner("C. ROBUSTNESS: mean over 5 seeds (this is the headline)")
    agg = {p: {"s": [], "c": []} for p in ("small", "large", "routed")}
    for seed in range(5):
        rr = run_all(40, seed=seed)
        for p in agg:
            agg[p]["s"].append(rr[p]["success_rate"])
            agg[p]["c"].append(rr[p]["total_cost"])
    for p in ("small", "large", "routed"):
        a = agg[p]
        print(f"  {p:<8} success={st.mean(a['s']):.0%} "
              f"(+/-{st.pstdev(a['s']):.2f})   cost={st.mean(a['c']):.0f}")
    sm, lg, ro = (st.mean(agg[p]["s"]) for p in ("small", "large", "routed"))
    cs, cl, cr = (st.mean(agg[p]["c"]) for p in ("small", "large", "routed"))
    print(f"\n  Routed captures {(ro-sm)/(lg-sm):.0%} of large's accuracy gain "
          f"for {(cr-cs)/(cl-cs):.0%} of its extra cost.")
    print("  => near-large accuracy at a fraction of the cost: the 'good corner'.")


if __name__ == "__main__":
    main()
