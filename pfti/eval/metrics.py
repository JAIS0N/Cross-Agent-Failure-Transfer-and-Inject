"""Aggregate metrics across scenario runs (plan §4.1)."""
from __future__ import annotations

from .shadow import shadow_stats


def summarize(results: list[dict]) -> dict:
    n = len(results)
    succ = sum(r["success"] for r in results)
    rep = sum(r["repeated_peer_failures"] for r in results)
    fails = sum(r["failures"] for r in results)
    return {"n_scenarios": n, "success_rate": succ / n if n else None,
            "total_failures": fails, "repeated_peer_failures": rep}


def shadow_table(per_level_results: dict) -> list[dict]:
    """per_level_results: level -> list of episode results (shadow mode)."""
    rows = []
    for level, results in per_level_results.items():
        agg = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for r in results:
            s = shadow_stats(r["trace"])
            for k in agg:
                agg[k] += s[k]
        p = agg["TP"] / (agg["TP"] + agg["FP"]) if agg["TP"] + agg["FP"] else None
        rc = agg["TP"] / (agg["TP"] + agg["FN"]) if agg["TP"] + agg["FN"] else None
        rows.append({"level": level, **agg, "precision": p, "recall": rc})
    return rows
