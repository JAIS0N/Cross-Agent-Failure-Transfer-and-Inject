"""Shadow-mode evaluation (plan §2.6) -- the measurement superpower.

The matcher runs and logs would-be warnings but injects nothing; every
pre-call is also checked against the ground-truth oracle. Matcher
precision/recall per granularity level, with NO behavioral confound:

  TP: would_warn AND oracle says the call fails
  FP: would_warn AND oracle says the call succeeds   (matcher-FP)
  FN: no warn AND oracle says fail AND a peer record existed (matcher-FN)
"""
from __future__ import annotations


def shadow_stats(trace_events: list[dict]) -> dict:
    tp = fp = fn = tn = 0
    store_nonempty = False
    for ev in trace_events:
        if ev.get("event_type") == "post_call" and ev.get("outcome") == "fail":
            store_nonempty = True
        if ev.get("event_type") != "pre_call":
            continue
        warn = ev.get("would_warn", False)
        fails = ev.get("oracle_would_fail") is not None
        if warn and fails:
            tp += 1
        elif warn and not fails:
            fp += 1
        elif not warn and fails and store_nonempty:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall}
