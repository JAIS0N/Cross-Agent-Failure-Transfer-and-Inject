"""What happens after a warning? Harm analysis from saved traces.

    python -m pfti.bench.trace_report bench_results_v2

Needs a run made with `run_all --save-traces`. For every warning delivered
in inject mode it records whether the warning was right (the held call
would really have failed -- oracle) and what the agent did next:

  rerouted  - next call differs from the held one
  repeated  - re-issued the same call (proceeds; advisory warning)
  stopped   - made no further tool call after the warning

It then pairs each inject run with the off run of the same model x scenario
x repeat and classifies every regression (off succeeded, inject failed):

  after a FALSE-positive warning  -> harm from a bad warning
  after TRUE-positive warnings only -> harm from how the model used a
                                        correct warning
  with no warning                  -> not attributable to PFTI (noise)
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path


def _load(trace_dir):
    runs = {}
    for fn in sorted(Path(trace_dir).glob("*.json")):
        d = json.loads(fn.read_text(encoding="utf-8"))
        runs[(d["model"], d["mode"], d["scenario"], d["repeat"])] = d
    return runs


def _call_key(e):
    return (e.get("tool"), json.dumps(e.get("args"), sort_keys=True))


def warnings_in(run):
    """[(agent, correct:bool, next_action:str, tool, args)] for each warning."""
    ev = [e for e in run["events"] if e.get("event_type") == "pre_call"]
    out = []
    for idx, e in enumerate(ev):
        if not e.get("warned"):
            continue
        correct = e.get("oracle_would_fail") is not None
        nxt = next((x for x in ev[idx + 1:] if x.get("agent") == e.get("agent")), None)
        if nxt is None:
            act = "stopped"
        elif _call_key(nxt) == _call_key(e):
            act = "repeated"
        else:
            act = "rerouted"
        out.append((e.get("agent"), correct, act, e.get("tool"), e.get("args")))
    return out


def _last_text(run, agent):
    msgs = (run.get("transcripts") or {}).get(agent) or []
    for m in reversed(msgs):
        if m.get("role") == "assistant" and m.get("content"):
            return str(m["content"])[:240].replace("\n", " ")
    return ""


def build(trace_dir):
    runs = _load(trace_dir)
    L = ["# What happens after a warning (from saved traces)\n"]
    by_model = defaultdict(Counter)
    for (m, mode, sc, rep), r in runs.items():
        if mode != "inject":
            continue
        for agent, correct, act, tool, args in warnings_in(r):
            by_model[m][("TP" if correct else "FP", act)] += 1
    L.append("## Agent response to each warning\n")
    L.append("| model | correct warnings: rerouted / repeated / stopped | false warnings: rerouted / repeated / stopped |")
    L.append("|---|--:|--:|")
    for m in sorted(by_model):
        c = by_model[m]
        f = lambda k: " / ".join(str(c[(k, a)]) for a in ("rerouted", "repeated", "stopped"))
        L.append(f"| {m} | {f('TP')} | {f('FP')} |")
    L.append("")

    L.append("## Regressions: off succeeded, inject failed (same model × scenario × repeat)\n")
    L.append("| model | scenario | rep | cause | warnings (correct?, next action) | agent's last words after warning |")
    L.append("|---|---|--:|---|---|---|")
    cause_n = Counter()
    gains = Counter()
    for (m, mode, sc, rep), r in sorted(runs.items()):
        if mode != "inject":
            continue
        off = runs.get((m, "off", sc, rep))
        if off is None:
            continue
        if r["success"] and not off["success"]:
            gains[m] += 1
        if not (off["success"] and not r["success"]):
            continue
        ws = warnings_in(r)
        if not ws:
            cause = "no warning (noise)"
        elif any(not c for _, c, _, _, _ in ws):
            cause = "after FALSE-positive warning"
        else:
            cause = "after correct warning(s) only"
        cause_n[(m, cause)] += 1
        desc = "; ".join(f"{a}:{'TP' if c else 'FP'},{act}" for a, c, act, _, _ in ws)
        last = _last_text(r, ws[-1][0]) if ws else ""
        L.append(f"| {m} | {sc} | {rep} | {cause} | {desc} | {last} |")
    L.append("")
    L.append("## Summary\n")
    models = sorted({k[0] for k in runs})
    for m in models:
        regs = {c: n for (mm, c), n in cause_n.items() if mm == m}
        L.append(f"- **{m}**: inject fixed {gains[m]} off-failures; regressions: "
                 + (", ".join(f"{n} {c}" for c, n in regs.items()) if regs else "none"))
    return "\n".join(L)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    out_dir = Path(argv[0] if argv else "bench_results")
    md = build(out_dir / "traces")
    (out_dir / "trace_report.md").write_text(md, encoding="utf-8")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
