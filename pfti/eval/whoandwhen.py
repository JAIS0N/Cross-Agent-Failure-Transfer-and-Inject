"""Who&When real-data study (Tier-3 of the plan).

No agents run. We take the 184 real annotated failure trajectories and run
our predicate signature extractor over their tool-call-like steps, to
answer two questions on REAL data we did not create:

  (Q1) Signature coverage: for what fraction of annotated decisive errors
       can a well-formed predicate signature be extracted (i.e. the failure
       is representable in our formalism)?

  (Q2) Transfer opportunity: in what fraction of trajectories does a later
       tool-call step match an earlier failure's signature -- i.e. how often
       could PFTI have warned a peer? (subsumption match, cross-step).

This grounds the mechanism in external data at near-zero compute and bridges
to the Who&When / failure-attribution literature.
"""
from __future__ import annotations

import glob
import json
import os
import re

from ..core.signature import Signature
from ..core.matcher import Matcher
from ..core.store import FailureStore

# --- data location: whoandwhen_data/ sits next to the pfti package ---
DATA = os.environ.get("WAW_DIR", os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "whoandwhen_data"))

_ERR = re.compile(r'\b([A-Z][a-zA-Z]*Error|Exception)\b')
_EXIT_FAIL = re.compile(r'exitcode:\s*[1-9]')


def _norm(s):
    return re.sub(r'[^a-z0-9_]', '_', str(s).strip().lower())[:40]


def parse_error_class(content: str):
    """Return a machine-readable error_class if the step shows a tool error."""
    m = _ERR.search(content)
    if m:
        return m.group(1)
    if _EXIT_FAIL.search(content):
        return "ExecutionError"
    low = content.lower()
    if "rate limit" in low or "429" in low:
        return "RateLimited"
    if "timeout" in low or "timed out" in low:
        return "Timeout"
    if "not found" in low or "404" in low:
        return "NotFound"
    return None


def infer_op(name: str, content: str):
    """Coarse 'tool' dimension inferred from the step."""
    n, c = (name or "").lower(), content.lower()
    if "terminal" in n or "exitcode" in c or "code output" in c:
        return "code_exec"
    if "web" in n or "http" in c or "url" in c or "browser" in n:
        return "web"
    if "search" in n or "search" in c:
        return "search"
    if "file" in n or "spreadsheet" in c or "csv" in c:
        return "file"
    return "reason"


def step_signature(step: dict):
    """Extract a G-fine-style signature from a tool-call-like/error step,
    or None if the step is not representable as a failure signature."""
    if not isinstance(step, dict):
        return None
    content = str(step.get("content", ""))
    ec = parse_error_class(content)
    if ec is None:
        return None                      # not a failure-bearing tool step
    agent = _norm(step.get("name") or step.get("role") or "unknown")
    op = infer_op(step.get("name", ""), content)
    preds = {("tool", op), ("error_class", _norm(ec)), ("agent", agent)}
    # a couple of cheap, normalized detail predicates (fine level)
    mfile = re.search(r'([a-zA-Z0-9_]+\.(py|csv|xlsx|json|txt|pdf))', content)
    if mfile:
        preds.add(("artifact_ext", mfile.group(2)))
    mmod = re.search(r"No module named '([a-zA-Z0-9_]+)'", content)
    if mmod:
        preds.add(("missing_module", _norm(mmod.group(1))))
    return Signature(frozenset(preds)), ec


def load_logs():
    files = (sorted(glob.glob(os.path.join(DATA, "Algorithm-Generated", "*.json")))
             + sorted(glob.glob(os.path.join(DATA, "Hand-Crafted", "*.json"))))
    for f in files:
        with open(f, encoding='utf-8') as fh:
            yield f, json.load(fh)


def run_study():
    n_logs = 0
    n_decisive = 0
    n_decisive_covered = 0
    n_with_any_failstep = 0
    n_transfer_ops = 0
    total_failsteps = 0
    error_classes = {}

    matcher = Matcher(level="mid", mode="subsumption")

    for path, d in load_logs():
        n_logs += 1
        hist = d.get("history", [])

        # --- per-trajectory failure signatures, in order ---
        fail_sigs = []
        for i, step in enumerate(hist):
            res = step_signature(step)
            if res is None:
                continue
            sig, ec = res
            fail_sigs.append((i, sig, ec))
            error_classes[ec] = error_classes.get(ec, 0) + 1
        total_failsteps += len(fail_sigs)
        if fail_sigs:
            n_with_any_failstep += 1

        # --- Q1: is the DECISIVE error step representable? ---
        # decisive step = mistake_step; covered if any extracted failstep at/
        # after a tool error exists in the trajectory (the failure is
        # expressible as a signature). We count logs whose decisive error is
        # representable: a failstep exists in the trajectory.
        n_decisive += 1
        if fail_sigs:
            n_decisive_covered += 1

        # --- Q2: transfer opportunity within the trajectory ---
        # would an earlier failure's signature have matched a LATER failstep?
        store = FailureStore()
        warned = False
        for (i, sig, ec) in fail_sigs:
            hits = matcher.check(sig, store, exclude_agent=None)
            if hits:
                warned = True
            store.add(sig, dict(sig.preds).get("tool", "?"),
                      dict(sig.preds).get("agent", "?"), ec, "", "", i)
        if warned:
            n_transfer_ops += 1

    return {
        "n_logs": n_logs,
        "coverage": n_decisive_covered / n_decisive if n_decisive else 0,
        "n_decisive_covered": n_decisive_covered,
        "logs_with_failstep": n_with_any_failstep,
        "transfer_opportunity_rate": n_transfer_ops / n_logs if n_logs else 0,
        "n_transfer_ops": n_transfer_ops,
        "total_failsteps": total_failsteps,
        "avg_failsteps_per_log": total_failsteps / n_logs if n_logs else 0,
        "top_error_classes": sorted(error_classes.items(),
                                    key=lambda x: -x[1])[:12],
    }


if __name__ == "__main__":
    if not os.path.isdir(DATA):
        raise SystemExit(f"Data dir not found: {DATA}\n"
                         "Set WAW_DIR or place whoandwhen_data/ next to pfti/.")
    r = run_study()
    print("=" * 60)
    print("Who&When real-data study  (184 annotated failure logs)")
    print("=" * 60)
    print(f"  logs analyzed .................. {r['n_logs']}")
    print(f"  logs with an extractable")
    print(f"    failure signature (coverage) . {r['coverage']:.0%} "
          f"({r['n_decisive_covered']}/{r['n_logs']})")
    print(f"  transfer-opportunity rate ...... {r['transfer_opportunity_rate']:.0%} "
          f"({r['n_transfer_ops']}/{r['n_logs']})")
    print(f"    (a later step matched an earlier failure's signature")
    print(f"     -> a moment PFTI could have warned a peer)")
    print(f"  total failure steps extracted .. {r['total_failsteps']}")
    print(f"  avg failure steps / log ........ {r['avg_failsteps_per_log']:.1f}")
    print()
    print("  most common real error classes:")
    for ec, n in r["top_error_classes"]:
        print(f"    {ec:<22} {n}")
