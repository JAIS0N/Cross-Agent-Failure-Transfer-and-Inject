"""Stress tests for the reproduction-filtered Who&When result.

    python -m pfti.eval.waw_repro_robustness [waw_repro_results] [review_reanalysis]

The primary analysis (waw_repro_analyze.py) was fixed before the run. This
script checks how much the headline depends on analysis choices:
  * how "no runnable code" is counted (lenient ITT / strict ITT / graded only)
  * counting each distinct case once instead of once per model
  * dropping one model at a time
  * dropping one error class at a time
  * what the warned agents did instead of repeating the error
"""
from __future__ import annotations

import json
import os
import sys


def _gee(d, y):
    import numpy as np
    from statsmodels.genmod.generalized_estimating_equations import GEE
    from statsmodels.genmod import families, cov_struct
    fe = " + C(model)" if d.model.nunique() > 1 else ""
    f = GEE.from_formula(f"{y} ~ inj{fe}", groups="log", data=d,
                         cov_struct=cov_struct.Exchangeable(),
                         family=families.Binomial()).fit()
    lo, hi = f.conf_int().loc["inj"]
    return (f"OR {np.exp(f.params['inj']):.2f} [{np.exp(lo):.2f}, "
            f"{np.exp(hi):.2f}], p={f.pvalues['inj']:.3f}")


def _rates(d, y):
    return f"{d[d.inj == 0][y].mean():.0%} → {d[d.inj == 1][y].mean():.0%}"


def _paired(d, y, unit):
    from scipy.stats import wilcoxon, binomtest
    c = d.groupby(list(unit) + ["inj"])[y].mean().unstack("inj").dropna()
    dd = c[1] - c[0]
    b, w = int((dd < 0).sum()), int((dd > 0).sum())
    sp = binomtest(b, b + w).pvalue if b + w else 1.0
    try:
        wp = wilcoxon(c[1], c[0]).pvalue
    except ValueError:
        wp = 1.0
    return (f"{len(c)} units, better/worse {b}/{w}, mean {dd.mean() * 100:+.1f} pts, "
            f"sign p={sp:.3f}, Wilcoxon p={wp:.3f}")


def build(res_dir):
    import pandas as pd
    rows = []
    with open(os.path.join(res_dir, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    t = pd.DataFrame(rows)
    t = t[t.phase == "test"].copy()
    t["graded"] = t.status == "graded"
    se = t.same_error.fillna(False).astype(bool)
    t["rep"] = (t.graded & se).astype(int)
    t["rep_strict"] = (~t.graded | se).astype(int)
    t["fail"] = (~t.graded | t.failed.fillna(True).astype(bool)).astype(int)
    t["inj"] = (t.arm == "inject").astype(int)
    g = t[t.graded]

    L = ["# Who&When result: stress tests\n",
         f"{len(t)} phase-2 samples, {t.groupby(['model', 'case_id']).ngroups} "
         f"model×case units, {t.case_id.nunique()} distinct cases from "
         f"{t.log.nunique()} logs "
         f"({sum(c.startswith('hand') for c in t.case_id.unique())} from the "
         f"hand-crafted half).\n",
         "All GEE rows: logistic regression, clustered by log, model as fixed effect.\n",
         "## 1. How 'no runnable code' is counted\n",
         "| variant | repeat off → inject | GEE |", "|---|--:|--:|",
         f"| primary: no code = not a repeat | {_rates(t, 'rep')} | {_gee(t, 'rep')} |",
         f"| strict: no code = a repeat | {_rates(t, 'rep_strict')} | {_gee(t, 'rep_strict')} |",
         f"| graded samples only | {_rates(g, 'rep')} | {_gee(g, 'rep')} |",
         f"| *any failure* (not just the same error) | {_rates(t, 'fail')} | {_gee(t, 'fail')} |",
         "",
         "## 2. What counts as one unit\n",
         f"- model × case (as pre-specified): {_paired(t, 'rep', ('model', 'case_id'))}",
         f"- distinct case (averaged over models): {_paired(t, 'rep', ('case_id',))}",
         "",
         "## 3. Leave one model out\n",
         "| dropped | repeat off → inject | GEE |", "|---|--:|--:|"]
    for m in t.model.unique():
        d = t[t.model != m]
        L.append(f"| {m} | {_rates(d, 'rep')} | {_gee(d, 'rep')} |")
    L += ["", "## 4. By historical error class\n",
          "| error class | units | repeat off → inject | without this class: GEE |",
          "|---|--:|--:|--:|"]
    for ec, n in t.groupby("later_ec").apply(
            lambda d: d.groupby(["model", "case_id"]).ngroups).sort_values(ascending=False).items():
        d_in, d_out = t[t.later_ec == ec], t[t.later_ec != ec]
        L.append(f"| {ec} | {n} | {_rates(d_in, 'rep')} | {_gee(d_out, 'rep')} |")

    def outcome(r):
        if r.status != "graded":
            return "no runnable code"
        if r.same_error:
            return "same error (repeat)"
        if r.failed:
            return "different error"
        return "ran OK"
    t["outcome"] = t.apply(outcome, axis=1)
    L += ["", "## 5. What happened instead of the repeat\n",
          "Share of samples. FileNotFoundError is shown separately because the effect is concentrated there.\n",
          "| subset | arm | same error | different error | ran OK | no runnable code |",
          "|---|---|--:|--:|--:|--:|"]
    for lab, d in (("FileNotFoundError cases", t[t.later_ec == "FileNotFoundError"]),
                   ("all other cases", t[t.later_ec != "FileNotFoundError"]),
                   ("all", t)):
        for arm in ("off", "inject"):
            s = d[d.arm == arm].outcome.value_counts(normalize=True)
            L.append(f"| {lab} | {arm} | {s.get('same error (repeat)', 0):.0%} | "
                     f"{s.get('different error', 0):.0%} | {s.get('ran OK', 0):.0%} | "
                     f"{s.get('no runnable code', 0):.0%} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    res = argv[0] if argv else "waw_repro_results"
    out = argv[1] if len(argv) > 1 else "review_reanalysis"
    os.makedirs(out, exist_ok=True)
    md = build(res)
    with open(os.path.join(out, "waw_repro_robustness.md"), "w", encoding="utf-8") as f:
        f.write(md)
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(md.encode(enc, errors="replace").decode(enc, errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
