"""Analysis for waw_repro.py (reproduction-filtered Who&When study).

    python -m pfti.eval.waw_repro_analyze waw_repro_results

Primary outcome: same-error repeat (the regenerated code fails with the
historical error class), phase-2 samples only. Intention-to-treat: a sample
with no runnable Python counts as NOT a repeat for this outcome and as a
failure for the any-failure outcome; per-protocol numbers (graded samples
only) are reported alongside.

Inference respects the design (samples nested in cases nested in logs, the
same case re-used across models):
  * case-level paired comparison (per model x case: inject rate - off rate),
    Wilcoxon signed-rank + sign test;
  * GEE logistic regression clustered by log, model as fixed effect;
  * mixed-effects logistic regression, random intercepts for case and log
    (statsmodels BinomialBayesMixedGLM, variational Bayes);
  * cluster bootstrap over logs for the risk difference.
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict


def _load(out_dir):
    rows = []
    with open(os.path.join(out_dir, "results.jsonl"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    meta = json.load(open(os.path.join(out_dir, "meta.json"), encoding="utf-8"))
    return rows, meta


def _pct(a, b):
    return f"{a}/{b} ({a / b:.0%})" if b else "0/0"


def analyze(out_dir):
    import numpy as np
    import pandas as pd
    rows, meta = _load(out_dir)
    df = pd.DataFrame(rows)
    ctx = {c["case_id"]: c["same_class_failure_in_context"] for c in meta["cases"]}
    df["graded"] = df["status"] == "graded"
    df["repeat_itt"] = (df["graded"] & df["same_error"].fillna(False).astype(bool)).astype(int)
    df["fail_itt"] = (~df["graded"] | df["failed"].fillna(True).astype(bool)).astype(int)
    df["inject"] = (df["arm"] == "inject").astype(int)
    df["ctx"] = df["case_id"].map(ctx).fillna(False).astype(bool)
    L, R = [], {"meta": {k: v for k, v in meta.items() if k != "cases"}}
    k_s, k_t, mr = meta["k_screen"], meta["k_test"], meta["min_repro"]

    L.append("# Who&When reproduction-filtered study\n")
    L.append(f"{meta['n_cases']} code-failure moments from {meta['n_logs']} logs. "
             f"Screen: {k_s} off samples per case at T={meta['temperature']}; keep "
             f"cases that repeat the historical error ≥{mr}/{k_s}. Test: fresh "
             f"{k_t} off + {k_t} inject samples per kept case."
             + ("  **(FAKE offline run — plumbing check only)**" if meta.get("fake") else "") + "\n")

    # ---------------- screening ----------------
    scr = df[df.phase == "screen"]
    L.append("## Screening (baseline reproduction)\n")
    L.append("| model | cases screened | baseline repeat rate (all samples) | cases kept |")
    L.append("|---|--:|--:|--:|")
    kept = {}
    for m in meta["models"]:
        s = scr[scr.model == m]
        per = s.groupby("case_id")["repeat_itt"].sum()
        kept[m] = set(per[per >= mr].index)
        L.append(f"| {m} | {per.size} | {_pct(int(s.repeat_itt.sum()), len(s))} | {len(kept[m])} |")
        R.setdefault("screen", {})[m] = dict(cases=int(per.size),
                                             repeat=int(s.repeat_itt.sum()),
                                             n=int(len(s)), kept=len(kept[m]))
    L.append("")

    test = df[df.phase == "test"].copy()
    if test.empty:
        L.append("_No phase-2 samples yet._")
        return "\n".join(L), R

    # ---------------- primary table ----------------
    L.append("## Phase 2: off vs inject on kept cases (fresh samples)\n")
    L.append("| model | kept cases | repeat off → inject (ITT) | repeat, per-protocol | any failure off → inject (ITT) | no runnable code off / inject |")
    L.append("|---|--:|--:|--:|--:|--:|")
    for m in meta["models"] + ["ALL"]:
        t = test if m == "ALL" else test[test.model == m]
        if t.empty:
            continue
        o, i = t[t.inject == 0], t[t.inject == 1]
        og, ig = o[o.graded], i[i.graded]
        lab = "**all models**" if m == "ALL" else m
        nk = t.groupby(["model", "case_id"]).ngroups
        L.append(f"| {lab} | {nk} | {o.repeat_itt.mean():.0%} → {i.repeat_itt.mean():.0%} "
                 f"(n={len(o)}/{len(i)}) | "
                 f"{og.same_error.astype(bool).mean() if len(og) else float('nan'):.0%} → "
                 f"{ig.same_error.astype(bool).mean() if len(ig) else float('nan'):.0%} | "
                 f"{o.fail_itt.mean():.0%} → {i.fail_itt.mean():.0%} | "
                 f"{(~o.graded).sum()} / {(~i.graded).sum()} |")
        R.setdefault("test", {})[m] = dict(
            kept=nk, off=float(o.repeat_itt.mean()), inj=float(i.repeat_itt.mean()),
            n_off=int(len(o)), n_inj=int(len(i)),
            fail_off=float(o.fail_itt.mean()), fail_inj=float(i.fail_itt.mean()))
    L.append("")

    # ---------------- case-level paired ----------------
    from scipy.stats import wilcoxon, binomtest
    cl = (test.groupby(["model", "case_id", "log", "inject"])["repeat_itt"].mean()
          .unstack("inject").dropna())
    cl.columns = ["off", "inject"]
    d = cl["inject"] - cl["off"]
    better, worse, same = int((d < 0).sum()), int((d > 0).sum()), int((d == 0).sum())
    try:
        wp = wilcoxon(cl["inject"], cl["off"], zero_method="wilcox").pvalue if (better + worse) else 1.0
    except ValueError:
        wp = 1.0
    sp = binomtest(better, better + worse, 0.5).pvalue if (better + worse) else 1.0
    L.append("## Case-level paired comparison (each model × case is one unit)\n")
    L.append(f"- units: {len(cl)}; inject lower repeat rate in **{better}**, higher in **{worse}**, tied in {same}")
    L.append(f"- mean change in repeat rate: {d.mean() * 100:+.1f} pts; Wilcoxon signed-rank p={wp:.3g}; sign test p={sp:.3g}")
    L.append("")
    R["paired"] = dict(units=int(len(cl)), better=better, worse=worse, tied=same,
                       mean_diff=float(d.mean()), wilcoxon_p=float(wp), sign_p=float(sp))

    # ---------------- clustered models ----------------
    L.append("## Clustered models (primary outcome: repeat, ITT)\n")
    L.append("| method | inject effect | 95% CI | p |")
    L.append("|---|--:|--:|--:|")
    try:
        # direct imports: statsmodels.api pulls in scipy.signal, whose DLLs
        # Windows Smart App Control can block
        from statsmodels.genmod.generalized_estimating_equations import GEE
        from statsmodels.genmod import families, cov_struct
        fe = " + C(model)" if test.model.nunique() > 1 else ""
        fit = GEE.from_formula(f"repeat_itt ~ inject{fe}", groups="log",
                               data=test, cov_struct=cov_struct.Exchangeable(),
                               family=families.Binomial()).fit()
        b = fit.params["inject"]
        lo, hi = fit.conf_int().loc["inject"]
        L.append(f"| GEE logistic, clustered by log | OR {np.exp(b):.2f} | "
                 f"[{np.exp(lo):.2f}, {np.exp(hi):.2f}] | {fit.pvalues['inject']:.3g} |")
        R["gee"] = dict(OR=float(np.exp(b)), lo=float(np.exp(lo)),
                        hi=float(np.exp(hi)), p=float(fit.pvalues["inject"]))
        try:
            from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
            mod = BinomialBayesMixedGLM.from_formula(
                f"repeat_itt ~ inject{fe}",
                {"case": "0 + C(case_id)", "log": "0 + C(log)"}, test)
            vb = mod.fit_vb()
            j = list(mod.exog_names).index("inject")
            mu, sd = vb.fe_mean[j], vb.fe_sd[j]
            L.append(f"| mixed-effects logistic, random case + log (VB) | OR {np.exp(mu):.2f} | "
                     f"[{np.exp(mu - 1.96 * sd):.2f}, {np.exp(mu + 1.96 * sd):.2f}] | — |")
            R["mixed"] = dict(OR=float(np.exp(mu)), lo=float(np.exp(mu - 1.96 * sd)),
                              hi=float(np.exp(mu + 1.96 * sd)))
        except Exception as e:
            L.append(f"| mixed-effects logistic | did not fit ({type(e).__name__}) | | |")
    except ImportError as e:
        L.append(f"| GEE / mixed | could not import statsmodels ({e}) | | |")

    rng = np.random.default_rng(0)
    logs = test["log"].unique()
    by = {g: s for g, s in test.groupby("log")}
    diffs = []
    for _ in range(4000):
        pick = rng.choice(logs, size=len(logs), replace=True)
        s = pd.concat([by[g] for g in pick])
        diffs.append(s[s.inject == 1].repeat_itt.mean() - s[s.inject == 0].repeat_itt.mean())
    pt = test[test.inject == 1].repeat_itt.mean() - test[test.inject == 0].repeat_itt.mean()
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    L.append(f"| cluster bootstrap over logs | {pt * 100:+.1f} pts | [{lo * 100:+.1f}, {hi * 100:+.1f}] | — |")
    R["boot"] = dict(diff=float(pt), lo=float(lo), hi=float(hi))
    L.append("")

    # ---------------- moderators ----------------
    L.append("## Moderators\n")
    for lab, mask in (("same-class failure already visible in context", test.ctx),
                      ("not visible in context", ~test.ctx)):
        t = test[mask]
        if t.empty:
            continue
        L.append(f"- {lab}: repeat {t[t.inject == 0].repeat_itt.mean():.0%} → "
                 f"{t[t.inject == 1].repeat_itt.mean():.0%} "
                 f"({t.groupby(['model', 'case_id']).ngroups} model×case units)")
    ec = test.groupby(["later_ec", "inject"])["repeat_itt"].mean().unstack()
    if ec.shape[1] == 2:
        L.append("\nBy historical error class (repeat rate off → inject):\n")
        for k, r in ec.iterrows():
            n = int((test.later_ec == k).sum() / 2)
            L.append(f"- {k}: {r[0]:.0%} → {r[1]:.0%} (n={n} per arm)")
    ack = test[test.inject == 1]["mentions_warning"].dropna()
    if len(ack):
        L.append(f"\nInject responses that engage with the warning (keyword check): {ack.mean():.0%}")
    L.append("")
    return "\n".join(L), R


def _figure(out_dir):
    try:
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows, meta = _load(out_dir)
    df = pd.DataFrame(rows)
    t = df[df.phase == "test"].copy()
    if t.empty:
        return
    t["rep"] = ((t.status == "graded") & t.same_error.fillna(False).astype(bool)).astype(int)
    models = [m for m in meta["models"] if m in set(t.model)]
    fig, axes = plt.subplots(1, len(models), figsize=(3.2 * len(models), 3.6),
                             sharey=True, squeeze=False)
    for ax, m in zip(axes[0], models):
        c = t[t.model == m].groupby(["case_id", "arm"]).rep.mean().unstack()
        c = c.sort_values("off", ascending=False)
        for i, (_, r) in enumerate(c.iterrows()):
            ax.plot([0, 1], [r.get("off", 0), r.get("inject", 0)],
                    color="#8a8f98", lw=1, alpha=.7)
        ax.plot([0, 1], [c["off"].mean(), c["inject"].mean()], color="#2563eb",
                lw=3, marker="o", label="mean")
        ax.set_xticks([0, 1], ["off", "inject"])
        ax.set_title(f"{m}\n({len(c)} kept cases)", fontsize=9)
        ax.set_ylim(-.05, 1.05)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("repeat rate per case (fresh samples)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "waw_repro_cases.png"), dpi=140)
    plt.close(fig)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    out_dir = argv[0] if argv else "waw_repro_results"
    md, R = analyze(out_dir)
    with open(os.path.join(out_dir, "summary.md"), "w", encoding="utf-8") as f:
        f.write(md)
    with open(os.path.join(out_dir, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(R, f, indent=1, default=str)
    _figure(out_dir)
    # Windows consoles (cp1252) can't print characters like "≥"; the files
    # above are already written in UTF-8, so only the echo is degraded.
    enc = getattr(sys.stdout, "encoding", None) or "utf-8"
    print(md.encode(enc, errors="replace").decode(enc, errors="replace"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
