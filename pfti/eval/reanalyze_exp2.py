"""Run from the repo root:  python -m pfti.eval.reanalyze_exp2

Re-analysis of the Who&When real-LLM replay (Experiment 2) per Prof. Hailu's
review: paired tests, clustered models, intention-to-treat, and the
same-trajectory / same-agent confound check.  Reads the saved run output
(waw_llm.json) and re-derives the 36 cases from the raw Who&When logs."""
import json, os, sys, random, re
from collections import Counter, defaultdict

REPO = os.path.abspath("." if len(sys.argv) < 2 else sys.argv[1])
WAW = os.path.abspath("waw_results/waw_llm.json" if len(sys.argv) < 3 else sys.argv[2])
OUT = os.path.abspath("review_reanalysis" if len(sys.argv) < 4 else sys.argv[3])
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)
os.chdir(REPO)

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from pfti.eval import whoandwhen_llm as W

res = json.load(open(WAW, encoding="utf-8"))
rows = res["rows"]
cases = W.find_transfer_points(only_op="code_exec")
n_cases = len(cases)
models = []
for r in rows:
    if r["model"] not in models:
        models.append(r["model"])
assert len(rows) == n_cases * len(models), (len(rows), n_cases, len(models))

# ---- align rows to cases (run order: model -> case) and sanity-check ----
recs = []
for i, r in enumerate(rows):
    k = i % n_cases
    c = cases[k]
    assert r["log"] == os.path.basename(c["log_path"]), (i, r["log"])
    if not r.get("error"):
        assert r["later_ec"] == c["later_ec"] and r["code_agent"] == c["code_agent"]
    recs.append((k, c, r))

out = {}
L = []  # markdown lines


def status(r, cond):
    if r.get("error"):
        return "error"
    if r.get(f"{cond}_parsed"):
        return "graded"
    if r.get(f"{cond}_unsafe"):
        return "unsafe"
    if r.get(f"{cond}_unrunnable"):
        return "shell"
    return "no_code"


# ---------------- 1. exclusions by arm ----------------
L.append("## 1. What was excluded, by arm\n")
L.append("| model | arm | graded | no code block | shell block | unsafe (skipped) |")
L.append("|---|---|--:|--:|--:|--:|")
excl = {}
for m in models:
    for cond in ("off", "inject"):
        cnt = Counter(status(r, cond) for _, _, r in recs if r["model"] == m)
        excl[(m, cond)] = cnt
        L.append(f"| {m} | {cond} | {cnt['graded']} | {cnt['no_code']} | "
                 f"{cnt['shell']} | {cnt['unsafe']} |")
tot = {cond: Counter() for cond in ("off", "inject")}
for (m, cond), c in excl.items():
    tot[cond].update(c)
for cond in ("off", "inject"):
    c = tot[cond]
    L.append(f"| **all** | {cond} | {c['graded']} | {c['no_code']} | {c['shell']} | {c['unsafe']} |")
L.append("")


# ---------------- 2. paired McNemar ----------------
def mcnemar(pairs):
    """pairs: list of (off_bool, inject_bool). exact two-sided."""
    b = sum(1 for o, i in pairs if o and not i)   # off yes, inject no (PFTI helped)
    c = sum(1 for o, i in pairs if (not o) and i)  # PFTI hurt
    n = b + c
    p = binomtest(b, n, 0.5).pvalue if n else 1.0
    return b, c, p


def pp(x):
    return f"{x:.2f}" if x >= 0.01 else f"{x:.3f}"


L.append("## 2. Paired analysis (McNemar, exact)\n")
L.append("Each case is run under both arms, so the comparison is within-case. "
         "Only the discordant pairs carry information: *helped* = repeat/failure "
         "off but not inject; *hurt* = the reverse.\n")
L.append("### 2a. Per-protocol (both arms produced runnable Python)\n")
L.append("| model | pairs | same-error: helped / hurt | p | fail: helped / hurt | p |")
L.append("|---|--:|--:|--:|--:|--:|")
pp_pairs_all = {"same": [], "fail": []}
for m in models + ["ALL"]:
    sub = [r for _, _, r in recs if (m == "ALL" or r["model"] == m)
           and status(r, "off") == "graded" and status(r, "inject") == "graded"]
    s = [(r["off_same_error"], r["inject_same_error"]) for r in sub]
    f = [(r["off_failed"], r["inject_failed"]) for r in sub]
    bs, cs, ps_ = mcnemar(s)
    bf, cf, pf = mcnemar(f)
    out[f"pp_{m}"] = dict(pairs=len(sub), same=(bs, cs, ps_), fail=(bf, cf, pf))
    lab = "**all 4 models**" if m == "ALL" else m
    L.append(f"| {lab} | {len(sub)} | {bs} / {cs} | {pp(ps_)} | {bf} / {cf} | {pp(pf)} |")
L.append("")

L.append("### 2b. Intention-to-treat (every case counted; no runnable code = failure)\n")
L.append("Same-error under ITT has no single right answer for a missing block, so "
         "both bounds are shown: *lenient* counts a missing block as not-a-repeat, "
         "*strict* counts it as a repeat.\n")
L.append("| model | n pairs | fail rate off → inject | fail: helped / hurt | p | same-err lenient off → inj | p | same-err strict off → inj | p |")
L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|")


def itt(r, cond, metric, strict=False):
    if status(r, cond) == "graded":
        return bool(r[f"{cond}_{metric}"])
    if metric == "failed":
        return True
    return strict


for m in models + ["ALL"]:
    sub = [r for _, _, r in recs if (m == "ALL" or r["model"] == m)]
    n = len(sub)
    f = [(itt(r, "off", "failed"), itt(r, "inject", "failed")) for r in sub]
    sl = [(itt(r, "off", "same_error"), itt(r, "inject", "same_error")) for r in sub]
    ss = [(itt(r, "off", "same_error", True), itt(r, "inject", "same_error", True)) for r in sub]
    bf, cf, pf = mcnemar(f)
    _, _, psl = mcnemar(sl)
    _, _, pss = mcnemar(ss)
    rate = lambda pairs, j: sum(p[j] for p in pairs) / len(pairs)
    out[f"itt_{m}"] = dict(n=n, fail=(rate(f, 0), rate(f, 1), bf, cf, pf),
                           same_len=(rate(sl, 0), rate(sl, 1), psl),
                           same_str=(rate(ss, 0), rate(ss, 1), pss))
    lab = "**all 4 models**" if m == "ALL" else m
    L.append(f"| {lab} | {n} | {rate(f,0):.0%} → {rate(f,1):.0%} | {bf} / {cf} | {pp(pf)} | "
             f"{rate(sl,0):.1%} → {rate(sl,1):.1%} | {pp(psl)} | "
             f"{rate(ss,0):.0%} → {rate(ss,1):.0%} | {pp(pss)} |")
L.append("")

# ---------------- 3. clustered models ----------------
# Import only the statsmodels pieces we use. `statsmodels.api` pulls in
# scipy.signal, whose compiled DLLs Windows Smart App Control can block.
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod import families, cov_struct
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM

long = []
for k, c, r in recs:
    for cond in ("off", "inject"):
        long.append(dict(case=k, log=os.path.basename(c["log_path"]),
                         model=r["model"], inject=int(cond == "inject"),
                         graded=int(status(r, cond) == "graded"),
                         fail_itt=int(itt(r, cond, "failed")),
                         same=(int(bool(r[f"{cond}_same_error"]))
                               if status(r, cond) == "graded" else np.nan),
                         fail=(int(bool(r[f"{cond}_failed"]))
                               if status(r, cond) == "graded" else np.nan)))
df = pd.DataFrame(long)
df.to_csv(os.path.join(OUT, "exp2_long.csv"), index=False)
n_logs = df["log"].nunique()


def gee(formula, data):
    mod = GEE.from_formula(formula, groups="log", data=data,
                           cov_struct=cov_struct.Exchangeable(),
                           family=families.Binomial())
    fit = mod.fit()
    b = fit.params["inject"]
    lo, hi = fit.conf_int().loc["inject"]
    return np.exp(b), np.exp(lo), np.exp(hi), fit.pvalues["inject"]


def bayes_mixed(ycol, data):
    d = data.dropna(subset=[ycol]).copy()
    d["case_s"] = d["case"].astype(str)
    mod = BinomialBayesMixedGLM.from_formula(
        f"{ycol} ~ inject + C(model)",
        {"log": "0 + C(log)", "case": "0 + C(case_s)"}, d)
    fit = mod.fit_vb()
    names = list(mod.exog_names)
    j = names.index("inject")
    mu, sd = fit.fe_mean[j], fit.fe_sd[j]
    return np.exp(mu), np.exp(mu - 1.96 * sd), np.exp(mu + 1.96 * sd)


def cluster_boot(ycol, data, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = data.dropna(subset=[ycol])
    logs = d["log"].unique()
    by = {g: sub for g, sub in d.groupby("log")}
    diffs = []
    for _ in range(B):
        pick = rng.choice(logs, size=len(logs), replace=True)
        s = pd.concat([by[g] for g in pick])
        a = s[s.inject == 1][ycol].mean()
        o = s[s.inject == 0][ycol].mean()
        diffs.append(a - o)
    point = d[d.inject == 1][ycol].mean() - d[d.inject == 0][ycol].mean()
    return point, np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


L.append("## 3. Clustered models (accounting for 4 models × 36 cases × 25 logs)\n")
L.append(f"The 288 attempts come from {n_cases} cases in {n_logs} logs, re-used across 4 models. "
         "Three analyses that respect that structure; all report the inject effect.\n")
L.append("| outcome | GEE logistic, clustered by log: OR [95% CI], p | mixed-effects logistic (random log + case): OR [95% CI] | cluster bootstrap by log: risk diff [95% CI] |")
L.append("|---|---|---|---|")
for ycol, lab in (("same", "same-error (per-protocol)"),
                  ("fail", "fail (per-protocol)"),
                  ("fail_itt", "fail (ITT)")):
    d = df.dropna(subset=[ycol])
    g = gee(f"{ycol} ~ inject + C(model)", d)
    try:
        bm = bayes_mixed(ycol, df)
        bms = f"{bm[0]:.2f} [{bm[1]:.2f}, {bm[2]:.2f}]"
    except Exception as e:
        bm, bms = None, f"did not fit ({type(e).__name__})"
    cb = cluster_boot(ycol, df)
    out[f"clust_{ycol}"] = dict(gee=g, bayes=bm, boot=cb)
    L.append(f"| {lab} | {g[0]:.2f} [{g[1]:.2f}, {g[2]:.2f}], p={pp(g[3])} | {bms} | "
             f"{cb[0]*100:+.1f} pts [{cb[1]*100:+.1f}, {cb[2]*100:+.1f}] |")
L.append("")

# ---------------- 4. confound: what does the warning add? ----------------
L.append("## 4. What does the warning actually carry? (confound check)\n")


def retained_indices(case):
    """Replicates build_messages' clipping: which history indices the model saw."""
    hist, upto, agent = case["history"], case["code_index"], case["code_agent"]
    prior = hist[:upto]
    if not prior:
        return set()

    def ln(step):
        body = W._clip(step.get("content", ""), W.MAX_TURN_CHARS)
        return len(f"[{step.get('name', '?')}]: {body}")
    keep = {0}
    budget = W.MAX_HISTORY_CHARS - ln(prior[0])
    for j in range(len(prior) - 1, 0, -1):
        l = ln(prior[j])
        if l > budget:
            break
        keep.add(j)
        budget -= l
    return keep


def proposer_before(hist, idx):
    for j in range(idx - 1, -1, -1):
        if W.extract_code(hist[j].get("content", "")):
            return hist[j].get("name", "?"), j
    return None, None


conf = []
for k, c in enumerate(cases):
    rec = c["peer_record"]
    hist = c["history"]
    peer_step = rec.step
    peer_ec = rec.error_class
    prop, prop_idx = proposer_before(hist, peer_step)
    kept = retained_indices(c)
    conf.append(dict(
        case=k, log=os.path.basename(c["log_path"]),
        subset=("Hand-Crafted" if "Hand-Crafted" in c["log_path"] else "Algorithm-Generated"),
        code_agent=c["code_agent"], later_ec=c["later_ec"],
        peer_step=peer_step, peer_ec=peer_ec, peer_record_agent=rec.agent,
        earlier_proposer=prop,
        same_proposer=(prop is not None and prop == c["code_agent"]),
        same_error_class=(str(peer_ec).lower() == str(c["later_ec"]).lower()),
        failure_in_context=(peer_step in kept),
        failing_code_in_context=(prop_idx in kept if prop_idx is not None else False),
        gap_turns=c["code_index"] - peer_step,
        omitted_turns=max(0, c["code_index"] - len(kept)),
    ))
cdf = pd.DataFrame(conf)
cdf.to_csv(os.path.join(OUT, "exp2_confound_cases.csv"), index=False)
n = len(cdf)


def frac(col):
    v = int(cdf[col].sum())
    return f"{v}/{n} ({v/n:.0%})"


rec_agents = Counter(cdf["peer_record_agent"])
L.append(f"- Earlier failure is **still inside the context the model sees** (not clipped): {frac('failure_in_context')}")
L.append(f"- The earlier failing code block itself is in context: {frac('failing_code_in_context')}")
L.append(f"- Earlier failing code was proposed by the **same agent** that is regenerating now: {frac('same_proposer')}")
L.append(f"- Earlier failure has the **same error class** as the historical failure being tested: {frac('same_error_class')}")
L.append(f"- Median gap between the earlier failure and the regenerated turn: {int(cdf['gap_turns'].median())} turns")
L.append(f"- Agent attributed to the stored failure record: {dict(rec_agents)}")
both = int((cdf.failure_in_context & cdf.same_proposer).sum())
novel = int((~cdf.failure_in_context & ~cdf.same_proposer).sum())
novel_same = int((~cdf.failure_in_context & ~cdf.same_proposer & cdf.same_error_class).sum())
L.append(f"- Warning is fully redundant (in context AND same agent): {both}/{n}")
L.append(f"- Warning is genuinely new peer information (out of context AND different agent): {novel}/{n}; "
         f"of those, same error class as the target failure: {novel_same}/{n}")
L.append("")
out["confound"] = dict(
    n=n, in_context=int(cdf.failure_in_context.sum()),
    code_in_context=int(cdf.failing_code_in_context.sum()),
    same_proposer=int(cdf.same_proposer.sum()),
    same_ec=int(cdf.same_error_class.sum()), redundant=both, novel=novel,
    novel_same_ec=novel_same, rec_agents=dict(rec_agents),
    median_gap=float(cdf.gap_turns.median()))

# ---- effect within the subsets where the warning could plausibly matter ----
L.append("### Effect restricted to cases where the warning names the same error class\n")
sub_idx = set(cdf[cdf.same_error_class].case)
for lab, pick in (("same error class", lambda k: k in sub_idx),
                  ("different error class", lambda k: k not in sub_idx)):
    s = [r for k, _, r in recs if pick(k)
         and status(r, "off") == "graded" and status(r, "inject") == "graded"]
    if not s:
        continue
    bs, cs, p = mcnemar([(r["off_same_error"], r["inject_same_error"]) for r in s])
    ro = sum(r["off_same_error"] for r in s) / len(s)
    ri = sum(r["inject_same_error"] for r in s) / len(s)
    L.append(f"- {lab}: {len(s)} graded pairs, same-error {ro:.1%} → {ri:.1%}, helped/hurt {bs}/{cs}, McNemar p={pp(p)}")
    out[f"subset_{lab}"] = dict(pairs=len(s), off=ro, inj=ri, b=bs, c=cs, p=p)
L.append("")

# ---------------- 5. per-case baseline reproduction ----------------
L.append("## 5. How often does the baseline reproduce the historical error?\n")
per_case = defaultdict(lambda: [0, 0])
for k, _, r in recs:
    if status(r, "off") == "graded":
        per_case[k][0] += int(bool(r["off_same_error"]))
        per_case[k][1] += 1
repro = Counter()
for k in range(n_cases):
    a, b = per_case[k]
    repro[a] += 1
L.append("Off-arm same-error count per case, out of the 4 models (temperature 0.2, one sample each):\n")
L.append("| models reproducing the historical error | cases |")
L.append("|---|--:|")
for a in sorted(repro):
    L.append(f"| {a} of 4 | {repro[a]} |")
L.append("")
out["repro_hist"] = dict(repro)
out["n_logs"] = n_logs
out["n_cases"] = n_cases

open(os.path.join(OUT, "exp2_reanalysis.md"), "w", encoding="utf-8").write("\n".join(L))
json.dump(out, open(os.path.join(OUT, "exp2_reanalysis.json"), "w"), indent=1, default=str)
print("\n".join(L))
