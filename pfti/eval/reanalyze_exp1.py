"""Run from the repo root:  python -m pfti.eval.reanalyze_exp1

Re-analysis of Experiment 1 (controlled testbed, repeats=3) per Prof.
Hailu's review. Uses only the saved aggregate (core_matrix.json) plus the
testbed code itself (for the false-positive feasibility check)."""
import json, os, sys, itertools, copy
REPO = "." if len(sys.argv) < 2 else sys.argv[1]
CORE = "bench_results/core_matrix.json" if len(sys.argv) < 3 else sys.argv[2]
OUT = "review_reanalysis" if len(sys.argv) < 4 else sys.argv[3]
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, REPO)

from pfti.core.signature import extract
from pfti.core.store import FailureStore
from pfti.core.matcher import Matcher
from pfti.envs.tools import World, TOOLS, ToolError, would_fail
from pfti.bench.scenarios_llm import SCENARIOS

cells = json.load(open(CORE, encoding="utf-8"))
C = {(c["model"], c["mode"]): c for c in cells}
models = []
for c in cells:
    if c["model"] not in models:
        models.append(c["model"])
L, out = [], {}

# ---------- 1. pooled outcomes ----------
def tot(mode, key):
    return sum(C[(m, mode)][key] for m in models)

L.append("## 1. Pooled outcomes (60 trials per mode)\n")
L.append("| metric | off | shadow | inject | inject vs off |")
L.append("|---|--:|--:|--:|--:|")
for key, lab in (("n_success", "task success"), ("failures", "executed tool failures"),
                 ("repeated_peer_failures", "repeated peer failures (matcher-defined)"),
                 ("warnings_delivered", "warnings delivered"),
                 ("ignored_warnings", "ignored warnings"),
                 ("total_tokens", "total tokens")):
    o, s, i = tot("off", key), tot("shadow", key), tot("inject", key)
    rel = f"{(i - o) / o:+.0%}" if o else "—"
    out[key] = (o, s, i)
    L.append(f"| {lab} | {o:,} | {s:,} | {i:,} | {rel} |")
L.append("")

# ---------- 2. shadow - off noise floor, every metric ----------
L.append("## 2. Noise floor: shadow minus off, every metric\n")
L.append("Shadow never changes what the agent sees, so any shadow–off gap is run-to-run "
         "nondeterminism (Ollama on GPU is not bit-reproducible even at temperature 0).\n")
L.append("| model | success | failures | repeated peer | tokens |")
L.append("|---|--:|--:|--:|--:|")
gaps = {}
for m in models:
    o, s = C[(m, "off")], C[(m, "shadow")]
    g = dict(succ=s["n_success"] - o["n_success"], fail=s["failures"] - o["failures"],
             rp=s["repeated_peer_failures"] - o["repeated_peer_failures"],
             tok=(s["total_tokens"] - o["total_tokens"]) / o["total_tokens"])
    gaps[m] = g
    L.append(f"| {m} | {g['succ']:+d}/15 | {g['fail']:+d} ({g['fail']/o['failures']:+.0%}) | "
             f"{g['rp']:+d} ({g['rp']/max(1,o['repeated_peer_failures']):+.0%}) | {g['tok']:+.1%} |")
out["gaps"] = gaps
L.append("")
L.append("Inject-vs-off effect next to the largest shadow–off gap for the same model:\n")
L.append("| model | Δ success (inject−off) | Δ failures | Δ tokens | |shadow−off| failures |")
L.append("|---|--:|--:|--:|--:|")
for m in models:
    o, i = C[(m, "off")], C[(m, "inject")]
    L.append(f"| {m} | {i['n_success']-o['n_success']:+d}/15 | {i['failures']-o['failures']:+d} | "
             f"{(i['total_tokens']-o['total_tokens'])/o['total_tokens']:+.1%} | {abs(gaps[m]['fail'])} |")
L.append("")

# ---------- 3. circularity: repeated-peer vs ignored ----------
L.append("## 3. Is 'repeated peer failure' circular under inject?\n")
L.append("| model | inject: repeated peer | inject: ignored warnings |")
L.append("|---|--:|--:|")
for m in models:
    i = C[(m, "inject")]
    L.append(f"| {m} | {i['repeated_peer_failures']} | {i['ignored_warnings']} |")
L.append("")

# ---------- 4. bad-call attempts: executed failures + calls held by a warning ----------
L.append("## 4. Are agents making fewer bad calls, or are bad calls being held?\n")
L.append("In inject, a call that matches a peer failure is *held* (not executed) the first time. "
         "A held call cannot be counted as a failure, so part of the failure drop is mechanical. "
         "Adding held calls back gives the number of doomed calls the agents actually *attempted*. "
         "Exact split of held true- vs false-positives needs traces (not saved in this run), so a range is shown.\n")
L.append("| model | off: failures | inject: failures | inject: held calls | inject: attempted bad calls (range) |")
L.append("|---|--:|--:|--:|--:|")
lo_t = hi_t = 0
for m in models:
    o, i = C[(m, "off")], C[(m, "inject")]
    held = i["warnings_delivered"]
    fp_max = min(held, i["shadow"]["fp"])
    lo, hi = i["failures"] + held - fp_max, i["failures"] + held
    lo_t += lo; hi_t += hi
    L.append(f"| {m} | {o['failures']} | {i['failures']} | {held} | {lo}–{hi} |")
L.append(f"| **all** | {tot('off','failures')} | {tot('inject','failures')} | "
         f"{tot('inject','warnings_delivered')} | {lo_t}–{hi_t} |")
out["bad_attempts"] = (tot("off", "failures"), lo_t, hi_t)
L.append("")

# ---------- 5. where can a false-positive warning occur at all? ----------
L.append("## 5. Where can a false-positive warning occur? (exhaustive check on the testbed)\n")
L.append("For each scenario: replay the seed agent's failing call into the store, then try a grid "
         "of plausible calls from another agent and count calls the matcher warns on that the "
         "oracle says would succeed.\n")


def candidate_calls(sc):
    paths = ["/data/x.txt", "/tmp/x.txt", "/out/x.txt", "/tmp/b.txt", "/tmp/a.txt",
             "/data/report.csv", "/tmp/sub/x.txt", "/x.txt"]
    sizes = ["q", "Y" * 15, "Y" * 80, "Z" * 200]
    for p, c in itertools.product(paths, sizes):
        yield "write_file", {"path": p, "content": c}
    for p in paths + ["/tmp", "/data", "/out", "/"]:
        yield "read_file", {"path": p}
        yield "list_dir", {"path": p}
    for e in ["/render", "/render-batch", "render", "/other"]:
        yield "call_api", {"endpoint": e, "payload": "job9"}
        yield "call_api", {"endpoint": e}
    for t in ["accounts", "users", "orders"]:
        yield "query_db", {"table": t}
        yield "query_db", {"table": t, "query": f"SELECT * FROM {t}"}
    for p in [None, "/tmp/r.txt", "/data/r.txt"]:
        a = {"code": "print(1)"}
        if p:
            a.update(path=p, content="r")
        yield "run_code", a


fp_table = {}
for sc in SCENARIOS:
    world = World(**copy.deepcopy(sc["world"]))
    store = FailureStore(scope="episode")
    matcher = Matcher(level="mid", mode="subsumption")
    seed = sc["agents"][0]
    tool, args = seed["script"]["primary"]
    step = 1
    try:
        TOOLS[tool](world, **args)  # e.g. rate_limit: first call succeeds
        # run the second agent's primary to create the failure record
        seed = sc["agents"][1]
        tool, args = seed["script"]["primary"]
        step = 2
        TOOLS[tool](world, **args)
    except ToolError as e:
        store.add(extract(tool, args, world), tool, seed["id"], e.error_class, e.msg, "x", step)
    fps, tps, total = [], 0, 0
    for t, a in candidate_calls(sc):
        try:
            orc = would_fail(t, a, world)
        except TypeError:
            continue
        hits = matcher.check(extract(t, a, world), store, exclude_agent="OtherAgent")
        total += 1
        if hits and orc is None:
            fps.append((t, a.get("path") or a.get("table") or a.get("endpoint"),
                        len(str(a.get("content", "")))))
        elif hits:
            tps += 1
    fp_table[sc["name"]] = (tps, fps)
    ex = "; ".join(f"{t} {x} (content {n} chars)" if t == "write_file" else f"{t} {x}"
                   for t, x, n in fps[:3])
    L.append(f"- **{sc['name']}**: {tps} warned calls that would fail, **{len(fps)} warned calls that would succeed**"
             + (f" — e.g. {ex}" if fps else ""))
out["fp_feasible"] = {k: len(v[1]) for k, v in fp_table.items()}
L.append("")

open(os.path.join(OUT, "exp1_reanalysis.md"), "w", encoding="utf-8").write("\n".join(L))
json.dump(out, open(os.path.join(OUT, "exp1_reanalysis.json"), "w"), indent=1, default=str)
print("\n".join(L))
