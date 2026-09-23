"""Who&When study, redesigned after review: reproduction-filtered, sampled,
peer-warning test.

The first replay (whoandwhen_llm.py) could not detect an effect: the
baseline reproduced the historical error only 8.5% of the time, the
"matching" earlier failure usually had a different error class, and it was
almost always still in the model's context. This design fixes all three:

  Pool      Every moment in the 184 logs where an agent proposed Python code
            and that code actually failed when the original run executed it
            (~90 moments across ~69 logs, vs 36 / 25 before).

  Phase 1   SCREEN. Replay the real conversation up to the proposal and
            sample the agent's turn k_screen times under `off` at
            temperature > 0. Execute every sample. A case is kept for a
            model when it reproduces the historical error class at least
            `min_repro` times. This is where repeated sampling earns its
            keep: it estimates each case's reproduction rate.

  Phase 2   TEST. On the kept cases only, draw FRESH samples -- k_test under
            `off` and k_test under `inject` -- so the off rate is not
            inflated by the selection in phase 1 (winner's curse).

  Warning   `inject` adds one PFTI peer-failure warning built from the
            historical failure at this very step, attributed to a parallel
            peer agent working the same task. It names the real error class
            and message. It is never in the replayed context (that failure
            happens after the proposal), and it comes from a different
            agent -- so it tests peer transfer, not in-context memory.

  Question  Given that a model is likely to repeat a peer's failure, does the
            warning stop it?

Results are appended to <out>/results.jsonl one sample at a time, so an
interrupted run resumes where it stopped. Analysis: waw_repro_analyze.py.

SAFETY: executes short model-generated Python snippets locally in a temp
dir with a timeout; code that deletes files, spawns shells or changes system
state is never executed (same denylist as whoandwhen_llm.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time

from ..core.inject import TEMPLATE
from .whoandwhen import load_logs, step_signature
from . import whoandwhen_llm as W

PEER_NAME = "ParallelPeer"


# --------------------------------------------------------------------------
# case pool
# --------------------------------------------------------------------------

def _error_excerpt(content: str, ec: str, n: int = 200) -> str:
    lines = [l.strip() for l in str(content).splitlines() if l.strip()]
    hit = [l for l in lines if ec and ec in l]
    line = hit[-1] if hit else (lines[-1] if lines else "")
    return line[:n]


def find_repro_cases(max_lookback: int = 2):
    """All moments where an agent's python code block (within `max_lookback`
    turns before) was executed and failed in the original run."""
    cases = []
    for path, d in load_logs():
        hist = d.get("history", [])
        sysprompts = d.get("system_prompt", {}) or {}
        for i, step in enumerate(hist):
            res = step_signature(step)
            if res is None:
                continue
            sig, ec = res
            if dict(sig.preds).get("tool") != "code_exec":
                continue
            j = None
            for jj in range(i - 1, max(-1, i - 1 - max_lookback), -1):
                code = W.extract_code(hist[jj].get("content", ""))
                if code:
                    j = jj if code[0] == "python" else None
                    break
            if j is None:
                continue
            log = os.path.basename(path)
            subset = "hand" if "Hand-Crafted" in path else "algo"
            cases.append({
                "case_id": f"{subset}/{log}:{i}",
                "log": f"{subset}/{log}",
                "log_path": path,
                "history": hist,
                "system_prompts": sysprompts if isinstance(sysprompts, dict) else {},
                "code_index": j,
                "code_agent": hist[j].get("name", "?"),
                "later_index": i,
                "later_agent": step.get("name", "?"),
                "later_ec": ec,
                "later_sig": sig,
                "error_msg": _error_excerpt(step.get("content", ""), ec),
                "hist_code": W.extract_code(hist[j].get("content", ""))[1][:1500],
            })
    return cases


# --------------------------------------------------------------------------
# warning + prompts
# --------------------------------------------------------------------------

def peer_warning(case: dict) -> str:
    """The standard PFTI template, filled from the historical failure at
    this step, attributed to a parallel peer. The pattern shows tool and
    error class (the Who&When dimensions are not in signature.DIM_LEVEL, so
    Signature.project() would reduce it to tool only)."""
    preds = dict(case["later_sig"].preds)
    pat = [f"tool={preds.get('tool', 'code_exec')}",
           f"error_class={str(case['later_ec']).lower()}"]
    if "missing_module" in preds:
        pat.append(f"missing_module={preds['missing_module']}")
    return TEMPLATE.format(peer=PEER_NAME, tool="code_exec",
                           sig="(" + ", ".join(pat) + ")",
                           step=case["code_index"],
                           error_class=case["later_ec"],
                           msg=case["error_msg"])


def build_messages(case: dict, arm: str):
    msgs = W.build_messages(case, warning_rec=None)
    if arm == "inject":
        msgs.insert(len(msgs) - 1, {"role": "user",
                                    "content": peer_warning(case)})
    return msgs


def prior_failure_in_context(case: dict) -> bool:
    """Is a failure of the SAME error class already visible in the replayed
    context? (moderator for the analysis)"""
    msgs = W.build_messages(case, warning_rec=None)
    ec = str(case["later_ec"])
    for m in msgs[1:-1]:
        body = m["content"]
        if ec in body and ("exitcode" in body or "Traceback" in body
                           or "Error" in body):
            return True
    return False


# --------------------------------------------------------------------------
# one sample
# --------------------------------------------------------------------------

def _fake_response(case: dict, arm: str, sample: int, seed: int) -> str:
    """Offline stand-in. Each case gets a fixed 'propensity' to repeat its
    historical error (derived from its id), so screening has something to
    find; inject cuts the propensity to a quarter."""
    h = int(hashlib.sha1(case["case_id"].encode()).hexdigest(), 16)
    p = (h % 100) / 100.0
    if arm == "inject":
        p *= 0.25
    rng = random.Random(f"{case['case_id']}|{arm}|{sample}|{seed}")
    warned = rng.random() >= p
    return W._fake_turn(case["later_ec"], warned)


def run_sample(case, client, model, arm, temperature, sample, seed,
               use_fake=False, max_tokens=700, timeout=12) -> dict:
    msgs = build_messages(case, arm)
    t0 = time.time()
    if use_fake:
        text = _fake_response(case, arm, sample, seed)
    else:
        resp = client.chat.completions.create(
            model=model, messages=msgs, temperature=temperature,
            max_tokens=max_tokens, seed=seed * 1000 + sample)
        text = resp.choices[0].message.content or ""
    row = {"status": None, "failed": None, "same_error": None,
           "gen_s": round(time.time() - t0, 2),
           "mentions_warning": bool(W._ACK.search(text)) if arm == "inject" else None}
    ex = W.extract_code(text)
    if ex is None:
        row["status"] = "no_code"
    elif ex[0] != "python":
        row["status"] = "shell"
    elif W.is_unsafe(ex[1]):
        row["status"] = "unsafe"
        row["code"] = ex[1][:400]
    else:
        out = W.execute_code("python", ex[1], timeout=timeout)
        failed, same = W.grade_execution(out, case["later_agent"],
                                         case["later_sig"])
        row.update(status="graded", failed=failed, same_error=same,
                   code=ex[1][:400], exec=out[:300])
    return row


# --------------------------------------------------------------------------
# driver (resumable)
# --------------------------------------------------------------------------

def _load_done(path):
    done = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue  # partial last line from an interrupted run
                done[(r["phase"], r["model"], r["case_id"], r["arm"],
                      r["sample"])] = r
    return done


def run(models, out_dir, k_screen=6, k_test=6, min_repro=2,
        temperature=0.7, seed=1, max_cases=None, client=None,
        use_fake=False, timeout=12, verbose=True):
    os.makedirs(out_dir, exist_ok=True)
    res_path = os.path.join(out_dir, "results.jsonl")
    done = _load_done(res_path)
    cases = find_repro_cases()
    if max_cases:
        cases = cases[:max_cases]
    ctx = {c["case_id"]: prior_failure_in_context(c) for c in cases}
    meta = {"models": models, "k_screen": k_screen, "k_test": k_test,
            "min_repro": min_repro, "temperature": temperature, "seed": seed,
            "n_cases": len(cases), "n_logs": len({c["log"] for c in cases}),
            "fake": use_fake,
            "cases": [{"case_id": c["case_id"], "log": c["log"],
                       "later_ec": c["later_ec"], "code_agent": c["code_agent"],
                       "same_class_failure_in_context": ctx[c["case_id"]]}
                      for c in cases],
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)

    fh = open(res_path, "a", encoding="utf-8")

    def do(phase, model, case, arm, s):
        key = (phase, model, case["case_id"], arm, s)
        if key in done:
            return done[key]
        try:
            r = run_sample(case, client, model, arm, temperature, s, seed,
                           use_fake=use_fake, timeout=timeout)
        except Exception as e:  # server hiccup: record, keep going
            r = {"status": "error", "error": f"{type(e).__name__}: {e}"[:300],
                 "failed": None, "same_error": None}
        r.update(phase=phase, model=model, case_id=case["case_id"],
                 log=case["log"], later_ec=case["later_ec"], arm=arm,
                 sample=s)
        if r["status"] != "error":  # errors are retried on resume
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done[key] = r
        return r

    t_start = time.time()
    n_calls = 0
    for model in models:
        # ---- phase 1: screen ----
        kept = []
        for ci, case in enumerate(cases):
            hits = 0
            for s in range(k_screen):
                r = do("screen", model, case, "off", s)
                n_calls += 1
                hits += int(bool(r.get("same_error")))
            if hits >= min_repro:
                kept.append(case)
            if verbose:
                el = time.time() - t_start
                print(f"  [{model}] screen {ci + 1}/{len(cases)} "
                      f"{case['case_id']:<28} repro {hits}/{k_screen}"
                      f"{'  KEEP' if hits >= min_repro else ''}"
                      f"   ({el / 60:.0f} min elapsed)", flush=True)
        if verbose:
            print(f"  [{model}] kept {len(kept)}/{len(cases)} cases "
                  f"(repro >= {min_repro}/{k_screen})", flush=True)
        # ---- phase 2: fresh off vs inject on kept cases ----
        for ci, case in enumerate(kept):
            tally = {"off": 0, "inject": 0}
            for s in range(k_test):
                for arm in ("off", "inject"):   # interleaved: same conditions
                    r = do("test", model, case, arm, s)
                    n_calls += 1
                    tally[arm] += int(bool(r.get("same_error")))
            if verbose:
                print(f"  [{model}] test {ci + 1}/{len(kept)} "
                      f"{case['case_id']:<28} repeat off {tally['off']}/{k_test}"
                      f"  inject {tally['inject']}/{k_test}", flush=True)
    fh.close()
    return {"n_cases": len(cases), "calls": n_calls,
            "seconds": round(time.time() - t_start, 1)}


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", default="qwen2.5:3b,qwen2.5:7b,qwen2.5:14b,llama3.1:8b")
    ap.add_argument("--out", default="waw_repro_results")
    ap.add_argument("--k-screen", type=int, default=6)
    ap.add_argument("--k-test", type=int, default=6)
    ap.add_argument("--min-repro", type=int, default=2)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--max-cases", type=int, default=None)
    ap.add_argument("--timeout", type=int, default=12)
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--list-cases", action="store_true",
                    help="print the case pool and exit (no model calls)")
    a = ap.parse_args(argv)
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    if a.list_cases:
        cs = find_repro_cases()
        from collections import Counter
        print(f"{len(cs)} cases from {len({c['log'] for c in cs})} logs")
        for ec, n in Counter(c["later_ec"] for c in cs).most_common():
            print(f"  {ec:<22} {n}")
        return 0

    client = None
    if not a.fake:
        from ..bench.providers import build_openai_client
        client = build_openai_client()
    print(f"Who&When reproduction-filtered study: models={models} "
          f"k_screen={a.k_screen} k_test={a.k_test} min_repro={a.min_repro} "
          f"T={a.temperature}", flush=True)
    r = run(models, a.out, k_screen=a.k_screen, k_test=a.k_test,
            min_repro=a.min_repro, temperature=a.temperature, seed=a.seed,
            max_cases=a.max_cases, client=client, use_fake=a.fake,
            timeout=a.timeout)
    print(f"done: {r}", flush=True)
    try:
        from . import waw_repro_analyze
        waw_repro_analyze.main([a.out])
    except Exception as e:
        print(f"(analysis skipped: {type(e).__name__}: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
