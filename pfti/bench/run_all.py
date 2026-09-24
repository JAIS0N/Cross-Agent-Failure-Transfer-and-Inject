"""One entrypoint for the whole before/after benchmark.

    python -m pfti.bench.run_all                # real Ollama models
    python -m pfti.bench.run_all --fake         # offline, deterministic
    PFTI_MODELS="qwen2.5:3b,llama3.1:8b" python -m pfti.bench.run_all

Writes JSON to bench_results/ (core_matrix.json, routing.json, meta.json),
then, if matplotlib is present, renders figures + a markdown summary via
pfti.bench.report.

Each real model x mode cell is independent; if a model isn't pulled the
cell is recorded with an error and the run continues.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from .providers import MODEL_MATRIX
from .core_matrix import run_matrix, run_scenario, shadow_prf
from .core_matrix import TraceLogger
from .scenarios_llm import SCENARIOS, SCENARIO_SETS
from . import routing_matrix as RM


def _default_models():
    env = os.environ.get("PFTI_MODELS")
    if env:
        return [m.strip() for m in env.split(",") if m.strip()]
    return list(MODEL_MATRIX.keys())


SUM_KEYS = ("failures", "repeated_peer_failures", "warnings_delivered",
            "ignored_warnings", "total_tokens", "held_tp", "held_fp",
            "doomed_attempts", "peer_doomed_attempts", "peer_doomed_executed",
            "text_calls")


def _run_core(models, use_fake, client, level, repeats, scenarios=None,
              trace_dir=None, modes=("off", "shadow", "inject"),
              parse_text_calls=False):
    """Run the core matrix, repeating each cell `repeats` times.

    Per-scenario and per-repeat results are kept (not just success counts)
    so every metric's run-to-run spread can be reported. With `trace_dir`,
    each scenario run's full event log and agent transcripts are saved."""
    scenarios = scenarios or SCENARIOS
    cells = []
    for model in models:
        for mode in modes:
            acc = {"model": model, "mode": mode, "n_success": 0,
                   "n_scenarios": 0, "error": None}
            for k in SUM_KEYS:
                acc[k] = 0
            events_all = []
            per = {}
            runs = []
            reps_done = 0
            for rep in range(repeats):
                try:
                    for sc in scenarios:
                        r = run_scenario(sc, mode, client, model, level=level,
                                         use_fake=use_fake,
                                         logger=TraceLogger(),
                                         keep_transcripts=trace_dir is not None,
                                         parse_text_calls=parse_text_calls)
                        acc["n_scenarios"] += 1
                        acc["n_success"] += int(r["success"])
                        for k in SUM_KEYS:
                            acc[k] += r[k]
                        events_all += r["events"]
                        d = per.setdefault(sc["name"], {"success": 0, "n": 0,
                                                        "kind": sc.get("kind")})
                        d["success"] += int(r["success"])
                        d["n"] += 1
                        for k in SUM_KEYS:
                            d[k] = d.get(k, 0) + r[k]
                        runs.append({"scenario": sc["name"], "repeat": rep,
                                     "success": r["success"],
                                     **{k: r[k] for k in SUM_KEYS}})
                        if trace_dir is not None:
                            safe = model.replace(":", "_").replace("/", "_")
                            fn = Path(trace_dir) / (
                                f"{safe}__{mode}__{sc['name']}__r{rep}.json")
                            fn.write_text(json.dumps(
                                {"model": model, "mode": mode,
                                 "scenario": sc["name"], "repeat": rep,
                                 "success": r["success"],
                                 **{k: r[k] for k in SUM_KEYS},
                                 "events": r["events"],
                                 "transcripts": r["transcripts"]},
                                indent=1, default=str), encoding="utf-8")
                    reps_done += 1
                except Exception as e:  # model not pulled / server down
                    acc["error"] = f"{type(e).__name__}: {e}"
                    break
            n = max(1, acc["n_scenarios"])
            acc["success_rate"] = acc["n_success"] / n
            acc["shadow"] = shadow_prf(events_all)
            acc["repeats"] = reps_done
            acc["per_scenario"] = per
            acc["runs"] = runs
            cells.append(acc)
            sh = acc["shadow"]
            tag = " [ERR]" if acc["error"] else ""
            print(f"  {model:<12} {mode:<7} succ={acc['success_rate']:>5.0%} "
                  f"fails={acc['failures']:>3} peer={acc['repeated_peer_failures']:>3} "
                  f"warn={acc['warnings_delivered']:>3} "
                  f"held tp/fp={acc['held_tp']}/{acc['held_fp']} "
                  f"textcalls={acc['text_calls']} "
                  f"ign={acc['ignored_warnings']:>3} "
                  f"tok={acc['total_tokens']:>7} "
                  f"P/R={sh['precision']:.2f}/{sh['recall']:.2f}{tag}",
                  flush=True)
    return cells


def _run_routing(models_tags, use_fake):
    try:
        models = (RM.build_fake_models(models_tags) if use_fake
                  else RM.build_real_models(models_tags))
        rows = RM.run_routing(models)
        for r in rows:
            print(f"  {r['policy']:<13} succ={r['success_rate']:>5.0%} "
                  f"cost={r['total_cost']:>6.1f} lat={r['total_latency_s']:>6.1f}s")
        return {"rows": rows, "error": None}
    except Exception as e:
        print(f"  routing skipped: {e}")
        return {"rows": [], "error": f"{type(e).__name__}: {e}"}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true",
                    help="use deterministic offline client (no Ollama)")
    ap.add_argument("--models", default=None,
                    help="comma-separated Ollama tags (overrides matrix)")
    ap.add_argument("--level", default="mid")
    ap.add_argument("--repeats", type=int, default=1,
                    help="repeat each core cell N times and average")
    ap.add_argument("--out", default="bench_results")
    ap.add_argument("--scenarios", default="v1", choices=sorted(SCENARIO_SETS),
                    help="v1 = original 5; v2 = adversarial-FP + reasoning; all")
    ap.add_argument("--save-traces", action="store_true",
                    help="save every scenario run's events + transcripts")
    ap.add_argument("--no-routing", action="store_true")
    ap.add_argument("--parse-text-calls", action="store_true",
                    help="also execute tool calls a model writes as JSON text "
                         "(llama3.1 does this after warnings); same in every mode")
    args = ap.parse_args(argv)
    scen = SCENARIO_SETS[args.scenarios]

    models = ([m.strip() for m in args.models.split(",")] if args.models
              else _default_models())
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    client = None
    if args.fake:
        from .providers import FakeLLM
        client = FakeLLM()
    else:
        from .providers import build_openai_client
        try:
            client = build_openai_client()
        except Exception as e:
            print(f"Could not build LLM client ({e}); falling back to --fake.")
            from .providers import FakeLLM
            client = FakeLLM()
            args.fake = True

    print(f"\n{'='*70}\nPFTI before/after benchmark  "
          f"({'FAKE offline' if args.fake else 'REAL Ollama'})\n"
          f"models: {', '.join(models)}\n{'='*70}")

    print("\n[1/2] CORE  vanilla(off) vs shadow vs PFTI(inject)")
    t0 = time.time()
    trace_dir = None
    if args.save_traces:
        trace_dir = out / "traces"
        trace_dir.mkdir(exist_ok=True)
    core = _run_core(models, args.fake, client, args.level, args.repeats,
                     scenarios=scen, trace_dir=trace_dir,
                     parse_text_calls=args.parse_text_calls)

    if args.no_routing:
        routing = {"rows": [], "error": "skipped (--no-routing)"}
    else:
        print("\n[2/2] ROUTING  always-<model> vs PFTI-routed")
        routing = _run_routing(models, args.fake)

    meta = {"fake": args.fake, "models": models, "level": args.level,
            "repeats": args.repeats, "seconds": round(time.time() - t0, 1),
            "scenario_set": args.scenarios,
            "scenarios": [s["name"] for s in scen],
            "save_traces": args.save_traces,
            "parse_text_calls": args.parse_text_calls,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    (out / "core_matrix.json").write_text(json.dumps(core, indent=2),
                                          encoding="utf-8")
    (out / "routing.json").write_text(json.dumps(routing, indent=2),
                                      encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, indent=2),
                                   encoding="utf-8")
    print(f"\nWrote JSON to {out}/  ({meta['seconds']}s)")

    try:
        from . import report
        report.build(str(out))
        print(f"Wrote figures + summary to {out}/")
    except Exception as e:
        print(f"(report/plots skipped: {e})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
