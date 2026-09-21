"""CLI for the Who&When real-LLM before/after study.

    python -m pfti.eval.run_whoandwhen_llm                 # real Ollama
    python -m pfti.eval.run_whoandwhen_llm --fake          # offline check
    python -m pfti.eval.run_whoandwhen_llm --models qwen2.5:7b --limit 5

Writes to waw_results/: waw_llm.json (every case, both conditions, the
regenerated code and its real execution output), summary.md, and
waw_before_after.png.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from .whoandwhen import DATA
from .whoandwhen_llm import run_study

DEFAULT_MODELS = ["qwen2.5:3b", "qwen2.5:7b", "qwen2.5:14b", "llama3.1:8b"]


def _pct(v):
    return "n/a" if v is None else f"{v:.0%}"


def write_summary(out: Path, result: dict, meta: dict):
    L = [f"# Who&When real-LLM before/after "
         f"({'offline fake' if meta['fake'] else 'real Ollama'})\n",
         f"_{meta['timestamp']} - {result['n_cases']} real code-execution "
         f"transfer-opportunity cases from {meta['n_logs']} Who&When logs - "
         f"repeats={result['repeats']} - {meta['seconds']}s_\n",
         "\nEach case: the real conversation is replayed up to the agent turn "
         "that proposed the failing code; the model regenerates that turn "
         "with no warning (off) and with a PFTI peer-failure warning built "
         "from the earlier real failure (inject). Both code blocks are "
         "actually executed and graded against the real historical error.\n",
         "\n| model | fail rate off -> inject | same-error rate off -> inject "
         "| code parsed off / inject | engaged with warning |",
         "|---|---|---|---|---|"]
    for s in result["summaries"]:
        L.append(f"| {s['model']} | {_pct(s['off_fail_rate'])} -> "
                 f"{_pct(s['inject_fail_rate'])} | "
                 f"{_pct(s['off_same_error_rate'])} -> "
                 f"{_pct(s['inject_same_error_rate'])} | "
                 f"{s['off_parsed']}/{s['off_n']} / "
                 f"{s['inject_parsed']}/{s['inject_n']} | "
                 f"{_pct(s['inject_ack_rate'])} |")
    L.append("\nUnsafe snippets skipped (never executed; denylist for file "
             "deletion / shells / system changes): " + ", ".join(
                 f"{s['model']} off={s['off_unsafe_skipped']} "
                 f"inject={s['inject_unsafe_skipped']}"
                 for s in result["summaries"]) + "\n")
    L += ["\n**fail rate**: regenerated code failed at all when executed. "
          "**same-error rate**: failed with the same error class as the real "
          "historical failure (a true repeat). Lower is better for both.\n",
          "\nCaveat: the original tool environment (downloaded files, web "
          "state) is not reproduced, so some regenerated code fails for "
          "environment reasons in BOTH conditions; the off vs inject "
          "difference is the comparison, not the absolute rate.\n"]
    (out / "summary.md").write_text("\n".join(L), encoding="utf-8")


def write_figure(out: Path, result: dict, meta: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    S = result["summaries"]
    models = [s["model"] for s in S]
    x = range(len(models))
    w = 0.38
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, key, title in (
            (axes[0], "fail_rate", "Regenerated code fails (any error)"),
            (axes[1], "same_error_rate", "Repeats the SAME real error")):
        off = [s[f"off_{key}"] or 0 for s in S]
        inj = [s[f"inject_{key}"] or 0 for s in S]
        ax.bar([i - w / 2 for i in x], off, w, label="vanilla (off)",
               color="#999999")
        ax.bar([i + w / 2 for i in x], inj, w, label="PFTI (inject)",
               color="#0072B2")
        for i, (a, b) in enumerate(zip(off, inj)):
            ax.text(i - w / 2, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
            ax.text(i + w / 2, b + 0.02, f"{b:.0%}", ha="center", fontsize=9)
        ax.set_ylim(0, 1.1)
        ax.set_title(title)
        ax.set_xticks(list(x))
        ax.set_xticklabels(models, rotation=20, ha="right")
        ax.legend(frameon=False)
    axes[0].set_ylabel("rate (lower is better)")
    fig.suptitle(f"Who&When real failures, {result['n_cases']} code-exec cases "
                 f"({'offline fake' if meta['fake'] else 'real Ollama'})",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(out / "waw_before_after.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true")
    ap.add_argument("--models", default=None)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N cases (quick smoke test)")
    ap.add_argument("--out", default="waw_results")
    args = ap.parse_args(argv)

    if not os.path.isdir(DATA):
        raise SystemExit(f"Who&When data not found at {DATA} "
                         "(set WAW_DIR or keep whoandwhen_data/ next to pfti/)")
    models = ([m.strip() for m in args.models.split(",")] if args.models
              else (["fake"] if args.fake else DEFAULT_MODELS))
    client = None
    if not args.fake:
        from ..bench.providers import build_openai_client
        client = build_openai_client()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    print(f"\n{'=' * 70}\nWho&When real-LLM before/after  "
          f"({'FAKE offline' if args.fake else 'REAL Ollama'})\n"
          f"models: {', '.join(models)}\n{'=' * 70}")
    t0 = time.time()
    result = run_study(models, client=client, use_fake=args.fake,
                       limit=args.limit, repeats=args.repeats)
    n_logs = sum(len(os.listdir(os.path.join(DATA, d)))
                 for d in ("Algorithm-Generated", "Hand-Crafted")
                 if os.path.isdir(os.path.join(DATA, d)))
    meta = {"fake": args.fake, "models": models, "repeats": args.repeats,
            "n_logs": n_logs, "seconds": round(time.time() - t0, 1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "waw_llm.json").write_text(
        json.dumps({"meta": meta, **result}, indent=2, default=str),
        encoding="utf-8")
    write_summary(out, result, meta)
    try:
        write_figure(out, result, meta)
    except Exception as e:
        print(f"(figure skipped: {e})")
    print(f"\nWrote {out}/ ({meta['seconds']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
