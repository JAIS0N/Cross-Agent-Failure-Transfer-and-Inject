"""Turn bench_results/*.json into figures + a markdown summary.

Figures (matplotlib, colorblind-safe Okabe-Ito palette):
  * core_before_after.png : success rate and repeated-peer-failures per
      model, vanilla(off) vs PFTI(inject) -- the headline before/after.
  * core_shadow_pr.png    : matcher precision/recall vs the oracle (shadow).
  * routing_tradeoff.png  : cost vs success, baselines vs PFTI-routed.
And summary.md with the full model x mode matrix.
"""
from __future__ import annotations

import json
from pathlib import Path

# Okabe-Ito colorblind-safe
C_OFF = "#999999"     # grey  = vanilla / before
C_INJECT = "#0072B2"  # blue  = PFTI / after
C_SHADOW = "#009E73"  # green
C_ACC = "#D55E00"     # vermillion


def _load(out):
    out = Path(out)
    core = json.loads((out / "core_matrix.json").read_text())
    routing = json.loads((out / "routing.json").read_text())
    meta = json.loads((out / "meta.json").read_text())
    return core, routing, meta


def _cell(core, model, mode):
    for c in core:
        if c["model"] == model and c["mode"] == mode:
            return c
    return None


def build(out="bench_results"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    core, routing, meta = _load(out)
    outp = Path(out)
    models = meta["models"]
    plt.rcParams.update({"font.size": 11})

    # ---- Figure 1: before/after success + repeated-peer ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    x = range(len(models))
    w = 0.38
    off_succ = [(_cell(core, m, "off") or {}).get("success_rate", 0)
                for m in models]
    inj_succ = [(_cell(core, m, "inject") or {}).get("success_rate", 0)
                for m in models]
    ax1.bar([i - w/2 for i in x], off_succ, w, label="vanilla (off)",
            color=C_OFF)
    ax1.bar([i + w/2 for i in x], inj_succ, w, label="PFTI (inject)",
            color=C_INJECT)
    ax1.set_ylabel("task success rate")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Task success: vanilla vs PFTI")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(models, rotation=20, ha="right")
    ax1.legend(frameon=False)
    for i, (a, b) in enumerate(zip(off_succ, inj_succ)):
        ax1.text(i - w/2, a + 0.02, f"{a:.0%}", ha="center", fontsize=9)
        ax1.text(i + w/2, b + 0.02, f"{b:.0%}", ha="center", fontsize=9)

    off_pf = [(_cell(core, m, "off") or {}).get("repeated_peer_failures", 0)
              for m in models]
    inj_pf = [(_cell(core, m, "inject") or {}).get("repeated_peer_failures", 0)
              for m in models]
    ax2.bar([i - w/2 for i in x], off_pf, w, label="vanilla (off)", color=C_OFF)
    ax2.bar([i + w/2 for i in x], inj_pf, w, label="PFTI (inject)",
            color=C_INJECT)
    ax2.set_ylabel("repeated peer failures")
    ax2.set_title("Repeated peer failures: vanilla vs PFTI")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(models, rotation=20, ha="right")
    ax2.legend(frameon=False)
    fig.suptitle("PFTI core before/after  "
                 f"({'offline fake' if meta['fake'] else 'real Ollama models'})",
                 y=1.02, fontsize=13)
    fig.tight_layout()
    fig.savefig(outp / "core_before_after.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 2: shadow precision/recall ----
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    prec = [(_cell(core, m, "shadow") or {}).get("shadow", {}).get(
        "precision", 0) for m in models]
    rec = [(_cell(core, m, "shadow") or {}).get("shadow", {}).get("recall", 0)
           for m in models]
    ax.bar([i - w/2 for i in x], prec, w, label="precision", color=C_SHADOW)
    ax.bar([i + w/2 for i in x], rec, w, label="recall", color=C_ACC)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("shadow-mode score vs oracle")
    ax.set_title("Matcher accuracy (shadow mode, G-mid)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(outp / "core_shadow_pr.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # ---- Figure 3: routing cost/accuracy tradeoff ----
    rows = routing.get("rows", [])
    if rows:
        fig, ax = plt.subplots(figsize=(7.5, 5))
        for r in rows:
            routed = r["policy"] == "routed"
            ax.scatter(r["total_cost"], r["success_rate"],
                       s=180 if routed else 90,
                       color=C_INJECT if routed else C_OFF,
                       edgecolor="black", zorder=3,
                       marker="*" if routed else "o")
            ax.annotate(r["policy"], (r["total_cost"], r["success_rate"]),
                        textcoords="offset points", xytext=(8, 6), fontsize=9)
        ax.set_xlabel("total cost (model-size proxy)")
        ax.set_ylabel("success rate")
        ax.set_title("Routing: baselines vs PFTI-routed (up-and-left is better)")
        fig.tight_layout()
        fig.savefig(outp / "routing_tradeoff.png", dpi=130,
                    bbox_inches="tight")
        plt.close(fig)

    _write_summary(outp, core, routing, meta, models)


def _write_summary(outp, core, routing, meta, models):
    L = []
    L.append(f"# PFTI before/after results "
             f"({'offline fake' if meta['fake'] else 'real Ollama'})\n")
    L.append(f"_Generated {meta['timestamp']} · models: "
             f"{', '.join(models)} · {meta['seconds']}s_\n")

    L.append("\n## Core matrix (model × mode)\n")
    L.append("| model | mode | success | failures | repeat-peer | warns | "
             "ignored | tokens | shadow P/R |")
    L.append("|---|---|--:|--:|--:|--:|--:|--:|--:|")
    for m in models:
        for mode in ("off", "shadow", "inject"):
            c = _cell(core, m, mode)
            if not c:
                continue
            sh = c.get("shadow", {})
            err = " ⚠" if c.get("error") else ""
            L.append(f"| {m}{err} | {mode} | {c['success_rate']:.0%} | "
                     f"{c['failures']} | {c['repeated_peer_failures']} | "
                     f"{c['warnings_delivered']} | {c['ignored_warnings']} | "
                     f"{c['total_tokens']} | "
                     f"{sh.get('precision',0):.2f}/{sh.get('recall',0):.2f} |")

    # headline deltas
    L.append("\n## Headline: vanilla (off) → PFTI (inject)\n")
    L.append("| model | success off→inject | repeat-peer off→inject | "
             "extra tokens |")
    L.append("|---|--:|--:|--:|")
    for m in models:
        o, i = _cell(core, m, "off"), _cell(core, m, "inject")
        if not (o and i):
            continue
        dt = i["total_tokens"] - o["total_tokens"]
        pct = (dt / o["total_tokens"] * 100) if o["total_tokens"] else 0
        L.append(f"| {m} | {o['success_rate']:.0%} → {i['success_rate']:.0%} | "
                 f"{o['repeated_peer_failures']} → "
                 f"{i['repeated_peer_failures']} | +{dt} ({pct:+.0f}%) |")

    rows = routing.get("rows", [])
    if rows:
        L.append("\n## Routing (always-model vs PFTI-routed)\n")
        L.append("| policy | success | cost | latency (s) |")
        L.append("|---|--:|--:|--:|")
        for r in rows:
            L.append(f"| {r['policy']} | {r['success_rate']:.0%} | "
                     f"{r['total_cost']:.1f} | {r['total_latency_s']:.1f} |")

    L.append("\n## Figures\n")
    L.append("- `core_before_after.png` — success & repeated-peer, off vs "
             "inject\n- `core_shadow_pr.png` — matcher precision/recall vs "
             "oracle\n- `routing_tradeoff.png` — cost/accuracy, routed vs "
             "baselines\n")
    (outp / "summary.md").write_text("\n".join(L), encoding="utf-8")


if __name__ == "__main__":
    import sys
    build(sys.argv[1] if len(sys.argv) > 1 else "bench_results")
