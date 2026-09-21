"""Cost-vs-accuracy plot (the money figure). Saves routing_tradeoff.png."""
from __future__ import annotations

import statistics as st

from .experiment import run_all


def make_plot(out="routing_tradeoff.png", seeds=5, n=40):
    agg = {p: {"s": [], "c": []} for p in ("small", "large", "routed")}
    for seed in range(seeds):
        r = run_all(n, seed=seed)
        for p in agg:
            agg[p]["s"].append(r[p]["success_rate"])
            agg[p]["c"].append(r[p]["total_cost"])
    pts = {p: (st.mean(agg[p]["c"]), st.mean(agg[p]["s"])) for p in agg}

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; data points:", pts)
        return pts
    fig, ax = plt.subplots(figsize=(5.5, 4))
    colors = {"small": "#888", "large": "#c0392b", "routed": "#27ae60"}
    labels = {"small": "always-small", "large": "always-large",
              "routed": "PFTI-routed"}
    for p, (c, s) in pts.items():
        ax.scatter(c, s * 100, s=180, color=colors[p], zorder=3)
        ax.annotate(labels[p], (c, s * 100), xytext=(8, 6),
                    textcoords="offset points", fontsize=10)
    ax.set_xlabel("Total cost (lower is better)")
    ax.set_ylabel("Task success rate (%)")
    ax.set_title("Predicate-guided routing:\nnear-large accuracy at low cost")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"saved {out}  points={pts}")
    return pts


if __name__ == "__main__":
    make_plot()
