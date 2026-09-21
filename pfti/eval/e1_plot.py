"""E1 figures (spec §4): recall/precision vs k (3 arms) + convergence."""
from __future__ import annotations

from .e1_unseen import run_e1


def make_plots(prefix="e1"):
    res = run_e1()
    ks = sorted(res)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing; results:", res)
        return res

    # Figure 1: recall & precision vs k, three arms
    fig, (axr, axp) = plt.subplots(1, 2, figsize=(10, 4))
    colors = {"A1": "#888", "A2": "#e67e22", "A3": "#27ae60"}
    labels = {"A1": "A1 instance-equality", "A2": "A2 fixed G-mid",
              "A3": "A3 learned (LGG)"}
    for arm in ("A1", "A2", "A3"):
        axr.plot(ks, [res[k][arm]["recall"] for k in ks], "-o",
                 color=colors[arm], label=labels[arm])
        axp.plot(ks, [res[k][arm]["precision"] for k in ks], "-o",
                 color=colors[arm], label=labels[arm])
    axr.set_title("Unseen-call recall vs k"); axr.set_xlabel("k (seed failures)")
    axr.set_ylabel("recall"); axr.set_ylim(-0.05, 1.05); axr.grid(alpha=.3)
    axr.legend(fontsize=8)
    axp.set_title("Unseen-call precision vs k\n(near-miss hard negatives)")
    axp.set_xlabel("k (seed failures)"); axp.set_ylabel("precision")
    axp.set_ylim(0.5, 1.05); axp.grid(alpha=.3); axp.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(f"{prefix}_recall_precision.png", dpi=130)

    # Figure 2: convergence |gsig D P*| vs k
    fig2, ax = plt.subplots(figsize=(5.5, 4))
    ax.plot(ks, [res[k]["gsig_symdiff"] for k in ks], "-o", color="#2980b9")
    ax.set_title("Convergence to the true precondition\n|gsig Δ P*| vs k")
    ax.set_xlabel("k (seed failures)"); ax.set_ylabel("|gsig Δ P*|")
    ax.grid(alpha=.3); ax.set_ylim(bottom=-0.02)
    fig2.tight_layout(); fig2.savefig(f"{prefix}_convergence.png", dpi=130)
    print(f"saved {prefix}_recall_precision.png and {prefix}_convergence.png")
    return res


if __name__ == "__main__":
    make_plots()
