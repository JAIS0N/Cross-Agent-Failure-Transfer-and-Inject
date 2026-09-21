"""Figure-1 demo (plan §2.6): the cross-agent transfer product, end to end.

  FileOps writes to /data  -> PermissionDenied -> stored
  Coder is about to write into /data           -> matcher fires at G-mid
  -> warning injected -> Coder writes to /tmp instead -> task succeeds.

Run:  python -m pfti.demo_figure1      (from the folder CONTAINING pfti/)
"""
from __future__ import annotations

import glob
import os

from .eval.runner import load_scenario, run_episode
from .eval.metrics import summarize, shadow_table

HERE = os.path.dirname(__file__)


def banner(t):
    print("\n" + "=" * 66 + f"\n{t}\n" + "=" * 66)


def show(result):
    for step, agent, what, detail in result["transcript"]:
        print(f"  step {step:>2}  {agent:<8} {what:<16} {detail}")
    print(f"  -> success={result['success']}  failures={result['failures']}"
          f"  repeated_peer_failures={result['repeated_peer_failures']}")


def main():
    scen = load_scenario(os.path.join(HERE, "envs/scenarios/01_perm_denied.yaml"))

    banner("A. VANILLA (hook off): agents step on the same rake")
    show(run_episode(scen, mode="off"))

    banner("B. PFTI (inject, G-mid, subsumption): peer failure transferred")
    r = run_episode(scen, mode="inject", level="mid")
    show(r)
    warn = next(d for _, _, w, d in r["transcript"] if w == "WARNED")
    print("\n  The injected warning (verbatim):\n")
    for line in warn.splitlines():
        print("   |", line)

    banner("C. G-FINE equality: nothing matches (different file) -> no help")
    show(run_episode(scen, mode="inject", level="fine", match_mode="equality"))

    banner("D. SHADOW MODE precision/recall vs oracle, all scenarios x levels")
    per_level = {}
    for level in ("coarse", "mid", "fine"):
        per_level[level] = [
            run_episode(load_scenario(p), mode="shadow", level=level)
            for p in sorted(glob.glob(os.path.join(HERE, "envs/scenarios/*.yaml")))]
    print(f"\n  {'level':<8}{'TP':>4}{'FP':>4}{'FN':>4}{'TN':>5}"
          f"{'precision':>11}{'recall':>9}")
    for row in shadow_table(per_level):
        p = f"{row['precision']:.2f}" if row["precision"] is not None else "  -"
        rc = f"{row['recall']:.2f}" if row["recall"] is not None else "  -"
        print(f"  {row['level']:<8}{row['TP']:>4}{row['FP']:>4}"
              f"{row['FN']:>4}{row['TN']:>5}{p:>11}{rc:>9}")

    banner("E. FULL SUITE: vanilla vs PFTI on all 5 scenarios")
    paths = sorted(glob.glob(os.path.join(HERE, "envs/scenarios/*.yaml")))
    for mode in ("off", "inject"):
        results = [run_episode(load_scenario(p), mode=mode) for p in paths]
        s = summarize(results)
        label = "vanilla" if mode == "off" else "PFTI(G-mid,subsumption)"
        print(f"  {label:<26} success={s['success_rate']:.0%}"
              f"  failures={s['total_failures']}"
              f"  repeated_peer_failures={s['repeated_peer_failures']}")


if __name__ == "__main__":
    main()
