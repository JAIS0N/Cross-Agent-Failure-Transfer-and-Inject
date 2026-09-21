"""E1: Generalization to Unseen Calls (spec §4) -- the acceptance experiment.

Question: does the learned layer warn correctly on concrete calls that
never appeared as instances?

Pure store/matcher evaluation -- no agents, no LLM, fully reproducible.
For each of 5 failure classes we control the true precondition P* and the
oracle is simply: would_fail(call) == (P* subset of call.preds).

Three arms:
  A1  instance-equality only (old mechanism)      -> unseen-recall ~ 0
  A2  fixed G-mid subsumption (old best config)
  A3  learned layer (LGG + guards)

Metrics vs k (number of seed failures shown): unseen-recall, unseen-
precision (against near-miss hard negatives), and convergence |gsig D P*|.
"""
from __future__ import annotations

import random

from ..core.signature import Signature
from ..core.generalize import LearningLayer

# --- local level map so we can design clean classes (mid vs fine dims) ---
FINE_DIMS = {"size", "key", "path", "reqid"}   # dropped by G-mid projection


def project_mid(preds):
    return frozenset(p for p in preds if p[0] not in FINE_DIMS)


# --- 5 failure classes: P* (true precondition) + irrelevant noise dims ---
# each noise dim carries several values so LGG has variation to intersect.
CLASSES = {
    "PermissionDenied": {
        "Pstar": {("tool", "write"), ("region", "eu"), ("perm", "ro")},
        "noise": {"path": [f"/f{i}" for i in range(12)],
                  "reqid": [f"r{i}" for i in range(12)],
                  "size": ["small", "med", "large"]},
    },
    "QuotaExceeded": {   # P* includes a FINE dim (size) -> A2 will over-warn
        "Pstar": {("tool", "write"), ("region", "us"), ("perm", "rw"),
                  ("size", "large")},
        "noise": {"path": [f"/g{i}" for i in range(12)],
                  "reqid": [f"q{i}" for i in range(12)]},
    },
    "RateLimited": {
        "Pstar": {("tool", "api"), ("endpoint", "render"), ("tier", "free")},
        "noise": {"key": [f"k{i}" for i in range(12)],
                  "reqid": [f"a{i}" for i in range(12)],
                  "size": ["small", "large"]},
    },
    "AuthError": {       # P* includes a FINE dim (key) -> A2 will over-warn
        "Pstar": {("tool", "api"), ("endpoint", "charge"), ("key", "revoked")},
        "noise": {"reqid": [f"c{i}" for i in range(12)],
                  "size": ["small", "med", "large"]},
    },
    "NotFound": {
        "Pstar": {("tool", "db"), ("table", "orders"), ("mode", "read")},
        "noise": {"key": [f"d{i}" for i in range(12)],
                  "reqid": [f"n{i}" for i in range(12)]},
    },
}


def would_fail(call: Signature, pstar: frozenset) -> bool:
    return pstar <= call.preds


def _make_failing(pstar, noise, rng) -> Signature:
    preds = set(pstar)
    for dim, vals in noise.items():
        if not any(d == dim for d, _ in pstar):   # don't clobber a P* dim
            preds.add((dim, rng.choice(vals)))
    return Signature(frozenset(preds))


def _make_near_miss(pstar, noise, rng) -> Signature:
    """Violate exactly ONE predicate of P* (hard negative)."""
    pstar = list(pstar)
    drop_dim, drop_val = rng.choice(pstar)
    preds = set()
    for d, v in pstar:
        if d == drop_dim:
            preds.add((d, v + "_SAFE"))   # flip that one predicate to a safe value
        else:
            preds.add((d, v))
    for dim, vals in noise.items():
        if not any(d == dim for d, _ in pstar):
            preds.add((dim, rng.choice(vals)))
    return Signature(frozenset(preds))


def _sym_diff(a: frozenset, b: frozenset) -> int:
    return len(a ^ b)


def run_class(name, spec, k, seed=0):
    rng = random.Random(seed)
    pstar = frozenset(spec["Pstar"])
    noise = spec["noise"]
    tool = dict(pstar)["tool"]

    # seed pool: k distinct failing calls (varied irrelevant dims)
    seeds, seen = [], set()
    while len(seeds) < k:
        s = _make_failing(pstar, noise, rng)
        if s.preds not in seen:
            seen.add(s.preds)
            seeds.append(s)

    # held-out: 50 failing (none identical to a seed) + 50 near-miss
    held_fail, held_miss = [], []
    while len(held_fail) < 50:
        s = _make_failing(pstar, noise, rng)
        if s.preds not in seen:
            held_fail.append(s)
    while len(held_miss) < 50:
        held_miss.append(_make_near_miss(pstar, noise, rng))

    # build the learned layer from the k seeds
    ll = LearningLayer()
    for s in seeds:
        ll.on_failure(s, tool, name)
    grec = ll.grecs.get((tool, name))
    gsig = grec.gsig.preds if grec else frozenset()

    # A2 pattern = G-mid projection of the seeds (all equal mid(P*) here)
    a2_patterns = {project_mid(s.preds) for s in seeds}
    seed_fine = {s.preds for s in seeds}

    def warn_A1(call):   # instance equality at G-fine
        return call.preds in seed_fine

    def warn_A2(call):   # fixed G-mid subsumption
        return any(p <= call.preds for p in a2_patterns)

    def warn_A3(call):   # learned layer subsumption
        return len(ll.check(call)) > 0

    arms = {"A1": warn_A1, "A2": warn_A2, "A3": warn_A3}
    out = {}
    for arm, fn in arms.items():
        tp = sum(1 for c in held_fail if fn(c))          # warned & would-fail
        fn_ = sum(1 for c in held_fail if not fn(c))
        fp = sum(1 for c in held_miss if fn(c))          # warned & would-succeed
        recall = tp / (tp + fn_) if tp + fn_ else 0.0
        precision = tp / (tp + fp) if tp + fp else 1.0
        out[arm] = {"recall": recall, "precision": precision}
    out["gsig_symdiff"] = _sym_diff(gsig, pstar)
    return out


def run_e1(ks=(2, 3, 4, 5, 6, 7, 8), seeds=(0, 1, 2)):
    import statistics as st
    results = {}   # k -> arm -> {recall,precision}; plus convergence
    for k in ks:
        agg = {a: {"recall": [], "precision": []} for a in ("A1", "A2", "A3")}
        sd = []
        pass4 = []  # per (class,seed) whether A3 recall>=0.8 & precision>=0.9
        for seed in seeds:
            for name, spec in CLASSES.items():
                r = run_class(name, spec, k, seed=seed)
                for a in agg:
                    agg[a]["recall"].append(r[a]["recall"])
                    agg[a]["precision"].append(r[a]["precision"])
                sd.append(r["gsig_symdiff"])
                pass4.append(r["A3"]["recall"] >= 0.8 and
                             r["A3"]["precision"] >= 0.9)
        results[k] = {
            a: {"recall": st.mean(agg[a]["recall"]),
                "precision": st.mean(agg[a]["precision"])} for a in agg}
        results[k]["gsig_symdiff"] = st.mean(sd)
        results[k]["A3_pass_frac"] = sum(pass4) / len(pass4)
    return results


if __name__ == "__main__":
    res = run_e1()
    print(f"{'k':>3} | {'A1 rec':>7}{'A2 rec':>7}{'A3 rec':>7} | "
          f"{'A1 prec':>8}{'A2 prec':>8}{'A3 prec':>8} | "
          f"{'|gsigDP*|':>10}{'A3 pass':>9}")
    for k, r in res.items():
        print(f"{k:>3} | "
              f"{r['A1']['recall']:>7.2f}{r['A2']['recall']:>7.2f}"
              f"{r['A3']['recall']:>7.2f} | "
              f"{r['A1']['precision']:>8.2f}{r['A2']['precision']:>8.2f}"
              f"{r['A3']['precision']:>8.2f} | "
              f"{r['gsig_symdiff']:>10.2f}{r['A3_pass_frac']:>9.0%}")
