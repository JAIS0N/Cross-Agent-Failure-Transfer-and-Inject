"""The Learning Layer: adaptive granularity via online LGG (spec §2).

Today granularity is a fixed config. This layer LEARNS the right
abstraction per failure class, online, from the failure instances
themselves, by intersecting predicate sets on the semilattice (least
general generalization, LGG), and self-corrects when it over-generalizes.

Everything here is symbolic and auditable -- NO LLM. It reuses the
existing Signature machinery. The instance log (always G-fine) is
unchanged; generalized records are DERIVED, never written by the hook.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .signature import Signature

# --- guard constants (spec §2.3, §2.4) ---
MIN_SIZE = 2      # min NON-tool predicates in a generalized record (anti-collapse)
K_MIN = 5         # min fires before a generalized record may be specialized
FP_MAX = 0.3      # max oracle-contradiction rate before split


def _nontool(preds) -> frozenset:
    return frozenset(p for p in preds if p[0] != "tool")


@dataclass
class GeneralizedRecord:
    gsig: Signature                 # the intersection (LGG) of the bucket's instances
    tool: str
    error_class: str
    provenance: list = field(default_factory=list)   # instance ids merged in
    fired: int = 0
    fired_correct: int = 0
    fired_wrong: int = 0
    active: bool = True
    hit_log: list = field(default_factory=list)       # (call_preds, oracle_fail)

    @property
    def fp_rate(self):
        return self.fired_wrong / self.fired if self.fired else 0.0


@dataclass
class _Instance:
    id: int
    sig: Signature


class LearningLayer:
    """Buckets keyed by (tool, error_class). Instances land in their bucket
    on write; generalized records are derived via LGG under two guards."""

    def __init__(self, min_size=MIN_SIZE, k_min=K_MIN, fp_max=FP_MAX,
                 logger=None):
        self.min_size = min_size
        self.k_min = k_min
        self.fp_max = fp_max
        self.logger = logger
        self.buckets: dict = {}          # key -> list[_Instance]
        self.grecs: dict = {}            # key -> GeneralizedRecord (active or not)
        self.subrecs: dict = {}          # key -> list[GeneralizedRecord] (post-split)
        self.refusals: list = []         # (key, reason)
        self.splits: list = []           # (key, dim)
        self._next_id = 0

    # ---------------- Guard 1 + LGG (spec §2.2, §2.3) ----------------
    def on_failure(self, sig: Signature, tool: str, error_class: str):
        key = (tool, error_class)
        inst = _Instance(self._next_id, sig)
        self._next_id += 1
        self.buckets.setdefault(key, []).append(inst)
        self._generalize_bucket(key)

    def _generalize_bucket(self, key):
        instances = self.buckets[key]
        if len(instances) < 2:
            return  # nothing to generalize yet
        g = frozenset.intersection(*[i.sig.preds for i in instances])
        if len(_nontool(g)) < self.min_size:      # Guard 1: anti-collapse
            self.refusals.append((key, "intersection_too_small"))
            self._log("gen_refused", key=key, reason="intersection_too_small",
                      size=len(_nontool(g)))
            self.grecs.pop(key, None)             # no valid generalized record
            return
        tool, error_class = key
        self.grecs[key] = GeneralizedRecord(
            gsig=Signature(g), tool=tool, error_class=error_class,
            provenance=[i.id for i in instances])

    # ---------------- matching (spec §2.1) ----------------
    def check(self, sig: Signature) -> list:
        """Generalized records whose gsig subsumes the current call.
        (subsumption: gsig.preds subset current_sig.preds)."""
        hits = []
        for key, g in self.grecs.items():
            if g.active and g.gsig.preds <= sig.preds:
                hits.append(g)
        for key, subs in self.subrecs.items():
            for g in subs:
                if g.active and g.gsig.preds <= sig.preds:
                    hits.append(g)
        return hits

    # ---------------- Guard 2: self-correction (spec §2.4) ----------------
    def record_fire(self, grec: GeneralizedRecord, call_sig: Signature,
                    oracle_fail: bool):
        grec.fired += 1
        if oracle_fail:
            grec.fired_correct += 1
        else:
            grec.fired_wrong += 1
        grec.hit_log.append((call_sig.preds, oracle_fail))
        self.maybe_specialize(grec)

    def maybe_specialize(self, grec: GeneralizedRecord):
        if grec.fired >= self.k_min and grec.fp_rate > self.fp_max:
            grec.active = False
            key = (grec.tool, grec.error_class)
            dim = self._best_separating_dimension(grec)
            if dim is None:
                # nothing separates -> fall back to instance matching
                self._log("gen_deactivated", key=key, reason="no_separator")
                return
            self.splits.append((key, dim))
            self._log("gen_split", key=key, dim=dim)
            subs = []
            for sub_instances in self._partition(self.buckets[key], dim):
                if len(sub_instances) < 2:
                    continue
                g = frozenset.intersection(*[i.sig.preds for i in sub_instances])
                if len(_nontool(g)) < self.min_size:
                    continue
                subs.append(GeneralizedRecord(
                    gsig=Signature(g), tool=grec.tool,
                    error_class=grec.error_class,
                    provenance=[i.id for i in sub_instances]))
            self.subrecs[key] = subs

    def _best_separating_dimension(self, grec: GeneralizedRecord):
        """Dimension present in wrongly-warned sigs but absent from gsig,
        scored by information gain over correct/wrong labels; argmax."""
        gdims = {d for d, _ in grec.gsig.preds}
        # candidate dims: appear in any hit, not already in gsig
        cand = set()
        for preds, _ in grec.hit_log:
            for d, _v in preds:
                if d not in gdims:
                    cand.add(d)
        if not cand:
            return None
        base_H = _entropy([ok for _p, ok in grec.hit_log])
        best_dim, best_gain = None, 0.0
        for d in cand:
            gain = base_H - _cond_entropy(grec.hit_log, d)
            if gain > best_gain:
                best_dim, best_gain = d, gain
        return best_dim if best_gain > 1e-9 else None

    def _partition(self, instances, dim):
        groups = {}
        for inst in instances:
            val = dict(inst.sig.preds).get(dim, "__none__")
            groups.setdefault(val, []).append(inst)
        return list(groups.values())

    def _log(self, event, **kw):
        if self.logger:
            self.logger.log(event_type=event, **kw)


def _entropy(labels):
    n = len(labels)
    if n == 0:
        return 0.0
    p = sum(1 for x in labels if x) / n
    return -sum(q * math.log2(q) for q in (p, 1 - p) if q > 0)


def _cond_entropy(hit_log, dim):
    groups = {}
    for preds, ok in hit_log:
        val = dict(preds).get(dim, "__none__")
        groups.setdefault(val, []).append(ok)
    n = len(hit_log)
    return sum(len(g) / n * _entropy(g) for g in groups.values())
