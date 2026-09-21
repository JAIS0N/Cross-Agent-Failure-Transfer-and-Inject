"""Predicate signature extraction (PFTI core, plan §2.2).

A signature is a conjunction of atomic predicates over a tool call and
observable world state, stored as frozenset[(dimension, value)].

Three granularity levels form a chain (coarse <= mid <= fine): coarser
levels are PROJECTIONS of the fine signature -- extract once at G-fine,
derive the rest with .project(). This containment is the partial order
the theory section (Lemma 1) needs.

Rules: values are normalized lowercase strings; numerics are bucketed;
never include free text or timestamps.
"""
from __future__ import annotations

from dataclasses import dataclass

LEVELS = ("coarse", "mid", "fine")
_ORDER = {"coarse": 0, "mid": 1, "fine": 2}

# dimension -> the coarsest level at which it appears
DIM_LEVEL = {
    # G-coarse: tool identity only
    "tool": "coarse",
    # G-mid: arg types + key observable state flags
    "arg_types": "mid",
    "path_prefix": "mid",
    "perm": "mid",
    "endpoint": "mid",
    "table": "mid",
    # G-fine: normalized concrete args
    "path": "fine",
    "query": "fine",
    "payload_kind": "fine",
    "size_bucket": "fine",
    "code_kind": "fine",
}


def _norm(v) -> str:
    return str(v).strip().lower()


def size_bucket(n: int) -> str:
    return "small" if n < 100 else ("med" if n < 10_000 else "large")


def path_prefix(path: str) -> str:
    parts = _norm(path).split("/")
    return "/" + parts[1] if len(parts) > 1 and parts[1] else "/"


@dataclass(frozen=True)
class Signature:
    preds: frozenset  # frozenset[tuple[str, str]]

    def project(self, level: str) -> "Signature":
        """Keep only predicates whose dimension exists at `level` or coarser."""
        if level not in _ORDER:
            raise ValueError(f"unknown level {level!r}")
        keep = {d for d, lv in DIM_LEVEL.items() if _ORDER[lv] <= _ORDER[level]}
        return Signature(frozenset(p for p in self.preds if p[0] in keep))

    def subsumes(self, other: "Signature") -> bool:
        """self generalizes other: every predicate of self holds in other.
        Frozenset-subset check: O(|preds|), decidable, interpretable."""
        return self.preds <= other.preds

    def readable(self) -> str:
        return "(" + ", ".join(f"{d}={v}" for d, v in sorted(self.preds)) + ")"

    def to_json(self):
        return sorted(list(p) for p in self.preds)

    @staticmethod
    def from_json(items) -> "Signature":
        return Signature(frozenset((d, v) for d, v in items))


def extract(tool: str, args: dict, world=None) -> Signature:
    """Extract the G-fine signature for a call. Coarser levels via .project()."""
    preds = {("tool", _norm(tool))}
    preds.add(("arg_types", ",".join(
        sorted(f"{k}:{type(v).__name__}" for k, v in args.items()))))

    p = args.get("path")
    if p is not None:
        pre = path_prefix(p)
        preds.add(("path_prefix", pre))
        preds.add(("path", _norm(p)))
        if world is not None:
            perm = world.perm_of(pre)
            if perm:
                preds.add(("perm", perm))
    if "content" in args:
        preds.add(("size_bucket", size_bucket(len(str(args["content"])))))
    if "endpoint" in args:
        preds.add(("endpoint", _norm(args["endpoint"])))
    if "payload" in args:
        preds.add(("payload_kind", type(args["payload"]).__name__.lower()))
    if "table" in args:
        preds.add(("table", _norm(args["table"])))
    if "query" in args:
        preds.add(("query", _norm(args["query"])[:80]))
    if "code" in args:
        preds.add(("code_kind", "script"))
    return Signature(frozenset(preds))
