"""Matching relations (plan §2.4, Def 2).

Equality at level L : project(record, L) == project(call, L)
Subsumption at level L: project(record, L).preds SUBSETEQ call_fine.preds
  (the stored, possibly generalized, failure pattern is implied by the
   current call). Equality-at-coarser-L is a special case of subsumption.

exclude_agent implements the key invariant: an agent is never warned
about its OWN records (self-warnings are the loop-guard baseline B2,
not the contribution).
"""
from __future__ import annotations

from .signature import Signature
from .store import FailureStore, Record


class Matcher:
    def __init__(self, level: str = "mid", mode: str = "subsumption"):
        assert mode in ("equality", "subsumption")
        self.level = level
        self.mode = mode

    def check(self, sig_fine: Signature, store: FailureStore,
              exclude_agent: str | None = None,
              current_step: int | None = None) -> list[Record]:
        hits = []
        call_proj = sig_fine.project(self.level)
        for r in store.alive(current_step):
            if exclude_agent is not None and r.agent == exclude_agent:
                continue
            rec_proj = r.sig.project(self.level)
            if self.mode == "equality":
                ok = rec_proj == call_proj
            else:
                ok = rec_proj.subsumes(sig_fine)
            if ok:
                r.hit_count += 1
                hits.append(r)
        return hits
