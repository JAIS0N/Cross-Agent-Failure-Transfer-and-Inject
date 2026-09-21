"""Append-only failure store (plan §2.3).

Records are ALWAYS stored at G-fine; match-time projection gives every
granularity from one store. In-memory list + optional JSONL persistence.

Failure definition v1: a tool call is a failure event iff
(a) the tool raised a ToolError, or
(b) the same agent issued an identical G-fine call twice in a row and
    both raised (retry exhaustion -- recorded once, flagged).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

from .signature import Signature


@dataclass
class Record:
    id: int
    sig: Signature          # always G-fine
    tool: str
    agent: str
    error_class: str
    msg: str
    episode: str = ""
    step: int = 0
    ts: float = field(default_factory=time.time)
    retry_exhausted: bool = False
    hit_count: int = 0


class FailureStore:
    """scope in {episode, session, global}; ttl = lifetime in steps (None = inf)."""

    def __init__(self, scope: str = "episode", ttl: int | None = None,
                 jsonl_path: str | None = None):
        self.scope = scope
        self.ttl = ttl
        self.jsonl_path = jsonl_path
        self.records: list[Record] = []
        self._next_id = 0
        self._last_fail: dict[str, Signature] = {}  # agent -> last failed fine sig

    def add(self, sig: Signature, tool: str, agent: str, error_class: str,
            msg: str = "", episode: str = "", step: int = 0) -> Record:
        retry = self._last_fail.get(agent) == sig
        self._last_fail[agent] = sig
        if retry:
            # retry exhaustion: don't duplicate; flag the existing record
            for r in reversed(self.records):
                if r.agent == agent and r.sig == sig:
                    r.retry_exhausted = True
                    return r
        rec = Record(self._next_id, sig, tool, agent, error_class, msg,
                     episode, step)
        self._next_id += 1
        self.records.append(rec)
        if self.jsonl_path:
            with open(self.jsonl_path, "a") as f:
                f.write(json.dumps({
                    "id": rec.id, "sig": sig.to_json(), "tool": tool,
                    "agent": agent, "error_class": error_class, "msg": msg,
                    "episode": episode, "step": step,
                    "retry_exhausted": rec.retry_exhausted}) + "\n")
        return rec

    def alive(self, current_step: int | None = None) -> list[Record]:
        if self.ttl is None or current_step is None:
            return list(self.records)
        return [r for r in self.records if current_step - r.step <= self.ttl]

    def note_success(self, agent: str):
        self._last_fail.pop(agent, None)

    def reset_episode(self):
        if self.scope == "episode":
            self.records.clear()
            self._last_fail.clear()
