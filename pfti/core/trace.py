"""JSONL trace logging -- the Day-0 schema (plan §1.2).

One event per line:
{episode, step, agent, event_type, tool, args, signature, outcome,
 error_class, warned, matched_record_id, oracle_would_fail,
 tokens_in, tokens_out}

Every metric in eval/metrics.py is computed from this file.
"""
from __future__ import annotations

import json


class TraceLogger:
    def __init__(self, path: str | None = None):
        self.path = path
        self.events: list[dict] = []

    def log(self, **event):
        sig = event.get("signature")
        if sig is not None and hasattr(sig, "to_json"):
            event["signature"] = sig.to_json()
        self.events.append(event)
        if self.path:
            with open(self.path, "a") as f:
                f.write(json.dumps(event, default=str) + "\n")
