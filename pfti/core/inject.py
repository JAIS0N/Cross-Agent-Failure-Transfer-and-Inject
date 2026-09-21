"""Warning injection (plan §2.5).

Fixed template, advisory (never blocking), at most k=2 warnings per step,
includes the human-readable signature (interpretability).
"""
from __future__ import annotations

from collections import defaultdict

from .store import Record

TEMPLATE = (
    "!! PEER FAILURE WARNING\n"
    "Agent {peer} attempted {tool} with a matching pattern {sig}\n"
    "at step {step} and failed with {error_class}: \"{msg}\".\n"
    "Consider: (a) verifying the precondition first, (b) an alternative\n"
    "tool/approach, or (c) proceeding only if your situation differs."
)

MAX_WARNINGS_PER_STEP = 2


def render(rec: Record, level: str = "mid") -> str:
    return TEMPLATE.format(peer=rec.agent, tool=rec.tool,
                           sig=rec.sig.project(level).readable(),
                           step=rec.step, error_class=rec.error_class,
                           msg=rec.msg)


class Injector:
    """Holds pending warnings per agent; drained into the agent's next
    prompt as a system-role message (LLM runner) or surfaced to scripted
    agents directly."""

    def __init__(self, k: int = MAX_WARNINGS_PER_STEP, level: str = "mid"):
        self.k = k
        self.level = level
        self._pending: dict[str, list[str]] = defaultdict(list)

    def arm(self, agent: str, hits: list[Record]) -> list[str]:
        msgs = [render(r, self.level) for r in hits[: self.k]]
        self._pending[agent].extend(msgs)
        return msgs

    def drain(self, agent: str) -> list[str]:
        msgs = self._pending.pop(agent, [])
        return msgs
