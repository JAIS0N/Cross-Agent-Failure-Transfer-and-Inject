"""The interception hook (plan §2.4) -- the single interception point.

Framework-agnostic by construction: `PFTIHook` exposes before_call /
after_call, and `pfti_wrap` turns any plain tool function into a wrapped
one. The LLM runner (demo_llm.py) and the scripted runner (eval/runner.py)
both drive the same hook, so results are comparable.

Modes: "inject" (warnings delivered), "shadow" (matcher runs + logs, no
injection -- the measurement superpower), "off" (vanilla).
"""
from __future__ import annotations

from .inject import Injector
from .matcher import Matcher
from .signature import extract
from .store import FailureStore
from .trace import TraceLogger


class PFTIHook:
    def __init__(self, store: FailureStore, matcher: Matcher,
                 injector: Injector | None = None, mode: str = "inject",
                 logger: TraceLogger | None = None, oracle=None):
        assert mode in ("inject", "shadow", "off")
        self.store = store
        self.matcher = matcher
        self.injector = injector or Injector(level=matcher.level)
        self.mode = mode
        self.logger = logger or TraceLogger()
        self.oracle = oracle  # would_fail(tool, args, world) -> (cls,msg)|None

    # ---- PRE-CALL: check peer failures ----
    def before_call(self, agent: str, tool: str, args: dict, world,
                    episode: str = "", step: int = 0) -> list[str]:
        sig = extract(tool, args, world)
        hits = [] if self.mode == "off" else self.matcher.check(
            sig, self.store, exclude_agent=agent, current_step=step)
        oracle_fail = self.oracle(tool, args, world) if self.oracle else None
        self.logger.log(
            episode=episode, step=step, agent=agent, event_type="pre_call",
            tool=tool, args=args, signature=sig,
            would_warn=bool(hits),
            warned=bool(hits) and self.mode == "inject",
            matched_record_id=[r.id for r in hits],
            oracle_would_fail=(oracle_fail[0] if oracle_fail else None))
        if hits and self.mode == "inject":
            return self.injector.arm(agent, hits)
        return []

    # ---- POST-CALL: harvest failures ----
    def after_call(self, agent: str, tool: str, args: dict, world,
                   error=None, episode: str = "", step: int = 0):
        sig = extract(tool, args, world)
        if error is not None:
            self.store.add(sig, tool, agent, error.error_class,
                           getattr(error, "msg", str(error)), episode, step)
        else:
            self.store.note_success(agent)
        self.logger.log(
            episode=episode, step=step, agent=agent, event_type="post_call",
            tool=tool, args=args, signature=sig,
            outcome="fail" if error else "ok",
            error_class=error.error_class if error else None)


def pfti_wrap(tool_fn, tool_name: str, agent_id: str, hook: PFTIHook,
              world, get_step=lambda: 0, episode: str = ""):
    """Wrap a plain tool function at registration time (plan §2.4)."""
    def wrapped(**kwargs):
        step = get_step()
        hook.before_call(agent_id, tool_name, kwargs, world, episode, step)
        try:
            out = tool_fn(world, **kwargs)
            hook.after_call(agent_id, tool_name, kwargs, world, None,
                            episode, step)
            return out
        except Exception as e:
            hook.after_call(agent_id, tool_name, kwargs, world, e,
                            episode, step)
            raise
    return wrapped
