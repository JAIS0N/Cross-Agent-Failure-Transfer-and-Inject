"""Scenario runner with scripted agents (deterministic, no LLM).

Each agent follows a plan of tool calls. If PFTI warns before a call and
the step declares an `on_warn` alternative, the agent takes it (this is
the scripted stand-in for 'the LLM heeds the warning' -- the LLM runner
in demo_llm.py replaces exactly this decision with a real model).
"""
from __future__ import annotations

import yaml

from ..core import FailureStore, Matcher, Injector, PFTIHook, TraceLogger
from ..envs.tools import World, ToolError, TOOLS, would_fail


def load_scenario(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_world(cfg: dict) -> World:
    return World(**cfg.get("world", {}))


def run_episode(scenario: dict, mode: str = "inject", level: str = "mid",
                match_mode: str = "subsumption",
                logger: TraceLogger | None = None) -> dict:
    world = build_world(scenario)
    store = FailureStore(scope="episode")
    hook = PFTIHook(store, Matcher(level=level, mode=match_mode),
                    Injector(level=level), mode=mode,
                    logger=logger or TraceLogger(), oracle=would_fail)
    episode = scenario["name"] + f"[{mode}/{level}/{match_mode}]"

    # interleave agents round-robin, one plan step per turn
    agents = [dict(a) for a in scenario["agents"]]
    cursors = {a["id"]: 0 for a in agents}
    step = 0
    transcript = []
    api_ok = db_ok = failures = 0
    fail_sigs_by_agent: list[tuple[str, object]] = []
    repeated_peer = 0

    active = True
    while active:
        active = False
        for a in agents:
            i = cursors[a["id"]]
            if i >= len(a["plan"]):
                continue
            active = True
            cursors[a["id"]] += 1
            step += 1
            item = a["plan"][i]
            tool, args = item["tool"], dict(item["args"])

            warnings = hook.before_call(a["id"], tool, args, world,
                                        episode, step)
            if warnings and "on_warn" in item:
                transcript.append((step, a["id"], "WARNED", warnings[0]))
                alt = item["on_warn"]
                tool, args = alt["tool"], dict(alt["args"])
                hook.before_call(a["id"], tool, args, world, episode, step)

            from ..core.signature import extract
            sig = extract(tool, args, world)
            try:
                out = TOOLS[tool](world, **args)
                hook.after_call(a["id"], tool, args, world, None,
                                episode, step)
                transcript.append((step, a["id"], f"{tool} OK", str(out)[:60]))
                if tool == "call_api":
                    api_ok += 1
                if tool == "query_db":
                    db_ok += 1
            except ToolError as e:
                # repeated-peer-failure: fine sig already failed for a peer
                if any(s.project("mid") == sig.project("mid") and ag != a["id"]
                       for ag, s in fail_sigs_by_agent):
                    repeated_peer += 1
                fail_sigs_by_agent.append((a["id"], sig))
                failures += 1
                hook.after_call(a["id"], tool, args, world, e, episode, step)
                transcript.append((step, a["id"], f"{tool} FAIL",
                                   f"{e.error_class}: {e.msg}"))

    # success check
    sc = scenario.get("success", {})
    success = True
    if "any_files" in sc:
        success &= any(p in world.files for p in sc["any_files"])
    if "all_files" in sc:
        success &= all(p in world.files for p in sc["all_files"])
    if "api_calls_ok" in sc:
        success &= api_ok >= sc["api_calls_ok"]
    if "db_queries_ok" in sc:
        success &= db_ok >= sc["db_queries_ok"]

    return {"name": scenario["name"], "mode": mode, "level": level,
            "match_mode": match_mode, "success": success,
            "failures": failures, "repeated_peer_failures": repeated_peer,
            "steps": step, "transcript": transcript,
            "trace": hook.logger.events}
