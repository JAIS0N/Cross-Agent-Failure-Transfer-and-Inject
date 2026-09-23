"""Core before/after experiment: real LLM agents x {off, shadow, inject}.

Same multi-agent testbed as the scripted runner, but each agent is driven
by a real (or fake) LLM choosing tool calls. The SAME PFTI machinery
(Signature/Matcher/FailureStore/Injector/oracle/TraceLogger) is used; only
the *actor* changes.

Warning delivery is act-time and advisory: when the model proposes a call
that matches a peer failure, the call is held and the warning is returned
as its tool result so the model can reconsider once. If the model proposes
the same call again it proceeds (advisory, never blocking) -- and if that
proceed then fails, it counts as an *ignored* warning. off = no matching;
shadow = matcher runs + is scored against the oracle but never injects
(pure measurement); inject = warnings delivered.
"""
from __future__ import annotations

import json

from ..core.signature import extract
from ..core.store import FailureStore
from ..core.matcher import Matcher
from ..core.inject import render
from ..core.trace import TraceLogger
from ..envs.tools import World, ToolError, TOOLS, would_fail
from .providers import strip_script
from .scenarios_llm import SCENARIOS

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write content to a file at an absolute path.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "read_file", "description": "Read a file at an absolute path.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "list_dir", "description": "List files in a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
    {"type": "function", "function": {
        "name": "run_code", "description": "Run a code snippet.",
        "parameters": {"type": "object", "properties": {
            "code": {"type": "string"}, "path": {"type": "string"},
            "content": {"type": "string"}}, "required": ["code"]}}},
    {"type": "function", "function": {
        "name": "call_api", "description": "Call an API endpoint.",
        "parameters": {"type": "object", "properties": {
            "endpoint": {"type": "string"}, "payload": {"type": "string"}},
            "required": ["endpoint"]}}},
    {"type": "function", "function": {
        "name": "query_db", "description": "Query a database table.",
        "parameters": {"type": "object", "properties": {
            "table": {"type": "string"}, "query": {"type": "string"}},
            "required": ["table"]}}},
]

SYSTEM = ("You are {agent}, one agent in a software-ops team that shares a "
          "workspace. Complete your task using the provided tools. Make one "
          "tool call at a time. If a tool fails, adapt. If a warning about a "
          "teammate's failure appears, take it seriously: prefer a verified "
          "alternative. Reply with the word DONE when your task is complete.")


def _usage(resp):
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0
    return (getattr(u, "prompt_tokens", 0) or 0,
            getattr(u, "completion_tokens", 0) or 0)


class _Counters:
    def __init__(self):
        self.api_ok = 0
        self.db_ok = 0
        self.failures = 0
        self.repeated_peer = 0
        self.warnings = 0
        self.ignored = 0
        self.p_tok = 0
        self.c_tok = 0
        self.fail_sigs = []  # (agent, fine_sig)
        # --- added after review (non-circular accounting) ---
        self.held_tp = 0          # calls held by a warning that WOULD have failed
        self.held_fp = 0          # calls held by a warning that would have SUCCEEDED
        self.doomed_attempts = 0  # proposed calls the oracle says would fail (held or not)
        self.peer_doomed = 0      # ...of which a DIFFERENT agent already failed with
                                  #    the same tool + error class (oracle-defined,
                                  #    independent of the matcher's signatures)
        self.peer_doomed_executed = 0  # ...and the call was actually executed
        self.fail_classes = []    # (agent, tool, error_class) of executed failures
        self.transcripts = {}     # agent -> message list (only if keep_transcripts)


def _run_agent(client, model, agent, task, script, world, store, matcher,
               logger, mode, level, ctr, episode, step0, use_fake,
               max_turns=8, keep_transcripts=False):
    # the fake needs the embedded script; a real model must not see it
    if use_fake:
        user_task = strip_script(task) + "\n<<<PFTI_SCRIPT " + \
            json.dumps(script or {}) + ">>>"
    else:
        user_task = strip_script(task)
    msgs = [{"role": "system", "content": SYSTEM.format(agent=agent)},
            {"role": "user", "content": user_task}]
    warned_keys = set()
    step = step0
    for _ in range(max_turns):
        resp = client.chat.completions.create(
            model=model, messages=msgs, tools=TOOL_SCHEMAS, temperature=0.0)
        p, c = _usage(resp)
        ctr.p_tok += p
        ctr.c_tok += c
        m = resp.choices[0].message
        tool_calls = getattr(m, "tool_calls", None)
        if not tool_calls:
            if m.content and "DONE" in m.content.upper():
                break
            msgs.append({"role": "assistant", "content": m.content or ""})
            continue

        # keep the assistant message (with its tool_calls) in history
        msgs.append(_assistant_to_dict(m))
        tc = tool_calls[0]  # one call at a time
        name = tc.function.name
        try:
            args = json.loads(tc.function.arguments or "{}")
        except Exception:
            args = {}
        step += 1
        sig = extract(name, args, world) if name in TOOLS else None

        # ---- act-time matcher check ----
        hits = []
        if sig is not None and mode != "off":
            hits = matcher.check(sig, store, exclude_agent=agent,
                                 current_step=step)
        try:
            oracle = would_fail(name, args, world) if name in TOOLS else None
        except TypeError:  # malformed args: the tool call itself will error
            oracle = None
        key = sig.project(level).preds if sig is not None else None
        will_warn = bool(hits) and mode == "inject" and key not in warned_keys
        peer_doomed = False
        if oracle is not None:
            ctr.doomed_attempts += 1
            peer_doomed = any(ag != agent and t == name and ec == oracle[0]
                              for ag, t, ec in ctr.fail_classes)
            ctr.peer_doomed += int(peer_doomed)
        logger.log(episode=episode, step=step, agent=agent,
                   event_type="pre_call", tool=name, args=args,
                   signature=sig, would_warn=bool(hits),
                   warned=will_warn,
                   matched_record_id=[r.id for r in hits],
                   oracle_would_fail=(oracle[0] if oracle else None))

        if will_warn:
            # deliver advisory warning as the tool result; hold execution
            warned_keys.add(key)
            ctr.warnings += 1
            if oracle is not None:
                ctr.held_tp += 1
            else:
                ctr.held_fp += 1
            warn = "\n".join(render(r, level) for r in hits[:2])
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": warn + "\n[This call was NOT executed. "
                         "Reconsider: choose a verified alternative, or "
                         "repeat the call to proceed anyway.]"})
            continue

        # ---- execute ----
        if name not in TOOLS:
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": f"ERROR: unknown tool {name}"})
            continue
        proceeded_despite_warning = key in warned_keys
        if peer_doomed:
            ctr.peer_doomed_executed += 1
        try:
            out = TOOLS[name](world, **args)
            _log_post(logger, episode, step, agent, name, args, sig, None)
            store.note_success(agent)
            if name == "call_api":
                ctr.api_ok += 1
            if name == "query_db":
                ctr.db_ok += 1
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": f"OK: {out}"})
        except ToolError as e:
            # repeated-peer-failure: a DIFFERENT agent already failed same mid
            mid = sig.project("mid")
            if any(ag != agent and s.project("mid") == mid
                   for ag, s in ctr.fail_sigs):
                ctr.repeated_peer += 1
            ctr.fail_sigs.append((agent, sig))
            ctr.fail_classes.append((agent, name, e.error_class))
            ctr.failures += 1
            if proceeded_despite_warning:
                ctr.ignored += 1
            store.add(sig, name, agent, e.error_class, e.msg, episode, step)
            _log_post(logger, episode, step, agent, name, args, sig, e)
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": f"ERROR {e.error_class}: {e.msg}"})
        except TypeError as e:  # model passed arguments the tool doesn't take
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": f"ERROR InvalidArgument: {e}"})
    if keep_transcripts:
        ctr.transcripts[agent] = msgs
    return step


def _assistant_to_dict(m):
    tcs = []
    for tc in (getattr(m, "tool_calls", None) or []):
        tcs.append({"id": tc.id, "type": "function",
                    "function": {"name": tc.function.name,
                                 "arguments": tc.function.arguments}})
    return {"role": "assistant", "content": m.content or "", "tool_calls": tcs}


def _log_post(logger, episode, step, agent, tool, args, sig, err):
    logger.log(episode=episode, step=step, agent=agent,
               event_type="post_call", tool=tool, args=args, signature=sig,
               outcome="fail" if err else "ok",
               error_class=err.error_class if err else None)


def _success(scenario, world, ctr):
    sc = scenario["success"]
    ok = True
    if "all_files" in sc:
        ok &= all(p in world.files for p in sc["all_files"])
    if "any_files" in sc:
        ok &= any(p in world.files for p in sc["any_files"])
    if "api_calls_ok" in sc:
        ok &= ctr.api_ok >= sc["api_calls_ok"]
    if "db_queries_ok" in sc:
        ok &= ctr.db_ok >= sc["db_queries_ok"]
    return bool(ok)


def _apply_world_event(world, ev):
    """Scenario-scripted change to the world between agents (e.g. an admin
    fixes a permission), used to test stale peer warnings."""
    for k, v in ev.get("set", {}).items():
        if k == "perm":
            world.perm.update(v)
        elif k == "api_budget":
            world.api_budget.update(v)
        elif k == "dirs":
            world.dirs.update(v)
        elif k == "tables":
            world.tables.update(v)
        elif k == "disk_quota":
            world.disk_quota = v


def run_scenario(scenario, mode, client, model, level="mid",
                 match_mode="subsumption", use_fake=False, logger=None,
                 keep_transcripts=False):
    world = World(**scenario["world"])
    store = FailureStore(scope="episode")
    matcher = Matcher(level=level, mode=match_mode)
    logger = logger or TraceLogger()
    ctr = _Counters()
    ep = f"{scenario['name']}[{model}/{mode}]"
    step = 0
    events = {ev["after_agent"]: ev for ev in scenario.get("world_events", [])}
    for idx, a in enumerate(scenario["agents"]):
        step = _run_agent(client, model, a["id"], a["task"], a.get("script"),
                          world, store, matcher, logger, mode, level, ctr, ep,
                          step, use_fake, keep_transcripts=keep_transcripts)
        if idx in events:
            _apply_world_event(world, events[idx])
    return {
        "scenario": scenario["name"], "model": model, "mode": mode,
        "success": _success(scenario, world, ctr),
        "failures": ctr.failures,
        "repeated_peer_failures": ctr.repeated_peer,
        "warnings_delivered": ctr.warnings,
        "ignored_warnings": ctr.ignored,
        "prompt_tokens": ctr.p_tok, "completion_tokens": ctr.c_tok,
        "total_tokens": ctr.p_tok + ctr.c_tok,
        "steps": step,
        "held_tp": ctr.held_tp, "held_fp": ctr.held_fp,
        "doomed_attempts": ctr.doomed_attempts,
        "peer_doomed_attempts": ctr.peer_doomed,
        "peer_doomed_executed": ctr.peer_doomed_executed,
        "events": list(logger.events),
        "transcripts": ctr.transcripts,
    }


def shadow_prf(events):
    """Matcher precision/recall vs oracle over pre_call events."""
    tp = fp = fn = 0
    for e in events:
        if e.get("event_type") != "pre_call":
            continue
        would = e.get("would_warn")
        danger = e.get("oracle_would_fail") is not None
        if would and danger:
            tp += 1
        elif would and not danger:
            fp += 1
        elif (not would) and danger:
            fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    return {"precision": prec, "recall": rec, "tp": tp, "fp": fp, "fn": fn}


def run_matrix(models, modes=("off", "shadow", "inject"), level="mid",
               use_fake=False, client=None, scenarios=None, verbose=True):
    if client is None:
        if use_fake:
            from .providers import FakeLLM
            client = FakeLLM()
        else:
            from .providers import build_openai_client
            client = build_openai_client()
    scenarios = scenarios or SCENARIOS
    cells = []
    for model in models:
        for mode in modes:
            agg = {"model": model, "mode": mode, "n_scenarios": 0,
                   "n_success": 0, "failures": 0, "repeated_peer_failures": 0,
                   "warnings_delivered": 0, "ignored_warnings": 0,
                   "total_tokens": 0}
            all_events = []
            per_scenario = []
            for sc in scenarios:
                r = run_scenario(sc, mode, client, model, level=level,
                                 use_fake=use_fake, logger=TraceLogger())
                agg["n_scenarios"] += 1
                agg["n_success"] += int(r["success"])
                for k in ("failures", "repeated_peer_failures",
                          "warnings_delivered", "ignored_warnings",
                          "total_tokens"):
                    agg[k] += r[k]
                all_events += r["events"]
                per_scenario.append({k: r[k] for k in
                                     ("scenario", "success", "failures",
                                      "repeated_peer_failures",
                                      "warnings_delivered", "ignored_warnings",
                                      "total_tokens", "steps")})
            agg["success_rate"] = agg["n_success"] / max(1, agg["n_scenarios"])
            agg["shadow"] = shadow_prf(all_events)
            agg["per_scenario"] = per_scenario
            cells.append(agg)
            if verbose:
                sh = agg["shadow"]
                print(f"  {model:<12} {mode:<7} "
                      f"success={agg['success_rate']:>5.0%} "
                      f"fails={agg['failures']:>2} "
                      f"repeat_peer={agg['repeated_peer_failures']:>2} "
                      f"warn={agg['warnings_delivered']:>2} "
                      f"ignored={agg['ignored_warnings']:>2} "
                      f"tok={agg['total_tokens']:>6} "
                      f"P/R={sh['precision']:.2f}/{sh['recall']:.2f}")
    return cells
