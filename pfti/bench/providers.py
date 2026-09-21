"""LLM providers for the benchmark.

Two clients, both exposing the OpenAI chat-completions interface used by
the runner (``client.chat.completions.create(model=, messages=, tools=,
temperature=)`` returning an object with ``.choices[0].message`` and
``.usage``):

* ``build_openai_client()`` -> a real OpenAI-compatible client (Ollama by
  default at http://localhost:11434/v1). This is what runs the real models.
* ``FakeLLM`` -> a deterministic, offline stand-in that reacts to injected
  peer-failure warnings exactly the way we want a cooperative model to,
  so the whole harness (metrics, aggregation, plots) can be validated
  with no model pulled. It is NOT a research result -- it exists to prove
  the plumbing, and it clearly separates `off` from `inject`.
"""
from __future__ import annotations

import json
import os
import re
import time

BASE_URL = os.environ.get("PFTI_BASE_URL", "http://localhost:11434/v1")

# The model matrix (Ollama tags). Cost is a size proxy for the routing side.
MODEL_MATRIX = {
    "qwen2.5:3b":  {"family": "qwen",  "size_b": 3,  "cost": 1.0},
    "qwen2.5:7b":  {"family": "qwen",  "size_b": 7,  "cost": 2.3},
    "qwen2.5:14b": {"family": "qwen",  "size_b": 14, "cost": 4.7},
    "llama3.1:8b": {"family": "llama", "size_b": 8,  "cost": 2.7},
}


def build_openai_client():
    """Real OpenAI-compatible client (Ollama / vLLM / LM Studio)."""
    from openai import OpenAI
    return OpenAI(base_url=BASE_URL, api_key=os.environ.get("PFTI_API_KEY",
                                                            "ollama"))


# --------------------------------------------------------------------------
# Deterministic offline fake (for CI / dry runs with no Ollama)
# --------------------------------------------------------------------------

class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments  # JSON string, like the real API


class _ToolCall:
    def __init__(self, cid, name, arguments):
        self.id = cid
        self.type = "function"
        self.function = _Fn(name, arguments)


class _Msg:
    def __init__(self, content=None, tool_calls=None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _Usage:
    def __init__(self, p, c):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = p + c


class _Resp:
    def __init__(self, message, usage):
        self.choices = [_Choice(message)]
        self.usage = usage


class _Completions:
    def __init__(self, parent):
        self._p = parent

    def create(self, model, messages, tools=None, temperature=0.0, **kw):
        return self._p._create(model, messages, tools, temperature)


class _Chat:
    def __init__(self, parent):
        self.completions = _Completions(parent)


class FakeLLM:
    """Offline stand-in.

    Policy, read purely from the message history so it is stateless:
      * If a '!! PEER FAILURE WARNING' system message is present, the agent
        heeds it and calls the ALTERNATIVE action for the matching pattern.
      * Else if its previous tool result was an ERROR, it retries the SAME
        doomed action once (this is the vanilla failure that repeats across
        agents), then reports DONE.
      * Else it makes its natural first attempt (the primary action).

    The primary/alternative actions come from the scenario's per-agent
    ``script`` (parsed out of the task text), so the fake needs no model.
    """

    def __init__(self, heed_prob=1.0):
        self.chat = _Chat(self)
        self.heed_prob = heed_prob
        self._tok = 0

    # crude token estimate so usage numbers are non-trivial in dry runs
    def _count(self, messages):
        return sum(len(str(m.get("content", ""))) for m in messages) // 4 + 8

    def _create(self, model, messages, tools, temperature):
        p_tok = self._count(messages)
        # find the embedded script (primary/alt) for this agent's task
        task = ""
        for m in messages:
            if m.get("role") == "user":
                task = m["content"]
        script = _parse_script(task)
        warned = any("PEER FAILURE WARNING" in str(m.get("content", ""))
                     for m in messages)
        last_tool_err = _last_tool_was_error(messages)
        already_alt = _already_did(messages, script.get("alt"))
        already_primary = _already_did(messages, script.get("primary"))

        if script.get("alt") and warned and not already_alt:
            call = script["alt"]
        elif last_tool_err and not already_alt:
            # vanilla path: retry the same doomed thing once more, then stop
            if already_primary and _primary_attempts(messages, script) >= 2:
                return self._done(p_tok)
            call = script.get("primary")
        elif not already_primary and script.get("primary"):
            call = script["primary"]
        elif script.get("alt") and not already_alt and last_tool_err:
            call = script["alt"]
        else:
            return self._done(p_tok)

        if not call:
            return self._done(p_tok)
        name, args = call
        tc = _ToolCall(f"call_{self._tok}", name, json.dumps(args))
        self._tok += 1
        msg = _Msg(content=None, tool_calls=[tc])
        c_tok = 12
        return _Resp(msg, _Usage(p_tok, c_tok))

    def _done(self, p_tok):
        return _Resp(_Msg(content="DONE"), _Usage(p_tok, 3))


# A compact, machine-parseable script travels inside the task string as a
# trailing '<<<PFTI_SCRIPT ...>>>' block so the fake can act it out. Real
# models never see it (the runner strips it before sending to a real client
# but leaves it for the fake). Format: primary and optional alt tool calls.

_SCRIPT_RE = re.compile(r"<<<PFTI_SCRIPT (.*?)>>>", re.S)


def _parse_script(task: str) -> dict:
    m = _SCRIPT_RE.search(task or "")
    if not m:
        return {}
    try:
        return json.loads(m.group(1))
    except Exception:
        return {}


def strip_script(task: str) -> str:
    return _SCRIPT_RE.sub("", task or "").strip()


def _call_eq(call, name_args):
    if not call or not name_args:
        return False
    return call[0] == name_args[0] and call[1] == name_args[1]


def _iter_assistant_calls(messages):
    for m in messages:
        tcs = m.get("tool_calls") if isinstance(m, dict) else getattr(
            m, "tool_calls", None)
        if not tcs:
            continue
        for tc in tcs:
            fn = tc["function"] if isinstance(tc, dict) else tc.function
            name = fn["name"] if isinstance(fn, dict) else fn.name
            raw = fn["arguments"] if isinstance(fn, dict) else fn.arguments
            try:
                args = json.loads(raw)
            except Exception:
                args = {}
            yield (name, args)


def _already_did(messages, call):
    if not call:
        return False
    return any(_call_eq(call, na) for na in _iter_assistant_calls(messages))


def _primary_attempts(messages, script):
    prim = script.get("primary")
    return sum(1 for na in _iter_assistant_calls(messages)
               if _call_eq(prim, na))


def _last_tool_was_error(messages):
    for m in reversed(messages):
        role = m.get("role") if isinstance(m, dict) else getattr(m, "role", "")
        if role == "tool":
            content = m.get("content", "") if isinstance(m, dict) else ""
            return "ERROR" in str(content)
    return False
