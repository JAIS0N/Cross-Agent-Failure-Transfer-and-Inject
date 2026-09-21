"""PFTI with REAL open-source LLM agents via an OpenAI-compatible server.

Works with Ollama (recommended), vLLM, or LM Studio. Setup:

  1. install ollama  (https://ollama.com)
  2. ollama pull qwen2.5:7b        # any tool-calling model works
  3. pip install openai pyyaml
  4. python -m pfti.demo_llm       # from the folder containing pfti/

Env overrides:
  PFTI_BASE_URL (default http://localhost:11434/v1)
  PFTI_MODEL    (default qwen2.5:7b)

Three agents (FileOps, Coder) driven by the SAME PFTI hook as the
scripted runner. Warnings are injected as a system message immediately
before the agent's next generation (plan §2.5).
"""
from __future__ import annotations

import json
import os

from .core import FailureStore, Matcher, Injector, PFTIHook, TraceLogger
from .envs.tools import World, ToolError, TOOLS, would_fail

BASE_URL = os.environ.get("PFTI_BASE_URL", "http://localhost:11434/v1")
MODEL = os.environ.get("PFTI_MODEL", "qwen2.5:7b")

TOOL_SCHEMAS = [
    {"type": "function", "function": {
        "name": "write_file",
        "description": "Write a file at an absolute path.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"]}}},
    {"type": "function", "function": {
        "name": "list_dir",
        "description": "List files in a directory.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string"}}, "required": ["path"]}}},
]


def run_agent(client, agent_id, task, hook, world, step_offset=0,
              max_turns=6):
    msgs = [{"role": "system", "content":
             f"You are {agent_id}, an agent in a software-ops team. "
             "Complete the task using the tools. If a warning about a "
             "peer's failure appears, take it seriously: verify or choose "
             "an alternative. Reply DONE when finished."},
            {"role": "user", "content": task}]
    step = step_offset
    for _ in range(max_turns):
        # deliver any pending peer-failure warnings BEFORE generation
        for w in hook.injector.drain(agent_id):
            msgs.append({"role": "system", "content": w})
            print(f"    [injected into {agent_id}]\n      " +
                  w.replace("\n", "\n      "))
        resp = client.chat.completions.create(
            model=MODEL, messages=msgs, tools=TOOL_SCHEMAS, temperature=0.3)
        m = resp.choices[0].message
        if not m.tool_calls:
            print(f"  {agent_id}: {m.content}")
            if m.content and "DONE" in m.content.upper():
                return step
            msgs.append({"role": "assistant", "content": m.content or ""})
            continue
        msgs.append(m)
        for tc in m.tool_calls:
            step += 1
            name = tc.function.name
            args = json.loads(tc.function.arguments)
            hook.before_call(agent_id, name, args, world, "llm-demo", step)
            # warnings armed now are drained before the NEXT generation
            try:
                out = TOOLS[name](world, **args)
                hook.after_call(agent_id, name, args, world, None,
                                "llm-demo", step)
                result = str(out)
                print(f"  {agent_id}: {name}({args}) -> OK")
            except ToolError as e:
                hook.after_call(agent_id, name, args, world, e,
                                "llm-demo", step)
                result = f"ERROR {e.error_class}: {e.msg}"
                print(f"  {agent_id}: {name}({args}) -> {result}")
            msgs.append({"role": "tool", "tool_call_id": tc.id,
                         "content": result})
    return step


def main():
    from openai import OpenAI  # pip install openai
    client = OpenAI(base_url=BASE_URL, api_key="ollama")

    world = World(dirs=["/", "/data", "/tmp"],
                  perm={"/data": "ro", "/tmp": "rw"})
    store = FailureStore()
    hook = PFTIHook(store, Matcher(level="mid", mode="subsumption"),
                    Injector(level="mid"), mode="inject",
                    logger=TraceLogger("llm_demo_trace.jsonl"),
                    oracle=would_fail)

    print("== Agent 1: FileOps (will hit PermissionDenied on /data) ==")
    s = run_agent(client, "FileOps",
                  "Save the text 'q1 numbers' to /data/report.csv.",
                  hook, world)
    print(f"\n  failure store now has {len(store.records)} record(s)\n")

    print("== Agent 2: Coder (about to make the SAME class of mistake) ==")
    run_agent(client, "Coder",
              "Save the text 'summary' to /data/summary.txt.",
              hook, world, step_offset=s)

    print("\nFiles in world:", sorted(world.files))
    print("Trace written to llm_demo_trace.jsonl")


if __name__ == "__main__":
    main()
