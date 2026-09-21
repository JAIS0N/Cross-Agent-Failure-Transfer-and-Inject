"""PFTI real-LLM before/after benchmark harness.

Produces the model x mode matrix the plan calls for:

    model in {qwen2.5:3b, qwen2.5:7b, qwen2.5:14b, llama3.1:8b}
    mode  in {off, shadow, inject}

`off` is the vanilla multi-agent system (no cross-agent memory), `shadow`
runs the matcher against the ground-truth oracle WITHOUT injecting (pure
measurement -> precision/recall), and `inject` actively warns the next
agent before it acts. off vs inject is the headline before/after; shadow
explains *why* it works (the matcher is accurate).

Everything is provider-agnostic (any OpenAI-compatible server: Ollama,
vLLM, LM Studio) and falls back to a deterministic fake client so the
pipeline is testable with no model pulled.
"""
