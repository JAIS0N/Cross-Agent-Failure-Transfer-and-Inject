"""Real open-source LLM models for the routing extension (via Ollama).

A "model" here is an actual local LLM (e.g. qwen2.5:3b as the cheap/weak
model, qwen2.5:14b as the expensive/strong one). We measure REAL latency
with a wall-clock timer, and grade the answer against the gold label, so
the routing experiment runs on genuine models instead of mock profiles.

The framework is unchanged: same PerfStore, Router, signatures. Only the
`.run()` that produces (answer, latency, cost, ok) is swapped.

Setup:
  install Ollama (https://ollama.com)
  ollama pull qwen2.5:3b
  ollama pull qwen2.5:14b     # or llama3.1:8b if 14b is too big
  pip install openai
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass

BASE_URL = os.environ.get("PFTI_BASE_URL", "http://localhost:11434/v1")

# name -> (ollama model tag, relative cost). Cost ~ proportional to size.
LLM_SPEC = {
    "small": (os.environ.get("PFTI_SMALL", "qwen2.5:3b"), 1.0),
    "large": (os.environ.get("PFTI_LARGE", "qwen2.5:14b"), 6.0),
}

PROMPTS = {
    "extract": ("Extract the requested field. Reply with ONLY the value.\n"
                "Record: {input}\nField: {q}"),
    "classify": ("Classify the item. Reply with ONLY one label from "
                 "{labels}.\nItem: {input}"),
    "summarize": ("Summarize in the exact required format. Reply with ONLY "
                  "the answer.\nText: {input}\nFormat: {q}"),
}


@dataclass
class LLMModel:
    name: str
    ollama_tag: str
    cost: float
    client: object

    def run(self, task_type, item, rng=None):
        prompt = PROMPTS[task_type].format(
            input=item.get("input", item.get("gold", "")),
            q=item.get("question", "the answer"),
            labels=item.get("labels", "[a, b]"))
        t0 = time.time()
        resp = self.client.chat.completions.create(
            model=self.ollama_tag,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0)
        latency_ms = int((time.time() - t0) * 1000)
        raw = (resp.choices[0].message.content or "").strip()
        answer = _normalize(raw)
        ok = (answer == _normalize(item["gold"]))
        return answer, latency_ms, self.cost, ok


def _normalize(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s.strip()


def build_llm_models():
    from openai import OpenAI
    client = OpenAI(base_url=BASE_URL, api_key="ollama")
    return {name: LLMModel(name, tag, cost, client)
            for name, (tag, cost) in LLM_SPEC.items()}
