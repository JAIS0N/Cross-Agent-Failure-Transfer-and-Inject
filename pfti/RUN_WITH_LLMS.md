# Running PFTI with real open-source LLMs (Ollama)

Nothing here needs a paid API. Everything runs locally and free.

## 1. Install Ollama (one time)

Download from https://ollama.com and install. Then pull two models with a
size/accuracy tradeoff:

```bash
ollama pull qwen2.5:3b       # the cheap / weak "small" model
ollama pull qwen2.5:14b      # the strong "large" model
#   (if 14b is too big for your machine, use:  ollama pull llama3.1:8b
#    and set  PFTI_LARGE=llama3.1:8b  before running)
```

Ollama runs a local server at http://localhost:11434 automatically.

## 2. Install the Python client

```bash
pip install openai pyyaml matplotlib
```

## 3a. Failure-transfer demo with real agents

```bash
python -m pfti.demo_llm
```

FileOps (a real LLM) is asked to write into read-only /data and fails.
The failure is stored. Coder (a real LLM) is then asked to do the same
class of task; the peer-failure warning is injected into its prompt
before it acts, and you watch it route around the failure on its own.
Trace saved to llm_demo_trace.jsonl.

Env overrides: PFTI_MODEL (default qwen2.5:7b), PFTI_BASE_URL.

## 3b. Model-routing experiment with real models

```bash
python -m pfti.routing.experiment_llm
```

Runs a small gradeable benchmark (extract / classify tasks) under three
conditions (always-small, always-large, PFTI-routed) using the two real
Ollama models, with real measured latency. Prints success / cost /
latency per condition.

Env overrides: PFTI_SMALL (default qwen2.5:3b), PFTI_LARGE (default
qwen2.5:14b).

## 4. Verify the integration without Ollama

The LLM code paths (prompting, grading, routing) are unit-tested with a
fake client, so you can confirm correctness before pulling any model:

```bash
python -m pytest pfti/tests/test_llm_paths.py -q
```

## Notes

- First call to each model is slow (model loads into memory); later calls
  are faster. For a smooth demo, run once beforehand to warm the models.
- On a laptop without a GPU, 3b runs fine on CPU; 14b will be slow —
  use llama3.1:8b as the "large" model instead.
- The mechanism is identical to the deterministic demos; only the actor
  (scripted/mock -> real LLM) changes. If a model ignores a warning, that
  is a measurable finding (the "ignore rate"), not a failure of the code.
