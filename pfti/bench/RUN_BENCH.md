# Real-LLM before/after benchmark

This produces the **model × mode** matrix for PFTI's core claim, plus the
routing before/after — with real open-source LLMs and no paid API.

| model        | off (vanilla) | shadow (measure) | inject (PFTI) |
|--------------|:-:|:-:|:-:|
| qwen2.5:3b   | ✓ | ✓ | ✓ |
| qwen2.5:7b   | ✓ | ✓ | ✓ |
| qwen2.5:14b  | ✓ | ✓ | ✓ |
| llama3.1:8b  | ✓ | ✓ | ✓ |

- **off** — vanilla multi-agent team, no cross-agent memory (the *before*).
- **shadow** — the matcher runs and is scored against the ground-truth
  oracle (precision/recall) but never injects: pure measurement, behaves
  exactly like `off`.
- **inject** — a peer-failure warning is delivered *before* the next
  agent's matching call (the *after*). Advisory: the agent may proceed
  anyway, which is recorded as an *ignored* warning.

## Quick start (Windows)

From the folder that **contains** `pfti/`:

```bat
run_pfti_bench.bat
```

That checks Python, installs deps, pulls the four models, and runs the
matrix. Results land in `bench_results/`.

## Manual / macOS / Linux

```bash
# 1. install Ollama:  https://ollama.com   (it serves on localhost:11434)
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
ollama pull llama3.1:8b
ollama pull qwen2.5:14b            # optional; slow on CPU

pip install openai pyyaml matplotlib
python -m pfti.bench.run_all --out bench_results
```

## Options

```bash
# fewer / different models (e.g. skip the heavy 14b)
PFTI_MODELS="qwen2.5:3b,qwen2.5:7b,llama3.1:8b" python -m pfti.bench.run_all

# average each cell over N repeats (LLMs are stochastic across loads)
python -m pfti.bench.run_all --repeats 3

# prove the pipeline with NO model pulled (deterministic fake client)
python -m pfti.bench.run_all --fake

# non-default Ollama endpoint (vLLM / LM Studio also work)
PFTI_BASE_URL=http://localhost:11434/v1 python -m pfti.bench.run_all
```

## Outputs (`bench_results/`)

- `summary.md` — the full matrix + headline off→inject deltas + routing table
- `core_matrix.json`, `routing.json`, `meta.json` — raw numbers
- `core_before_after.png` — success & repeated-peer, vanilla vs PFTI
- `core_shadow_pr.png` — matcher precision/recall vs the oracle
- `routing_tradeoff.png` — cost/accuracy, baselines vs PFTI-routed

Send the whole `bench_results/` folder back and it becomes the final
combined report.

## Notes

- First call to each model is slow (it loads into RAM); the runner tolerates
  it. For a smooth run, `ollama run qwen2.5:3b hi` once beforehand to warm up.
- `qwen2.5:14b` needs ~9 GB and is slow without a GPU. If it stalls, drop it
  via `PFTI_MODELS` — the matrix still tells the story with 3b/7b/8b.
- If a model isn't pulled, its row is recorded with an error and the rest of
  the matrix still runs.
- A model ignoring a warning is a *measurable finding* (the ignore rate),
  not a bug — it's reported in the matrix.

## After review (Sept 2026)

```
python -m pfti.bench.run_all --scenarios all --repeats 1 --save-traces --no-routing --out bench_results_v2
python -m pfti.bench.trace_report bench_results_v2
```

* `--scenarios v1|v2|all` — v1 = the original five; v2 adds 3 adversarial
  false-positive scenarios, 1 control and 2 reasoning-bug scenarios.
* `--save-traces` — every scenario run's events and agent transcripts go to
  `<out>/traces/`; `trace_report` classifies what agents did after each
  warning and attributes every off-success/inject-failure regression.
* New per-cell metrics: `held_tp` / `held_fp` (calls a warning held that
  would / would not have failed), `doomed_attempts`, `peer_doomed_attempts`
  and `peer_doomed_executed` (oracle-defined: a different agent already
  failed with the same tool + error class). `repeated_peer_failures` is kept
  for comparability but is circular under inject (<= ignored warnings).
