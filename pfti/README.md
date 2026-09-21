# PFTI — Predicate-Guided Cross-Agent Failure Transfer & Injection

When one agent in a multi-agent system fails a tool call, PFTI converts the
failure into a **predicate signature**, stores it, and **injects a live
warning** into any peer agent about to make a matching call — turning
post-hoc failure attribution into online, act-time failure prevention.

## Layout (matches the plan)

```
pfti/
  core/       signature.py  store.py  matcher.py  inject.py  hooks.py  trace.py
  envs/       tools.py (6 mock tools + ground-truth oracle)  scenarios/*.yaml
  eval/       runner.py  shadow.py  metrics.py
  baselines/  (week 2: loop_guard, broadcast, vector_mem, static_precond)
  tests/      test_pfti.py (13 tests, incl. Day-2 acceptance tests)
  demo_figure1.py   deterministic Figure-1 demo (no LLM needed)
  demo_llm.py       same mechanism with real open-source LLM agents
```

## Quickstart (no LLM required)

From the folder **containing** `pfti/`:

```bash
pip install pyyaml pytest
python -m pytest pfti/tests -q        # 13 tests
python -m pfti.demo_figure1           # the full demo
```

The demo shows, in order:
A. vanilla — two agents hit the same PermissionDenied rake;
B. PFTI (G-mid, subsumption) — Coder is warned about FileOps' failure
   *before acting* and succeeds via /tmp (this is Figure 1);
C. G-fine equality — nothing matches, no help (motivates granularity);
D. shadow-mode precision/recall vs the oracle per granularity level —
   coarse: recall 1.0 / precision <1 (false warnings);
   mid: the interior optimum; fine: precision 1.0 / recall ~0.17;
E. full 5-scenario suite: vanilla vs PFTI success & repeated-peer-failures.

Section D is the empirical face of Lemma 1 (monotonicity) and Lemma 2
(interior optimum).

## Demo with real open-source LLM agents (Ollama)

```bash
# 1. install ollama: https://ollama.com
ollama pull qwen2.5:7b               # any tool-calling model works
pip install openai
python -m pfti.demo_llm
```

Env: `PFTI_MODEL` (default qwen2.5:7b), `PFTI_BASE_URL`
(default http://localhost:11434/v1 — also works with vLLM / LM Studio).

Agent 1 (FileOps) is asked to write into read-only `/data` and fails;
the failure is harvested. Agent 2 (Coder) is asked to do the same class
of task; the warning is injected as a system message before its next
generation, and you watch the model route around the failure.

## Key invariants (tested)

- An agent is **never** warned about its own records (self = loop-guard
  baseline B2, not the contribution). Cross-agent transfer is the product.
- Warnings are **advisory**, max 2 per step, human-readable signature.
- Store holds **only failures**, written only by the hook, read only at
  tool-call time.
- Coarse ⊑ mid ⊑ fine are projections of one G-fine record;
  subsumption is an O(|preds|) frozenset-subset check.
- Shadow mode changes nothing behaviorally (tested) — pure measurement.

## Next steps (week 2 of the plan)

Baselines B1–B5 in `baselines/`, 25-scenario suite, granularity ablation
grid (mode × level × matcher × seeds), lemmas in LaTeX.

---

# Routing Extension — predicate-guided model routing from shared trials

Same PFTI mechanism, richer record. Instead of storing only *what failed*,
each agent stores *how a model performed* on a subtask:

```
(task_type=extract, flavor=table, model=small) -> {ok, latency, cost}
```

A different agent facing a matching subtask reads peers' records and
routes to the model observed to work best — learning the routing policy
from teammates' trials instead of rediscovering it.

## Run

```bash
pip install pyyaml matplotlib
python -m pfti.demo_routing              # the 3-condition demo + convergence
python -m pfti.routing.plot             # saves routing_tradeoff.png
python -m pytest pfti/tests/test_routing.py -q
```

## The experiment

Agents solve **gradeable** subtasks (extract / classify / summarize; each
carries a gold answer, so success = answer == gold — no LLM judge). Two
models with a per-task tradeoff: `small` (cheap, best at classify) vs
`large` (expensive, best at extract/summarize). Three conditions over the
same batch:

| condition | success | cost | what it is |
|-----------|--------:|-----:|------------|
| always-small | ~58% | 40 | cheap floor |
| always-large | ~84% | 240 | expensive ceiling |
| **PFTI-routed** | ~74% | 136 | learns the policy from peer records |

**Headline:** routed captures ~58% of the accuracy gain of the big model
for ~48% of its extra cost — the "good corner" of the cost/accuracy plot
(`routing_tradeoff.png`). It converges: early subtasks explore, later ones
route from peer knowledge (use-of-peer-knowledge rises 15% -> 77% across
the batch).

## Files

```
routing/
  models.py       mock models with defined acc/latency/cost + oracle
  router.py       PerfStore + Router (explore-then-exploit, self-excluded)
  experiment.py   the 3-condition experiment
  plot.py         cost-vs-accuracy figure
demo_routing.py   the runnable demo
```

Granularity still applies (coarse = task_type only, mid = +model,
fine = +flavor), so Lemmas 1–2 carry over to routing decisions. Ground
truth (`best_model_oracle`) lets you measure whether routing converges to
the optimal policy — the analogue of shadow-mode precision/recall.

To use REAL open-source models: swap `MockModel.run` for an Ollama call
(`qwen2.5:3b` as small, `qwen2.5:14b` as large) and measure real latency
with `time.time()`. The framework is unchanged.
