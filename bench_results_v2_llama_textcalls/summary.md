# PFTI before/after results (real Ollama)

_Generated 2026-09-22 21:39:55 · models: llama3.1:8b · 132.0s_


## Core matrix (model × mode)

| model | mode | success | failures | repeat-peer | warns | ignored | tokens | shadow P/R |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| llama3.1:8b | off | 91% | 22 | 6 | 0 | 0 | 42338 | 1.00/0.00 |
| llama3.1:8b | shadow | 91% | 22 | 6 | 0 | 0 | 42166 | 0.67/0.27 |
| llama3.1:8b | inject | 91% | 20 | 1 | 9 | 1 | 54525 | 0.54/0.27 |

## Headline: vanilla (off) → PFTI (inject)

| model | success off→inject | repeat-peer off→inject | extra tokens |
|---|--:|--:|--:|
| llama3.1:8b | 91% → 91% | 6 → 1 | +12187 (+29%) |

## Figures

- `core_before_after.png` — success & repeated-peer, off vs inject
- `core_shadow_pr.png` — matcher precision/recall vs oracle
- `routing_tradeoff.png` — cost/accuracy, routed vs baselines
