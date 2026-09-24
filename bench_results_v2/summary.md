# PFTI before/after results (real Ollama)

_Generated 2026-09-22 20:18:15 · models: qwen2.5:3b, qwen2.5:7b, qwen2.5:14b, llama3.1:8b · 2453.4s_


## Core matrix (model × mode)

| model | mode | success | failures | repeat-peer | warns | ignored | tokens | shadow P/R |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| qwen2.5:3b | off | 82% | 34 | 11 | 0 | 0 | 72532 | 1.00/0.00 |
| qwen2.5:3b | shadow | 82% | 35 | 11 | 0 | 0 | 72668 | 0.85/0.31 |
| qwen2.5:3b | inject | 82% | 34 | 9 | 13 | 9 | 83186 | 0.83/0.44 |
| qwen2.5:7b | off | 73% | 48 | 21 | 0 | 0 | 77640 | 1.00/0.00 |
| qwen2.5:7b | shadow | 73% | 48 | 21 | 0 | 0 | 77640 | 0.88/0.44 |
| qwen2.5:7b | inject | 82% | 32 | 7 | 14 | 7 | 83892 | 0.83/0.43 |
| qwen2.5:14b | off | 64% | 54 | 12 | 0 | 0 | 69021 | 1.00/0.00 |
| qwen2.5:14b | shadow | 64% | 53 | 16 | 0 | 0 | 68878 | 0.84/0.30 |
| qwen2.5:14b | inject | 64% | 43 | 1 | 13 | 1 | 77866 | 0.58/0.21 |
| llama3.1:8b | off | 73% | 29 | 6 | 0 | 0 | 47486 | 1.00/0.00 |
| llama3.1:8b | shadow | 73% | 29 | 6 | 0 | 0 | 47808 | 0.67/0.21 |
| llama3.1:8b | inject | 55% | 14 | 1 | 8 | 1 | 53632 | 0.70/0.35 |

## Headline: vanilla (off) → PFTI (inject)

| model | success off→inject | repeat-peer off→inject | extra tokens |
|---|--:|--:|--:|
| qwen2.5:3b | 82% → 82% | 11 → 9 | +10654 (+15%) |
| qwen2.5:7b | 73% → 82% | 21 → 7 | +6252 (+8%) |
| qwen2.5:14b | 64% → 64% | 12 → 1 | +8845 (+13%) |
| llama3.1:8b | 73% → 55% | 6 → 1 | +6146 (+13%) |

## Figures

- `core_before_after.png` — success & repeated-peer, off vs inject
- `core_shadow_pr.png` — matcher precision/recall vs oracle
- `routing_tradeoff.png` — cost/accuracy, routed vs baselines
