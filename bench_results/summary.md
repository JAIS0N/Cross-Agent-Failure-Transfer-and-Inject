# PFTI before/after results (real Ollama)

_Generated 2026-09-21 12:56:53 · models: qwen2.5:3b, qwen2.5:7b, qwen2.5:14b, llama3.1:8b · 3988.3s_


## Core matrix (model × mode)

| model | mode | success | failures | repeat-peer | warns | ignored | tokens | shadow P/R |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| qwen2.5:3b | off | 100% | 44 | 12 | 0 | 0 | 85077 | 1.00/0.00 |
| qwen2.5:3b | shadow | 100% | 45 | 12 | 0 | 0 | 85089 | 1.00/0.27 |
| qwen2.5:3b | inject | 100% | 33 | 0 | 15 | 0 | 88461 | 1.00/0.31 |
| qwen2.5:7b | off | 80% | 51 | 20 | 0 | 0 | 101861 | 1.00/0.00 |
| qwen2.5:7b | shadow | 80% | 51 | 21 | 0 | 0 | 105777 | 1.00/0.41 |
| qwen2.5:7b | inject | 100% | 35 | 0 | 19 | 0 | 109843 | 0.90/0.34 |
| qwen2.5:14b | off | 67% | 55 | 12 | 0 | 0 | 87787 | 1.00/0.00 |
| qwen2.5:14b | shadow | 73% | 60 | 21 | 0 | 0 | 91298 | 1.00/0.35 |
| qwen2.5:14b | inject | 60% | 43 | 0 | 16 | 0 | 89536 | 1.00/0.27 |
| llama3.1:8b | off | 80% | 48 | 12 | 0 | 0 | 66007 | 1.00/0.00 |
| llama3.1:8b | shadow | 80% | 48 | 12 | 0 | 0 | 66282 | 0.80/0.25 |
| llama3.1:8b | inject | 53% | 39 | 3 | 15 | 3 | 76400 | 0.71/0.29 |

## Headline: vanilla (off) → PFTI (inject)

| model | success off→inject | repeat-peer off→inject | extra tokens |
|---|--:|--:|--:|
| qwen2.5:3b | 100% → 100% | 12 → 0 | +3384 (+4%) |
| qwen2.5:7b | 80% → 100% | 20 → 0 | +7982 (+8%) |
| qwen2.5:14b | 67% → 60% | 12 → 0 | +1749 (+2%) |
| llama3.1:8b | 80% → 53% | 12 → 3 | +10393 (+16%) |

## Routing (always-model vs PFTI-routed)

| policy | success | cost | latency (s) |
|---|--:|--:|--:|
| qwen2.5:3b | 92% | 12.0 | 6.7 |
| qwen2.5:7b | 92% | 27.6 | 6.8 |
| qwen2.5:14b | 92% | 56.4 | 16.1 |
| llama3.1:8b | 67% | 32.4 | 7.2 |
| routed | 75% | 24.0 | 63.3 |

## Figures

- `core_before_after.png` — success & repeated-peer, off vs inject
- `core_shadow_pr.png` — matcher precision/recall vs oracle
- `routing_tradeoff.png` — cost/accuracy, routed vs baselines
