# PFTI before/after results (real Ollama)

_Generated 2026-09-22 21:37:41 · models: qwen2.5:3b, qwen2.5:7b, qwen2.5:14b, llama3.1:8b · 714.3s_


## Core matrix (model × mode)

| model | mode | success | failures | repeat-peer | warns | ignored | tokens | shadow P/R |
|---|---|--:|--:|--:|--:|--:|--:|--:|
| qwen2.5:3b | off | 0% | 17 | 6 | 0 | 0 | 24622 | 1.00/0.00 |
| qwen2.5:3b | shadow | 0% | 14 | 4 | 0 | 0 | 23404 | 1.00/0.29 |
| qwen2.5:3b | inject | 0% | 13 | 3 | 2 | 3 | 65716 | 1.00/0.33 |
| qwen2.5:7b | off | 100% | 0 | 0 | 0 | 0 | 15721 | 1.00/1.00 |
| qwen2.5:7b | shadow | 100% | 0 | 0 | 0 | 0 | 14494 | 1.00/1.00 |
| qwen2.5:7b | inject | 100% | 0 | 0 | 0 | 0 | 14494 | 1.00/1.00 |
| qwen2.5:14b | off | 100% | 0 | 0 | 0 | 0 | 10751 | 1.00/1.00 |
| qwen2.5:14b | shadow | 100% | 2 | 0 | 0 | 0 | 12882 | 1.00/0.00 |
| qwen2.5:14b | inject | 100% | 2 | 0 | 0 | 0 | 12882 | 1.00/0.00 |
| llama3.1:8b | off | 50% | 3 | 2 | 0 | 0 | 10635 | 1.00/0.00 |
| llama3.1:8b | shadow | 50% | 3 | 2 | 0 | 0 | 10684 | 1.00/0.67 |
| llama3.1:8b | inject | 50% | 1 | 0 | 2 | 0 | 12253 | 1.00/0.67 |

## Headline: vanilla (off) → PFTI (inject)

| model | success off→inject | repeat-peer off→inject | extra tokens |
|---|--:|--:|--:|
| qwen2.5:3b | 0% → 0% | 6 → 3 | +41094 (+167%) |
| qwen2.5:7b | 100% → 100% | 0 → 0 | +-1227 (-8%) |
| qwen2.5:14b | 100% → 100% | 0 → 0 | +2131 (+20%) |
| llama3.1:8b | 50% → 50% | 2 → 0 | +1618 (+15%) |

## Figures

- `core_before_after.png` — success & repeated-peer, off vs inject
- `core_shadow_pr.png` — matcher precision/recall vs oracle
- `routing_tradeoff.png` — cost/accuracy, routed vs baselines
