# Who&When reproduction-filtered study

90 code-failure moments from 65 logs. Screen: 6 off samples per case at T=0.7; keep cases that repeat the historical error ≥2/6. Test: fresh 6 off + 6 inject samples per kept case.

## Screening (baseline reproduction)

| model | cases screened | baseline repeat rate (all samples) | cases kept |
|---|--:|--:|--:|
| qwen2.5:3b | 90 | 49/539 (9%) | 11 |
| qwen2.5:7b | 90 | 61/540 (11%) | 13 |
| qwen2.5:14b | 90 | 44/540 (8%) | 10 |
| llama3.1:8b | 90 | 31/540 (6%) | 5 |

## Phase 2: off vs inject on kept cases (fresh samples)

| model | kept cases | repeat off → inject (ITT) | repeat, per-protocol | any failure off → inject (ITT) | no runnable code off / inject |
|---|--:|--:|--:|--:|--:|
| qwen2.5:3b | 11 | 50% → 32% (n=66/66) | 52% → 35% | 83% → 73% | 3 / 6 |
| qwen2.5:7b | 13 | 58% → 42% (n=78/78) | 60% → 45% | 79% → 73% | 3 / 5 |
| qwen2.5:14b | 10 | 68% → 57% (n=60/60) | 73% → 61% | 85% → 92% | 4 / 4 |
| llama3.1:8b | 5 | 50% → 50% (n=30/30) | 58% → 60% | 77% → 83% | 4 / 5 |
| **all models** | 39 | 57% → 44% (n=234/234) | 61% → 48% | 82% → 79% | 14 / 20 |

## Case-level paired comparison (each model × case is one unit)

- units: 39; inject lower repeat rate in **25**, higher in **10**, tied in 4
- mean change in repeat rate: -13.2 pts; Wilcoxon signed-rank p=0.0154; sign test p=0.0167

## Clustered models (primary outcome: repeat, ITT)

| method | inject effect | 95% CI | p |
|---|--:|--:|--:|
| GEE logistic, clustered by log | OR 0.58 | [0.37, 0.91] | 0.0182 |
| mixed-effects logistic, random case + log (VB) | OR 0.54 | [0.41, 0.71] | — |
| cluster bootstrap over logs | -13.2 pts | [-23.2, -2.2] | — |

## Moderators

- same-class failure already visible in context: repeat 62% → 54% (4 model×case units)
- not visible in context: repeat 57% → 43% (35 model×case units)

By historical error class (repeat rate off → inject):

- AttributeError: 43% → 63% (n=30 per arm)
- ExecutionError: 50% → 50% (n=6 per arm)
- FileNotFoundError: 67% → 39% (n=96 per arm)
- HTTPError: 33% → 17% (n=6 per arm)
- KeyError: 67% → 75% (n=12 per arm)
- ModuleNotFoundError: 50% → 25% (n=24 per arm)
- NameError: 60% → 47% (n=30 per arm)
- NotFound: 50% → 50% (n=18 per arm)
- ValueError: 42% → 42% (n=12 per arm)

Inject responses that engage with the warning (keyword check): 45%
