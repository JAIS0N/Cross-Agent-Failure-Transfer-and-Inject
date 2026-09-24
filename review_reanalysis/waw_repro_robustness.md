# Who&When result: stress tests

468 phase-2 samples, 39 model×case units, 21 distinct cases from 18 logs (0 from the hand-crafted half).

All GEE rows: logistic regression, clustered by log, model as fixed effect.

## 1. How 'no runnable code' is counted

| variant | repeat off → inject | GEE |
|---|--:|--:|
| primary: no code = not a repeat | 57% → 44% | OR 0.58 [0.37, 0.91], p=0.018 |
| strict: no code = a repeat | 63% → 53% | OR 0.64 [0.41, 1.00], p=0.049 |
| graded samples only | 61% → 48% | OR 0.60 [0.37, 0.96], p=0.033 |
| *any failure* (not just the same error) | 82% → 79% | OR 0.85 [0.61, 1.18], p=0.335 |

## 2. What counts as one unit

- model × case (as pre-specified): 39 units, better/worse 25/10, mean -13.2 pts, sign p=0.017, Wilcoxon p=0.015
- distinct case (averaged over models): 21 units, better/worse 12/6, mean -9.7 pts, sign p=0.238, Wilcoxon p=0.081

## 3. Leave one model out

| dropped | repeat off → inject | GEE |
|---|--:|--:|
| qwen2.5:7b | 57% → 45% | OR 0.60 [0.36, 1.00], p=0.049 |
| qwen2.5:3b | 60% → 49% | OR 0.63 [0.38, 1.05], p=0.074 |
| qwen2.5:14b | 53% → 40% | OR 0.57 [0.32, 1.00], p=0.050 |
| llama3.1:8b | 58% → 43% | OR 0.53 [0.36, 0.79], p=0.002 |

## 4. By historical error class

| error class | units | repeat off → inject | without this class: GEE |
|---|--:|--:|--:|
| FileNotFoundError | 16 | 67% → 39% | OR 0.89 [0.56, 1.40], p=0.611 |
| AttributeError | 5 | 43% → 63% | OR 0.47 [0.31, 0.71], p=0.000 |
| NameError | 5 | 60% → 47% | OR 0.58 [0.35, 0.96], p=0.034 |
| ModuleNotFoundError | 4 | 50% → 25% | OR 0.61 [0.37, 1.00], p=0.051 |
| NotFound | 3 | 50% → 50% | OR 0.55 [0.34, 0.89], p=0.016 |
| KeyError | 2 | 67% → 75% | OR 0.55 [0.35, 0.88], p=0.013 |
| ValueError | 2 | 42% → 42% | OR 0.56 [0.35, 0.91], p=0.019 |
| HTTPError | 1 | 33% → 17% | OR 0.58 [0.36, 0.92], p=0.021 |
| ExecutionError | 1 | 50% → 50% | OR 0.57 [0.36, 0.91], p=0.018 |

## 5. What happened instead of the repeat

Share of samples. FileNotFoundError is shown separately because the effect is concentrated there.

| subset | arm | same error | different error | ran OK | no runnable code |
|---|---|--:|--:|--:|--:|
| FileNotFoundError cases | off | 67% | 11% | 19% | 3% |
| FileNotFoundError cases | inject | 39% | 31% | 26% | 4% |
| all other cases | off | 51% | 23% | 18% | 8% |
| all other cases | inject | 48% | 23% | 17% | 12% |
| all | off | 57% | 18% | 18% | 6% |
| all | inject | 44% | 26% | 21% | 9% |
