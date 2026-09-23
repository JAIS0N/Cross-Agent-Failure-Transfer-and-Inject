## 1. What was excluded, by arm

| model | arm | graded | no code block | shell block | unsafe (skipped) |
|---|---|--:|--:|--:|--:|
| qwen2.5:3b | off | 35 | 0 | 1 | 0 |
| qwen2.5:3b | inject | 31 | 0 | 4 | 1 |
| qwen2.5:7b | off | 27 | 2 | 6 | 0 |
| qwen2.5:7b | inject | 28 | 0 | 7 | 0 |
| qwen2.5:14b | off | 26 | 2 | 8 | 0 |
| qwen2.5:14b | inject | 26 | 2 | 8 | 0 |
| llama3.1:8b | off | 29 | 1 | 6 | 0 |
| llama3.1:8b | inject | 31 | 1 | 3 | 1 |
| **all** | off | 117 | 5 | 21 | 0 |
| **all** | inject | 116 | 3 | 22 | 2 |

## 2. Paired analysis (McNemar, exact)

Each case is run under both arms, so the comparison is within-case. Only the discordant pairs carry information: *helped* = repeat/failure off but not inject; *hurt* = the reverse.

### 2a. Per-protocol (both arms produced runnable Python)

| model | pairs | same-error: helped / hurt | p | fail: helped / hurt | p |
|---|--:|--:|--:|--:|--:|
| qwen2.5:3b | 31 | 1 / 1 | 1.00 | 4 / 3 | 1.00 |
| qwen2.5:7b | 25 | 3 / 0 | 0.25 | 4 / 1 | 0.38 |
| qwen2.5:14b | 24 | 0 / 1 | 1.00 | 0 / 2 | 0.50 |
| llama3.1:8b | 27 | 0 / 1 | 1.00 | 3 / 3 | 1.00 |
| **all 4 models** | 107 | 4 / 3 | 1.00 | 11 / 9 | 0.82 |

### 2b. Intention-to-treat (every case counted; no runnable code = failure)

Same-error under ITT has no single right answer for a missing block, so both bounds are shown: *lenient* counts a missing block as not-a-repeat, *strict* counts it as a repeat.

| model | n pairs | fail rate off → inject | fail: helped / hurt | p | same-err lenient off → inj | p | same-err strict off → inj | p |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| qwen2.5:3b | 36 | 78% → 75% | 4 / 3 | 1.00 | 8.3% → 8.3% | 1.00 | 11% → 22% | 0.22 |
| qwen2.5:7b | 36 | 83% → 72% | 5 / 1 | 0.22 | 11.1% → 5.6% | 0.62 | 36% → 28% | 0.45 |
| qwen2.5:14b | 36 | 83% → 83% | 2 / 2 | 1.00 | 8.3% → 11.1% | 1.00 | 36% → 39% | 1.00 |
| llama3.1:8b | 36 | 72% → 67% | 5 / 3 | 0.73 | 0.0% → 2.8% | 1.00 | 19% → 17% | 1.00 |
| **all 4 models** | 144 | 79% → 74% | 16 / 9 | 0.23 | 6.9% → 6.9% | 1.00 | 26% → 26% | 1.00 |

## 3. Clustered models (accounting for 4 models × 36 cases × 25 logs)

The 288 attempts come from 36 cases in 25 logs, re-used across 4 models. Three analyses that respect that structure; all report the inject effect.

| outcome | GEE logistic, clustered by log: OR [95% CI], p | mixed-effects logistic (random log + case): OR [95% CI] | cluster bootstrap by log: risk diff [95% CI] |
|---|---|---|---|
| same-error (per-protocol) | 0.99 [0.56, 1.77], p=0.99 | 0.79 [0.36, 1.76] | +0.1 pts [-4.5, +4.7] |
| fail (per-protocol) | 0.80 [0.52, 1.23], p=0.30 | 0.69 [0.41, 1.16] | -6.3 pts [-17.0, +2.6] |
| fail (ITT) | 0.77 [0.53, 1.12], p=0.18 | 0.66 [0.41, 1.07] | -4.9 pts [-12.5, +2.5] |

## 4. What does the warning actually carry? (confound check)

- Earlier failure is **still inside the context the model sees** (not clipped): 34/36 (94%)
- The earlier failing code block itself is in context: 26/36 (72%)
- Earlier failing code was proposed by the **same agent** that is regenerating now: 13/36 (36%)
- Earlier failure has the **same error class** as the historical failure being tested: 5/36 (14%)
- Median gap between the earlier failure and the regenerated turn: 1 turns
- Agent attributed to the stored failure record: {'pythondebugging_expert': 5, 'computer_terminal': 29, 'json_expert': 2}
- Warning is fully redundant (in context AND same agent): 13/36
- Warning is genuinely new peer information (out of context AND different agent): 2/36; of those, same error class as the target failure: 0/36

### Effect restricted to cases where the warning names the same error class

- same error class: 19 graded pairs, same-error 5.3% → 5.3%, helped/hurt 1/1, McNemar p=1.00
- different error class: 88 graded pairs, same-error 10.2% → 9.1%, helped/hurt 3/2, McNemar p=1.00

## 5. How often does the baseline reproduce the historical error?

Off-arm same-error count per case, out of the 4 models (temperature 0.2, one sample each):

| models reproducing the historical error | cases |
|---|--:|
| 0 of 4 | 29 |
| 1 of 4 | 5 |
| 2 of 4 | 1 |
| 3 of 4 | 1 |
