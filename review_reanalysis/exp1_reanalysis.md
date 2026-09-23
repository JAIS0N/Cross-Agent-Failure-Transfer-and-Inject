## 1. Pooled outcomes (60 trials per mode)

| metric | off | shadow | inject | inject vs off |
|---|--:|--:|--:|--:|
| task success | 49 | 50 | 47 | -4% |
| executed tool failures | 198 | 204 | 150 | -24% |
| repeated peer failures (matcher-defined) | 56 | 66 | 3 | -95% |
| warnings delivered | 0 | 0 | 65 | — |
| ignored warnings | 0 | 0 | 3 | — |
| total tokens | 340,732 | 348,446 | 364,240 | +7% |

## 2. Noise floor: shadow minus off, every metric

Shadow never changes what the agent sees, so any shadow–off gap is run-to-run nondeterminism (Ollama on GPU is not bit-reproducible even at temperature 0).

| model | success | failures | repeated peer | tokens |
|---|--:|--:|--:|--:|
| qwen2.5:3b | +0/15 | +1 (+2%) | +0 (+0%) | +0.0% |
| qwen2.5:7b | +0/15 | +0 (+0%) | +1 (+5%) | +3.8% |
| qwen2.5:14b | +1/15 | +5 (+9%) | +9 (+75%) | +4.0% |
| llama3.1:8b | +0/15 | +0 (+0%) | +0 (+0%) | +0.4% |

Inject-vs-off effect next to the largest shadow–off gap for the same model:

| model | Δ success (inject−off) | Δ failures | Δ tokens | |shadow−off| failures |
|---|--:|--:|--:|--:|
| qwen2.5:3b | +0/15 | -11 | +4.0% | 1 |
| qwen2.5:7b | +3/15 | -16 | +7.8% | 0 |
| qwen2.5:14b | -1/15 | -12 | +2.0% | 5 |
| llama3.1:8b | -4/15 | -9 | +15.7% | 0 |

## 3. Is 'repeated peer failure' circular under inject?

| model | inject: repeated peer | inject: ignored warnings |
|---|--:|--:|
| qwen2.5:3b | 0 | 0 |
| qwen2.5:7b | 0 | 0 |
| qwen2.5:14b | 0 | 0 |
| llama3.1:8b | 3 | 3 |

## 4. Are agents making fewer bad calls, or are bad calls being held?

In inject, a call that matches a peer failure is *held* (not executed) the first time. A held call cannot be counted as a failure, so part of the failure drop is mechanical. Adding held calls back gives the number of doomed calls the agents actually *attempted*. Exact split of held true- vs false-positives needs traces (not saved in this run), so a range is shown.

| model | off: failures | inject: failures | inject: held calls | inject: attempted bad calls (range) |
|---|--:|--:|--:|--:|
| qwen2.5:3b | 44 | 33 | 15 | 48–48 |
| qwen2.5:7b | 51 | 35 | 19 | 52–54 |
| qwen2.5:14b | 55 | 43 | 16 | 59–59 |
| llama3.1:8b | 48 | 39 | 15 | 48–54 |
| **all** | 198 | 150 | 65 | 207–215 |

## 5. Where can a false-positive warning occur? (exhaustive check on the testbed)

For each scenario: replay the seed agent's failing call into the store, then try a grid of plausible calls from another agent and count calls the matcher warns on that the oracle says would succeed.

- **perm_denied**: 8 warned calls that would fail, **0 warned calls that would succeed**
- **notfound_dir**: 4 warned calls that would fail, **0 warned calls that would succeed**
- **rate_limit**: 1 warned calls that would fail, **0 warned calls that would succeed**
- **quota**: 10 warned calls that would fail, **6 warned calls that would succeed** — e.g. write_file /tmp/x.txt (content 1 chars); write_file /tmp/x.txt (content 15 chars); write_file /tmp/b.txt (content 1 chars)
- **missing_table**: 1 warned calls that would fail, **0 warned calls that would succeed**
