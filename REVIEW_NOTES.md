# Responding to Prof. Hailu's review: what he asked, what we found, what changed

This file explains the review so you can follow it in the code without relying on the .bat scripts. Line numbers refer to the files as they are in this commit.

---

## 0. Quick refresher: the two experiments

**Experiment 1 (controlled testbed).** Several agents share a fake world (`pfti/envs/tools.py`), which has files, folders with permissions, an API with a rate limit, a disk quota and database tables. Each scenario is set up so that agent A hits a failure first (for example, writing into a read-only folder), and later agents face the same trap. The agents are real Ollama models calling tools. There are three modes:

- `off`: no PFTI.
- `shadow`: the matcher runs and is scored against the oracle, but no warning is ever shown. The agent's behaviour should be identical to `off`.
- `inject`: when an agent proposes a call that matches a peer's earlier failure, the call is **held** (not executed). The agent gets the warning text back in place of a tool result. If it proposes the same call again, the call goes through.

The driver is `pfti/bench/core_matrix.py`. The scenarios are in `pfti/bench/scenarios_llm.py`.

**Experiment 2 (Who&When replay).** These are real multi-agent failure logs (184 of them). At 36 moments where an agent's code failed, we replayed the conversation, had a model regenerate that code turn with and without a warning, ran the result, and checked whether it failed with the same error as history. The code is in `pfti/eval/whoandwhen_llm.py`.

---

## 1. What the professor said, in plain terms

| # | His point | Short version |
|---|---|---|
| 1 | The 56 → 3 headline may be guaranteed by design | The "repeated peer failure" metric can only be non-zero under inject if an agent ignores a warning, so it measures obedience, not prevention. |
| 2 | Lead with the outcome that matters | Success is flat (49/60 vs 47/60). Lead with "failures −24%, success not improved, model-dependent". |
| 3 | The noise floor is understated | Shadow should equal off, but 14b's repeats went 12 → 21. Report the shadow−off gap for every metric. |
| 4 | Look into the llama regression | Guess: false-positive warnings steered llama away from paths that worked. Check the traces. |
| 5 | Minor points | Inject costs up to 16% more tokens. The 5 scenarios suit PFTI by design. Routing is too thin for the main paper. |
| 6 | Exp 2 statistics are wrong | The design is paired (the same case under both arms) and clustered (36 cases reused over 4 models, from 25 logs). Fisher's test assumes independence. Use McNemar or a mixed model. |
| 7 | Exclusions differ by arm | Parse rates differ between off and inject. Add an intention-to-treat analysis. |
| 8 | Confound: the warning may repeat the history | The "peer" failure comes from the same conversation and is probably still in the context the model sees. Measure this. |
| 9 | Don't just add repeats | The baseline repeats the error only 8.5% of the time, so there is almost nothing to prevent. Screen for cases that reproduce, then test on those. For Exp 1, add scenarios (reasoning bugs, adversarial false positives), not repeats. |

---

## 2. What we found, point by point

All numbers below come from `review_reanalysis/exp1_reanalysis.md` and `review_reanalysis/exp2_reanalysis.md`. You can regenerate both in about a minute with `run_review_reanalysis.bat`; it needs no Ollama.

### Point 1: the headline is circular. **He is right.**

In `pfti/bench/core_matrix.py`:

- Line 140: before each call, `matcher.check(sig, store, exclude_agent=agent)` looks for a **peer's** stored failure whose G-mid pattern matches.
- Line 147: under inject, a match means `will_warn = True`. Line 161 holds the call and returns the warning.
- Lines 195–199: a failure counts as a "repeated peer failure" when a **different agent** already failed with the **same G-mid signature**.

Any call that satisfies the line-195 condition also satisfies the line-140 matcher check. So under inject it is always warned and held the first time. It can only fail if the agent re-issues it after the warning, and line 204 counts that as `ignored`. The result is that under inject, repeated-peer ≤ ignored warnings, by construction. The data show exactly that: llama had 3 and 3, and every other model had 0 and 0. The new test `test_repeated_peer_metric_is_bounded_by_ignored_warnings_under_inject` (in `pfti/tests/test_review_fixes.py`) pins this down.

### Point 2: lead with the real outcome. **Agreed, and it goes one step further.**

| pooled over 4 models × 15 trials | off | shadow | inject |
|---|--:|--:|--:|
| task success | 49 | 50 | 47 |
| executed tool failures | 198 | 204 | 150 (−24%) |
| total tokens | 340,732 | 348,446 | 364,240 (+7%) |

**Something he did not mention:** part of the −24% is also mechanical. A held call is never executed, so it can never be counted as a failure. If you add the 65 held calls back in, the agents *attempted* 207–215 doomed calls under inject, versus 198 under off. (It is a range because this run didn't save traces, so we can't tell exactly how many held calls were false positives.) PFTI doesn't make agents propose fewer bad calls. It stops them from executing. That still has value when a failed call costs something (quota burned, side effects), but the paper should say so. A fair framing:

> PFTI intercepts ~65 doomed calls and reduces executed tool failures by 24%, but agents do not propose fewer doomed calls, and end-task success does not improve overall (49/60 vs 47/60) and is model-dependent.

### Point 3: noise floor. **He is right; here it is for every metric.**

| model | success (shadow−off) | failures | repeated peer | tokens |
|---|--:|--:|--:|--:|
| qwen2.5:3b | +0/15 | +1 | +0 | +0.0% |
| qwen2.5:7b | +0/15 | +0 | +1 | +3.8% |
| qwen2.5:14b | +1/15 | **+5** | **+9 (+75%)** | +4.0% |
| llama3.1:8b | +0/15 | +0 | +0 | +0.4% |

This happens because Ollama on a GPU isn't bit-for-bit reproducible even at temperature 0. For 14b the noise (5 failures) is almost half the inject effect (12 failures).

### Point 4: the llama regression. **His guess is probably wrong. The harm seems to come from correct warnings.**

The old run saved only per-scenario success counts, not traces, so we can't read what llama actually did. What we *can* do is check, on the testbed itself, where a false-positive warning is even possible. `pfti/eval/reanalyze_exp1.py`, section 5, tries a grid of about 100 plausible calls in every scenario:

- perm_denied, notfound_dir, rate_limit, missing_table: **0** possible false positives. The G-mid signature contains exactly the condition that causes the failure (table name, folder prefix + permission, endpoint), so a matching call always fails.
- quota: 6 possible false positives. Disk size is a *fine*-level dimension, so a small write to `/tmp` matches a peer's too-large write.

llama's regressions were in **missing_table (3/3 → 0/3)** and **perm_denied (3/3 → 2/3)**, and a false positive can't happen in either. So in those scenarios llama was hurt by **correct** warnings. One plausible mechanism: the held call returns "[This call was NOT executed…]", and llama gives up or treats that as the answer instead of trying the `users` table. The new trace-saving run confirms or refutes this (see §4). llama's 6 false positives must have come from `quota`, where its success stayed at 3/3, so those did no harm.

### Point 5: minor points

- Tokens under inject vs off: +4.0% (3b), +7.8% (7b), +2.0% (14b), **+15.7% (llama)**. He is right.
- Construct validity: agreed. That is why V2 adds scenarios PFTI is *not* designed for (§3).
- Routing: the new Exp 1 run skips it (`--no-routing`). Put it in an appendix or drop it.
- Correction to his note: qwen2.5:7b also had 2 false positives under inject (precision 0.90), so llama is not the only model with them.

### Point 6: Exp 2 statistics. **He is right. Redone properly, the conclusion is unchanged: no detectable effect.**

Now in `pfti/eval/reanalyze_exp2.py`:

- **Paired (McNemar):** out of 107 case-pairs where both arms produced runnable code, the warning *helped* on the same-error outcome in 4 and *hurt* in 3 (p = 1.00). For any failure it was 11 vs 9 (p = 0.82).
- **Clustered:** GEE clustered by log gives OR 0.99 [0.56, 1.77]. A mixed model with random case and log effects gives OR 0.79 [0.36, 1.76]. A cluster bootstrap gives +0.1 pts [−4.5, +4.7]. Every interval spans "no effect".

### Point 7: intention-to-treat. **Done; same answer.**

Counting "no runnable code" as a failure: fail rate is 79% → 74%, helped/hurt 16/9, p = 0.23. Same-error rate is 6.9% → 6.9%. Exclusions were close between arms (off 27 / inject 28 of 144).

### Point 8: the confound. **He is right, and it is worse than he suspected.**

| check (over the 36 cases) | result |
|---|--:|
| earlier failure still inside the context the model sees | **34/36 (94%)** |
| earlier failing code proposed by the *same* agent | 13/36 (36%) |
| earlier failure has the **same error class** as the one we test | **5/36 (14%)** |
| warning is new peer info (out of context AND different agent) | 2/36 |

**My error:** `find_transfer_points` in `pfti/eval/whoandwhen_llm.py` (line 80) uses `Matcher(level="mid")`. Who&When's dimensions (`error_class`, `missing_module`, …) are not listed in `DIM_LEVEL` in `pfti/core/signature.py` (line 22), so at "mid" a Who&When signature collapses to just `(tool=code_exec)`. In practice, "matching earlier failure" meant *any* earlier code failure. In 31 of the 36 cases the warning was about a **different error** from the one we graded against. The test couldn't work as designed. It is documented at the top of that file.

The same collapse affects the older static analysis in `pfti/eval/whoandwhen.py` (not written in this session). Its headline numbers need rewording:

| claim | as reported | with a stricter definition |
|---|--:|--:|
| "transfer-opportunity rate" | 26% of logs | 14% if the error class must match; **5%** if a *different* agent must have proposed the earlier failing code |
| "signature coverage" | 65% | That figure means "the log has any failure step". Only ~31% of logs have an extractable failure at the annotated decisive step (approximate; depends on how `mistake_step` is indexed). |

**One more bug found:** 1 of the 144 rows (qwen2.5:7b, log 109) crashed with a Windows encoding error when writing the model's code to a temp file, because the code contained the character "≈". It is fixed in `execute_code` (UTF-8 for the file and the child process) and covered by a test.

### Point 9: what to run next. **Built as he described.**

The baseline problem is visible in our own data: in 29 of the 36 cases, **none** of the 4 models reproduced the historical error even once.

---

## 3. What changed in the code

| file | change | why |
|---|---|---|
| `pfti/bench/core_matrix.py` | New counters: `held_tp`/`held_fp` (was the held call really doomed?), `doomed_attempts`, `peer_doomed_attempts`/`peer_doomed_executed` (a different agent already failed with the same tool + error class, **per the oracle, not the matcher**). Optional transcripts. `world_events` (the world can change between agents). Malformed tool arguments no longer crash a whole cell. | Points 1, 2, 4 |
| `pfti/bench/scenarios_llm.py` | `SCENARIOS_V2`: 3 adversarial false-positive scenarios (`prefix_collision`, `quota_small_ok`, `stale_ratelimit`), 1 control (`stale_perm_control`: the world changes in a way the signature *can* see, so PFTI should stay quiet), 2 reasoning-bug scenarios (`retired_endpoint`, `guess_dir`). The original 5 are unchanged. | Points 4, 5, 9 |
| `pfti/bench/run_all.py` | `--scenarios v1/v2/all`, `--save-traces`, `--no-routing`; keeps per-run results for every metric. | Points 3, 4 |
| `pfti/bench/trace_report.py` (new) | For each warning: was it right, and did the agent reroute, repeat or stop? Every regression (off succeeded, inject failed) is attributed to a false-positive warning, correct warnings only, or no warning (noise). | Point 4 |
| `pfti/eval/whoandwhen_llm.py` | UTF-8 fix; docstring corrected and known limitations recorded. Case selection unchanged so the first run stays reproducible. | Point 8, bug |
| `pfti/eval/waw_repro.py` (new) | The redesigned Exp 2 (see below). Resumable. | Point 9 |
| `pfti/eval/waw_repro_analyze.py` (new) | Paired case-level test, GEE clustered by log, mixed model (random case + log), cluster bootstrap, ITT, moderators. | Points 6, 7 |
| `pfti/eval/reanalyze_exp1.py`, `reanalyze_exp2.py` (new) | Re-analysis of the **existing** results; output is in `review_reanalysis/`. | Points 1–8 |
| `pfti/tests/test_review_fixes.py` (new) | 13 offline tests for all of the above. | |

### How the redesigned Exp 2 works (`pfti/eval/waw_repro.py`)

1. **Case pool** (`find_repro_cases`, line 68): every moment where an agent wrote Python that then failed when the original run executed it. That gives **90 cases from 65 logs**, up from 36 from 25, because it no longer requires an earlier "matching" failure.
2. **Phase 1, screen**: 6 samples per case under `off` at temperature 0.7. Keep a case (per model) if it repeats the historical error class in ≥ 2 of 6 samples. Repeats are useful here because they estimate each case's reproduction rate.
3. **Phase 2, test**: on kept cases only, draw **fresh** samples, 6 under `off` and 6 under `inject`, alternating. They are fresh so the off rate isn't inflated by the screening itself (the winner's curse).
4. **The warning** (`peer_warning`, line 114): the standard PFTI template, filled from the historical failure *at this very step* and attributed to `ParallelPeer`. It names the real error class and message. That failure happens after the replayed context, so it is never already in context, and it comes from a different agent. This tests peer transfer, not memory.
5. It answers his question: *given a model likely to repeat a peer's failure, does the warning stop it?*

---

## 4. What to run, in what order

Run these from the repo folder, one at a time (they share the GPU).

| order | file | needs Ollama? | time | read afterwards |
|---|---|---|---|---|
| 1 | `run_review_reanalysis.bat` | no | ~1 min | `review_reanalysis\exp1_reanalysis.md`, `exp2_reanalysis.md` (already included in this commit) |
| 2 | `run_exp1_v2.bat` | yes | ~45–60 min | `bench_results_v2\summary.md`, `bench_results_v2\trace_report.md` ← answers the llama question |
| 3 | `run_waw_repro.bat` | yes | ~2 h per model, 6–9 h for all 4, best overnight | `waw_repro_results\summary.md`, `waw_repro_cases.png`. If it stops, run it again and it resumes. |

What to look for when they finish:

- **trace_report.md:** in the *Regressions* table, the `cause` column. If llama's missing_table rows say "after correct warning(s) only", §2 point 4 is confirmed. The *agent's last words* column shows how it misread the warning.
- **Exp 1 v2 summary:** adversarial scenarios, i.e. how often models *repeat the call anyway* after a wrong warning (good) versus divert (harm). Control scenario: warnings should be 0.
- **waw_repro summary:** first *Screening* (how many cases per model reproduce at all). Then the *Phase 2* table and the *Clustered models* table.

---
