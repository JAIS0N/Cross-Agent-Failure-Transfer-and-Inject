# Latest Review


**Status (Sept 23):** every run the review called for is finished. §2 answers his points using the old results. §5 has the new results, §6 the story they add up to for the paper, and §7 what goes back to him.

**Bottom line in one paragraph.** A peer warning reliably stops agents from *executing* a call that repeats an **environment-condition** failure (read-only folder, used-up quota, missing file). That holds in the testbed and, now, on real Who&When data (repeat rate 57% → 44%, p = 0.018). It does **not** help with logic bugs in the agent's own code. It does **not** make tasks succeed more often: agents avoid the known error and then usually fail another way. The only harm we found came from *how* the warning is delivered (llama3.1 breaks its tool-call format after one), not from what it says.

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

### Point 4: the llama regression. **His guess was wrong. Settled by the new traces (§5.1): the harm came from how the warning is delivered, not from bad warnings.**

*What follows is the reasoning done before the traces existed; it pointed the right way.*

The old run saved only per-scenario success counts, not traces, so we can't read what llama actually did. What we *can* do is check, on the testbed itself, where a false-positive warning is even possible. `pfti/eval/reanalyze_exp1.py`, section 5, tries a grid of about 100 plausible calls in every scenario:

- perm_denied, notfound_dir, rate_limit, missing_table: **0** possible false positives. The G-mid signature contains exactly the condition that causes the failure (table name, folder prefix + permission, endpoint), so a matching call always fails.
- quota: 6 possible false positives. Disk size is a *fine*-level dimension, so a small write to `/tmp` matches a peer's too-large write.

llama's regressions were in **missing_table (3/3 → 0/3)** and **perm_denied (3/3 → 2/3)**, and a false positive can't happen in either. So in those scenarios llama was hurt by **correct** warnings. One plausible mechanism: the held call returns "[This call was NOT executed…]", and llama gives up or treats that as the answer instead of trying the `users` table. The new trace-saving run confirms or refutes this (see §4). llama's 6 false positives must have come from `quota`, where its success stayed at 3/3, so those did no harm.

### Point 5: minor points

- Tokens under inject vs off: +4.0% (3b), +7.8% (7b), +2.0% (14b), **+15.7% (llama)**. He is right. In the 11-scenario run it is +8% to +15% per model, +12% overall.
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

The baseline problem is visible in our own data: in 29 of the 36 cases, **none** of the 4 models reproduced the historical error even once. The redesigned study has now been run; results in §5.3.

---

## 3. What changed in the code

| file | change | why |
|---|---|---|
| `pfti/bench/core_matrix.py` | New counters: `held_tp`/`held_fp` (was the held call really doomed?), `doomed_attempts`, `peer_doomed_attempts`/`peer_doomed_executed` (a different agent already failed with the same tool + error class, **per the oracle, not the matcher**). Optional transcripts. `world_events` (the world can change between agents). Malformed tool arguments no longer crash a whole cell. | Points 1, 2, 4 |
| `pfti/bench/scenarios_llm.py` | `SCENARIOS_V2`: 3 adversarial false-positive scenarios (`prefix_collision`, `quota_small_ok`, `stale_ratelimit`), 1 control (`stale_perm_control`: the world changes in a way the signature *can* see, so PFTI should stay quiet), 2 reasoning-bug scenarios (`retired_endpoint`, `guess_dir`). The original 5 are unchanged. **Fixed after the first v2 run:** the two reasoning scenarios could not be solved (`list_dir` lists files, not folders; `/v2/render` was unguessable), so each world now has a pointer file (`/README.txt`, `/api/ENDPOINTS.txt`). | Points 4, 5, 9 |
| `pfti/bench/run_all.py` | `--scenarios v1/v2/all/reasoning`, `--save-traces`, `--no-routing`, `--parse-text-calls`; keeps per-run results for every metric. | Points 3, 4 |
| `pfti/bench/core_matrix.py` (second change) | `parse_text_call`: counts tool calls a model writes as JSON text instead of a real call (`text_calls`), and with `--parse-text-calls` executes them, identically in every mode. | Point 4 |
| `pfti/bench/trace_report.py` (new) | For each warning: was it right, and did the agent reroute, repeat or stop? Every regression (off succeeded, inject failed) is attributed to a false-positive warning, correct warnings only, or no warning (noise). | Point 4 |
| `pfti/eval/whoandwhen_llm.py` | UTF-8 fix; docstring corrected and known limitations recorded. Case selection unchanged so the first run stays reproducible. | Point 8, bug |
| `pfti/eval/waw_repro.py` (new) | The redesigned Exp 2 (see below). Resumable. | Point 9 |
| `pfti/eval/waw_repro_analyze.py` (new) | Paired case-level test, GEE clustered by log, mixed model (random case + log), cluster bootstrap, ITT, moderators. The primary outcome and tests were fixed here **before** the real run. | Points 6, 7 |
| `pfti/eval/waw_repro_robustness.py` (new) | Stress tests of the Who&When result: how "no runnable code" is counted, per-model vs per-case units, leave-one-model-out, leave-one-error-class-out, what happened instead of the repeat. Output: `review_reanalysis/waw_repro_robustness.md`. | §5.3 |
| `pfti/eval/reanalyze_exp1.py`, `reanalyze_exp2.py` (new) | Re-analysis of the **existing** results; output is in `review_reanalysis/`. | Points 1–8 |
| `pfti/tests/test_review_fixes.py` (new) | 16 offline tests for all of the above. | |

### How the redesigned Exp 2 works (`pfti/eval/waw_repro.py`)

1. **Case pool** (`find_repro_cases`, line 68): every moment where an agent wrote Python that then failed when the original run executed it. That gives **90 cases from 65 logs**, up from 36 from 25, because it no longer requires an earlier "matching" failure.
2. **Phase 1, screen**: 6 samples per case under `off` at temperature 0.7. Keep a case (per model) if it repeats the historical error class in ≥ 2 of 6 samples. Repeats are useful here because they estimate each case's reproduction rate.
3. **Phase 2, test**: on kept cases only, draw **fresh** samples, 6 under `off` and 6 under `inject`, alternating. They are fresh so the off rate isn't inflated by the screening itself (the winner's curse).
4. **The warning** (`peer_warning`, line 114): the standard PFTI template, filled from the historical failure *at this very step* and attributed to `ParallelPeer`. It names the real error class and message. That failure happens after the replayed context, so it is never already in context, and it comes from a different agent. This tests peer transfer, not memory.
5. It answers his question: *given a model likely to repeat a peer's failure, does the warning stop it?*

---

## 4. What was run (all finished)

| date | launcher | what it did | output |
|---|---|---|---|
| Sep 22 | `run_review_reanalysis.bat` | re-analysis of the old results, no models | `review_reanalysis/exp1_reanalysis.md`, `exp2_reanalysis.md` |
| Sep 22 | `run_exp1_v2.bat` | 11 scenarios × off/shadow/inject × 4 models, traces saved (41 min) | `bench_results_v2/` (`summary.md`, `trace_report.md`, `traces/`) |
| Sep 22 | `run_exp1_v2_followup.bat` | (A) the 2 fixed reasoning scenarios, 4 models; (B) llama on all 11 with `--parse-text-calls` (14 min) | `bench_results_v2_reasoning/`, `bench_results_v2_llama_textcalls/` |
| Sep 23 | `run_waw_repro_7b.bat`, then `run_waw_repro.bat` | reproduction-filtered Who&When study, 7b first, then the other 3 (~8 h total) | `waw_repro_results/` (`summary.md`, `waw_repro_cases.png`, `results.jsonl`) |
| Sep 23 | `python -m pfti.eval.waw_repro_robustness` | stress tests of the Who&When result, no models | `review_reanalysis/waw_repro_robustness.md` |

All runs: one repeat at temperature 0 for Experiment 1 (the professor's point: at T=0 scenarios, not repeats, add information); temperature 0.7 with 6 samples per case per arm for Experiment 2.

---

## 5. What the new runs found

### 5.1 Experiment 1, 11 scenarios (`bench_results_v2/`)

**The original 5 scenarios reproduced last week's pattern exactly.** 7b is again fixed by warnings on `notfound_dir`; llama again fails `missing_table` under inject; 14b doesn't change.

**The mechanism, measured without circularity** (pooled over 4 models; "doomed" = the oracle says the call would fail, counted when proposed, in every mode):

| | off | shadow | inject |
|---|--:|--:|--:|
| doomed calls **proposed** | 165 | 165 | 162 |
| ...of which repeat a failure a *different* agent already had (oracle-defined) | 72 | 77 | 76 |
| ...and were actually **executed** | 72 | 77 | **37** |
| executed tool failures | 165 | 165 | **123 (−25%)** |
| tasks solved (9 working scenarios × 4 models = 36) | 32 | 32 | 31 |
| tokens | 266,679 | 266,994 | 298,576 (+12%) |

In plain words: agents *propose* just as many doomed calls with warnings on. PFTI stops about half of the peer-repeats from *running*. Task success does not change. The shadow column shows the noise level (77 vs 72).

**Wrong warnings (the 3 trap scenarios):** 9 false-positive warnings were delivered (48 warnings in total, 39 correct). The Qwen models always either re-sent the call or found another route, and never lost a task because of a wrong warning. llama lost one (`stale_ratelimit`). Caveat for the paper: each trap task *tells* the agent the situation changed ("the rate limit reset a minute ago"), which makes a wrong warning easier to see through than it would be in real use.

**Control scenario:** 0 warnings for all 4 models, as intended (a state-aware signature stays quiet when the world visibly changed).

**The llama question, answered from the traces.** After a warning, llama usually makes the right decision ("the accounts table doesn't exist, I'll query users instead") but then writes the tool call as text instead of making it, and invents a result. Tool calls written as text: 10 with warnings on vs 5 with them off; the Qwen models: 0. The follow-up run (`bench_results_v2_llama_textcalls/`) executes such calls, in every mode:

| llama3.1:8b, 11 scenarios, text calls accepted | off | shadow | inject |
|---|--:|--:|--:|
| tasks solved | 10/11 | 10/11 | 10/11 |
| peer-repeat calls proposed / executed | 6 / 6 | 6 / 6 | 7 / 1 |

The regressions disappear. So llama was hurt by *how* the warning reaches it (as a tool result, which pushes it out of native tool calling), not by *what* it says. The professor's guess (false positives) was wrong: llama's regressions were in scenarios where a false positive is impossible (§2, point 4). Side effect: accepting text calls also helps llama with no warnings (10/11 vs 8/11), so the old llama numbers understated it in every mode.

### 5.2 The fixed reasoning scenarios (`bench_results_v2_reasoning/`)

No signal for PFTI. qwen2.5:7b and 14b read the pointer file first and solved both, in every mode, without failing, so there was nothing to prevent. qwen2.5:3b failed both in every mode, and warnings didn't help (its tokens went from 24.6k to 65.7k). llama solved 1 of 2 in every mode. Honest takeaway: capable models checked before acting, so the "act before checking" failure mostly didn't occur.

### 5.3 Experiment 2, reproduction-filtered Who&When (`waw_repro_results/`)

**Screening.** 90 real code-failure moments, 6 samples each without a warning (T = 0.7):

| model | baseline repeat rate | cases kept (≥ 2/6) |
|---|--:|--:|
| qwen2.5:3b | 9% | 11 |
| qwen2.5:7b | 11% | 13 |
| qwen2.5:14b | 8% | 10 |
| llama3.1:8b | 6% | 5 |

**Test** (fresh samples on kept cases; 39 model×case units, 21 distinct cases, 18 logs):

| model | repeats the historical error: off → inject |
|---|--:|
| qwen2.5:3b | 50% → 32% |
| qwen2.5:7b | 58% → 42% |
| qwen2.5:14b | 68% → 57% |
| llama3.1:8b | 50% → 50% (only 5 cases) |
| **all** | **57% → 44%** |

Pre-specified primary test (GEE clustered by log): **OR 0.58 [0.37, 0.91], p = 0.018**. 25 units improved, 10 got worse, 4 tied (sign test p = 0.017). Cluster bootstrap: −13 points [−23, −2]. The fresh-sample design mattered: in screening the kept cases repeated 69% of the time, in fresh samples 58% (7b), exactly the winner's-curse inflation it was built to avoid.

**Stress tests** (`review_reanalysis/waw_repro_robustness.md`):

| check | result |
|---|---|
| count "no runnable code" as a repeat (strict) | 63% → 53%, p = 0.049 (holds, barely) |
| graded samples only | 61% → 48%, p = 0.033 (holds) |
| drop one model at a time | p = 0.002 to 0.074 (holds, weakens without 3b) |
| count each *distinct case* once | 12 better / 6 worse, p = 0.08 to 0.24 (**not significant**) |
| drop FileNotFoundError cases | 51% → 48%, p = 0.61 (**effect disappears**) |
| *any* failure, not just the same error | 82% → 79%, p = 0.34 (**no change**) |

What the three bold rows mean:
- **The effect is one error type.** Missing-file errors drop from 67% to 39%. Every other error type together: 51% → 48%. Logic bugs (AttributeError, KeyError) get slightly *more* repeats with the warning.
- **The warning changes which error happens, not whether the code works.** On the missing-file cases, 31% of warned answers failed with a *different* error (vs 11% without) and 26% ran OK (vs 19%).
- **The sample is still small.** Counted per distinct case, it is not significant. And all kept cases come from the automatically generated half of Who&When; none from the hand-crafted half.

---

## 6. The story for the paper

1. **Mechanism (strong, reproduced across runs and datasets).** PFTI stops agents from *executing* calls that repeat a peer's **environment-condition** failure: about half of such calls in the testbed, and a 13-point drop in same-error repeats on real Who&When failures (p = 0.018, carried by missing-file errors).
2. **No gain in task success (consistent everywhere).** Testbed: 32 vs 31 of 36. Who&When: any-failure 82% vs 79%. Agents avoid the known error and then fail differently.
3. **Scope (construct validity, now with evidence).** It works for failures caused by the environment, which is what it was designed for; it does not help, and may slightly hurt, with logic bugs in the agent's own code.
4. **Harm comes from delivery, not content.** Wrong warnings rarely hurt when the task context contradicts them; the one real harm was llama's tool-call format breaking after a warning, and it vanishes when text calls are accepted.
5. **Cost.** About +12% tokens; much more for a weak model that can't use the warning.
6. **Honest corrections to earlier claims.** The "56 → 3 (−95%)" headline is circular; the old Who&When static figures (26% transfer opportunity, 65% coverage) overstate, and the first real-data replay's null result was caused by a case-selection bug (§2, point 8).

---
