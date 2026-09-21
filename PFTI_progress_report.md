# PFTI Project — Progress Report

**Author:** Anisha Fernandes
**Project:** Predicate-Guided Cross-Agent Failure Transfer & Injection (PFTI)
**Status as of this report:** Core mechanism, granularity ablation, and the
Learning Layer (adaptive granularity) built and tested; 27 unit tests
passing. Real-dataset validation is the next step.

---

## 1. The problem we are solving

In a multi-agent system, several AI agents work together on a task — one
plans, one writes code, one handles files. Today they do not learn from
each other's mistakes. If one agent's tool call fails (say, writing to a
read-only folder), another agent minutes later will happily attempt the
same doomed action. Existing research only diagnoses these failures
*after* the whole task is over (post-hoc failure attribution).

PFTI turns that post-mortem into live prevention: when one agent fails, we
convert the failure into a compact logical description, store it, and warn
any *other* agent that is about to make a matching call — before it acts.

## 2. The professor's objective (the "why")

The direction set by the professor (and Hailu) has three goals:

1. **Cross-agent, act-time prevention.** Move from analysing failures after
   the fact to preventing them during the task, at the exact moment of the
   risky call. This is the "mid-stage intelligence exchange" idea from the
   multi-agent collaboration survey — agents exchanging distilled knowledge
   while they work, not before or after.

2. **Symbolic, not neural.** The mechanism must be logical and auditable —
   every warning must be explainable ("it warned because pattern X matched")
   rather than an opaque model output. Explicitly: no LLM inside the
   learning loop.

3. **Formal weight for the paper.** Frame the abstraction level of a failure
   description as a formal knob (a partial order on predicates), prove
   simple properties about it, and validate them experimentally. The
   professor called this "what gives the paper AAAI-level weight."

## 3. What we built

### 3.1 The core mechanism
- **Signature** — a failure is represented as a set of (dimension, value)
  predicates, e.g. {(tool=write_file), (path_prefix=/data), (perm=ro)}.
- **Three granularity levels** (coarse / mid / fine) as projections of one
  fine signature — the formal knob for how abstract a description is.
- **Failure store** — an append-only log of failures, plus matching that
  never warns an agent about its own record (cross-agent only).
- **Injection** — an advisory warning inserted into a peer's context before
  it acts (it may still proceed; this lets us measure behaviour change).
- **Shadow mode** — the matcher runs and logs would-be warnings without
  injecting, checked against a ground-truth oracle. This is how we measure
  precision/recall with no behavioural confound.

### 3.2 The controlled testbed (our current "dataset")
A 3-agent software-ops team (Planner, Coder, FileOps) with six mock tools
(write_file, read_file, list_dir, run_code, call_api, query_db). Each tool
fails deterministically based on a hidden world state we control, so for any
call we can compute the true answer — would it fail? — without an LLM. This
ground-truth **oracle** is what makes precise measurement possible.

### 3.3 The Learning Layer (the most recent, specced deliverable)
Originally, granularity was a fixed setting. The Learning Layer makes the
system **learn the right abstraction per failure class, online**:
- Failures are bucketed by (tool, error_class).
- Within a bucket, their signatures are **intersected** (least general
  generalization, LGG) to derive the shared pattern — keeping the common
  cause, dropping irrelevant differences.
- **Guard 1 (anti-collapse):** refuse a pattern that generalises too far.
- **Guard 2 (self-correction):** if a learned pattern starts producing false
  warnings, deactivate it, find the dimension that separates good from bad
  warnings, split the bucket, and re-generalise. This is candidate
  elimination running live at runtime.

## 4. Core objective of the experiment (E1)

**The single question E1 answers:** can the system correctly warn about
concrete calls it has *never seen before*?

The old exact-match method could only warn about failures it had already
seen verbatim — useless in practice, since no two calls are identical. The
Learning Layer should warn about *new* calls that match the learned rule.
E1 proves this with numbers, comparing three arms:
- **A1** — old instance-equality (exact match only)
- **A2** — fixed medium granularity (the previous best setting)
- **A3** — the new Learning Layer (LGG + guards)

E1 runs on 5 failure classes whose true precondition we control. For each,
the system sees k example failures, then is tested on 100 brand-new calls
(50 that should fail, 50 near-miss traps that look similar but are safe).
We measure **recall** (did it catch the real dangers?) and **precision**
(did it avoid false alarms?). The near-miss traps are what make high
precision meaningful — a lazy "warn about everything" system fails them.

## 5. Findings so far

### 5.1 Base mechanism (granularity ablation, shadow mode vs oracle)
| Granularity | Precision | Recall |
|-------------|-----------|--------|
| coarse | 0.75 | 1.00 |
| **mid** | **1.00** | **1.00** |
| fine | 1.00 | 0.17 |

This confirms the predicted trade-off: too coarse over-warns (precision
drops), too fine matches almost nothing (recall collapses), the middle is
best. On the 5-scenario suite, turning the mechanism on took task success
from 20% to 100% and repeated peer failures from 6 to 0.

### 5.2 Learning Layer (E1), averaged over 5 classes and 3 seeds
| k (examples) | A1 recall | A2 recall | A3 recall | A1 prec | A2 prec | A3 prec | convergence \|gsig Δ P*\| |
|---|---|---|---|---|---|---|---|
| 2 | 0.00 | 1.00 | 0.95 | 1.00 | 0.91 | 1.00 | 0.07 |
| 5 | 0.00 | 1.00 | 1.00 | 1.00 | 0.91 | 1.00 | 0.00 |
| 8 | 0.00 | 1.00 | 1.00 | 1.00 | 0.91 | 1.00 | 0.00 |

Reading this:
- **A1 (old method): recall 0** on unseen calls — it genuinely cannot
  generalise. This is the headline contrast.
- **A2 (fixed mid): precision stuck at ~0.91** — it over-warns on near-miss
  traps because it blindly drops fine detail some failures actually need.
- **A3 (Learning Layer): recall 1.0 at precision 1.0**, and its learned
  pattern converges exactly to the true failure condition by ~4-5 examples.
  It matches or beats the hand-tuned setting **without any tuning**.
- All five acceptance criteria the professor set were met, including a
  deliberately over-general case where Guard 2 fires and repairs itself.

### 5.3 An honest nuance
A3 reaches perfect precision almost immediately, but recall lags slightly
until ~5 examples. So the real risk of early generalisation is *missed*
warnings (when too few examples share enough variety), not false alarms.
This is exactly the "not enough variation" pitfall the spec anticipated,
and the convergence curve quantifies how many examples are "enough."

## 6. What the dataset is — and what it will be

**Now:** a self-built controlled testbed with a ground-truth oracle. This is
deliberate — precise precision/recall measurement requires knowing the true
answer, which only a controlled world gives you. It is not an external
dataset, and we should say so plainly.

**Next:** validate on real multi-agent failure logs so results are not
confined to data we created. The leading candidate is **Who&When** (184 real
failure logs from 127 multi-agent systems, annotated with which agent failed
and when). We would run our signature extractor over these logs and report
how much of real-world failure our representation covers, and how often a
later step matches an earlier failure (a real warning opportunity).

## 7. A note on "learning" (a recurring point of confusion)
The system does **not** train on a dataset in the machine-learning sense.
The LLM agents are frozen. "Learning" here means two things: (a) agents
learn from each other's failures *during* a task via the shared store, and
(b) the Learning Layer induces general failure rules from instances, live
and symbolically. No weights are updated; nothing needs GPUs. Datasets are
for *evaluation*, not training.

## 8. Status and next steps
- Built and tested: core mechanism, granularity ablation, Learning Layer
  (LGG + both guards), E1 experiment with plots, 27 passing unit tests.
- Also prototyped (side exploration, not core): predicate-guided model
  routing, where agents share performance records to route tasks to the
  right model — reached near-large accuracy at ~half the cost.
- Next: (1) Who&When real-data coverage study; (2) write up the lemmas and
  the convergence property formally; (3) baselines for the paper.
