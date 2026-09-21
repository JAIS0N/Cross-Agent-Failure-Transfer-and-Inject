# Who&When Real-Data Study — First Result

**What this is:** our predicate signature extractor run over the 184 real
annotated multi-agent failure logs in Who&When (126 algorithm-generated +
58 hand-crafted). No agents are run and no LLM is used — this is a pure,
reproducible pass over real failure trajectories.

**Two questions we asked of real data we did not create:**

1. **Signature coverage** — can our formalism represent real failures? For
   what fraction of logs can we extract a well-formed predicate signature
   from a genuine tool error in the trajectory?
2. **Transfer opportunity** — how often could PFTI have helped? In what
   fraction of trajectories does a later failure step match (by predicate
   subsumption) an earlier failure's signature — i.e. a moment where a peer
   warning could have fired?

## Results

| Metric | Value |
|--------|-------|
| Logs analyzed | 184 |
| Signature coverage | **65%** (119/184) |
| Transfer-opportunity rate | **26%** (47/184) |
| Total failure steps extracted | 262 |
| Avg failure steps per log | 1.4 |

**Most common real error classes** (these become our error_class buckets):
NotFound (73), BadRequestError (25), NameError (22), ValueError (21),
Timeout (21), ExecutionError (20), TypeError (14), FileNotFoundError (12),
RateLimited (12), AttributeError (10), KeyError (8).

## How to read this
- **65% coverage** means our predicate representation captures a real,
  machine-readable failure signature in roughly two-thirds of real failure
  logs, straight out of the box, with a lightweight error-class + operation
  + agent extractor. The uncovered ~35% are logs whose decisive error is
  *semantic* (the tool ran fine but the reasoning/answer was wrong) — these
  produce no tool error to key on, which matches our stated scope: PFTI v1
  deliberately targets explicit tool failures, not semantic ones.
- **26% transfer opportunity** means in about 1 in 4 real failed
  trajectories, a later failing step matched an earlier failure's pattern —
  a concrete, real-data moment where cross-agent warning could have fired.
  This is a lower bound: it only counts within-trajectory repeats using the
  strict subsumption matcher at mid granularity.

## Honest caveats
- Coverage of 65% is *representation* coverage, not a claim that PFTI would
  fix 65% of failures. It says: our formalism can express these failures.
- The 26% is within-trajectory only. Cross-trajectory transfer (one team's
  failure warning another team) would be measured separately and is
  expected to be higher once signatures are pooled across logs.
- Semantic failures (no tool error) are out of scope by design; they are the
  main reason coverage is not higher, and are named as future work.

## Reproduce
```
python -m pfti.eval.whoandwhen        # prints the table above
# data lives in whoandwhen_data/ (Algorithm-Generated/ + Hand-Crafted/)
```
Source: Who&When, Zhang et al., ICML 2025.
https://huggingface.co/datasets/Kevin355/Who_and_When
