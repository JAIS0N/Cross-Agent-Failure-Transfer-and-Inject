# E1 Result Note — The Learning Layer (Adaptive Granularity via Online LGG)

**Flag:** `learning_layer=on/off`. Mechanism merged behind it; fixed-
granularity configs remain runnable as comparison arms.

## What was predicted
The learned layer, built by intersecting the predicate sets of failures in
the same `(tool, error_class)` bucket (least general generalization), should
converge to each failure class's true precondition **P\*** and therefore warn
correctly on concrete calls **never seen as instances** — something the old
instance-equality mechanism cannot do by construction.

## What was measured (E1, 5 failure classes, 3 seeds, k = 2…8)

| k | A1 recall | A2 recall | A3 recall | A1 prec | A2 prec | A3 prec | \|gsig Δ P\*\| | A3 pass |
|---|----------:|----------:|----------:|--------:|--------:|--------:|-------------:|--------:|
| 2 | 0.00 | 1.00 | 0.95 | 1.00 | 0.91 | 1.00 | 0.07 | 93% |
| 4 | 0.00 | 1.00 | 0.94 | 1.00 | 0.91 | 1.00 | 0.07 | 93% |
| 5 | 0.00 | 1.00 | 1.00 | 1.00 | 0.91 | 1.00 | 0.00 | 100% |
| 8 | 0.00 | 1.00 | 1.00 | 1.00 | 0.91 | 1.00 | 0.00 | 100% |

- **A1 (instance-equality):** unseen-recall ≈ 0 at all k — the intended
  contrast. Memorizing exact calls transfers nothing to new calls.
- **A2 (fixed G-mid):** high recall but precision plateaus at ~0.91 — it
  over-warns on near-miss hard negatives because the fixed mid projection
  drops the fine dimensions that are part of P\* for some classes
  (QuotaExceeded needs `size`, AuthError needs `key`).
- **A3 (learned LGG):** reaches recall 1.0 by k=5 at precision 1.0, and
  **dominates A2 on the precision/recall Pareto without any tuning** — it
  recovers exactly the dims each class needs, coarse or fine.
- **Convergence:** `|gsig Δ P*|` falls to 0 by k=5 — the empirical face of
  the (not-yet-written) convergence lemma.

## Pass criteria (decided before running) — all met
1. A3 unseen-recall ≥ 0.8 by k=4 on ≥4/5 classes, precision ≥ 0.9 vs
   near-misses ✓ (93% of class·seed cells pass at k=4; precision 1.00).
2. A3 ≥ A2 on the recall/precision Pareto ✓ (equal recall, higher precision).
3. Guard 2 fires and repairs a deliberately over-general class ✓ (lifecycle
   test `test_3_kmin_fpmax_split`: an over-general record with a coincidental
   correlated dim is deactivated and split on `perm` into precise sub-records).

## What surprised us
A3 hits precision 1.0 essentially immediately (k=2), while recall lags
slightly until k=5 — i.e. the risk of early generalization is *missed*
warnings (a coincidental shared irrelevant dim narrows `gsig`), not *false*
warnings. This says the near-miss precision probes were the easy half; the
harder guarantee is having enough irrelevant-dim variation for the
intersection to shed noise, which is precisely the §5 "not enough variation"
pitfall. Convergence at k≈5 with 5-value noise dims quantifies "enough".

## Reproduce
```
python -m pfti.demo_learning          # the three-wall narrative
python -m pfti.eval.e1_unseen         # the results table above
python -m pfti.eval.e1_plot           # e1_recall_precision.png, e1_convergence.png
python -m pytest pfti/tests/test_learning_layer.py -q   # 5 lifecycle tests
```
