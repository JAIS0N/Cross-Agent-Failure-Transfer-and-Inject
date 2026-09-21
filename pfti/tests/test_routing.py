"""Tests for the routing extension."""
from pfti.routing.models import MODELS, make_batch, best_model_oracle, SUBTASK_TYPES
from pfti.routing.router import PerfStore, PerfRecord, Router, perf_signature, task_key
from pfti.routing.experiment import run_all, run_condition


def test_oracle_is_per_task_nontrivial():
    best = {t: best_model_oracle(t) for t in SUBTASK_TYPES}
    assert len(set(best.values())) > 1  # not the same model for everything


def test_router_cold_start_uses_default():
    r = Router(PerfStore(), MODELS, default="small")
    model, why = r.choose("extract", "plain", exclude_agent="A")
    assert model == "small" and why == "cold_start"


def test_router_excludes_own_records():
    store = PerfStore()
    store.add(PerfRecord("A", "extract", "plain", "large", True, 1200, 6.0))
    r = Router(store, MODELS)
    # A cannot learn from its own record -> cold start for A
    m_a, why_a = r.choose("extract", "plain", exclude_agent="A")
    assert why_a == "cold_start"
    # B can see A's record
    m_b, why_b = r.choose("extract", "plain", exclude_agent="B")
    assert why_b != "cold_start"


def test_router_learns_best_after_exploration():
    store = PerfStore()
    # peers tried both models on 'extract': large succeeds, small fails
    for _ in range(3):
        store.add(PerfRecord("A", "extract", "plain", "large", True, 1200, 6.0))
        store.add(PerfRecord("A", "extract", "plain", "small", False, 200, 1.0))
    r = Router(store, MODELS)
    model, why = r.choose("extract", "plain", exclude_agent="B")
    assert model == "large" and why.startswith("learned")


def test_routed_beats_small_and_cheaper_than_large():
    agg = {p: {"s": [], "c": []} for p in ("small", "large", "routed")}
    for seed in range(5):
        r = run_all(40, seed=seed)
        for p in agg:
            agg[p]["s"].append(r[p]["success_rate"])
            agg[p]["c"].append(r[p]["total_cost"])
    import statistics as st
    sm, lg, ro = (st.mean(agg[p]["s"]) for p in ("small", "large", "routed"))
    cs, cl, cr = (st.mean(agg[p]["c"]) for p in ("small", "large", "routed"))
    assert sm < ro <= lg + 0.02        # routed accuracy between small and large
    assert cs < cr < cl                # routed cost between small and large


def test_granularity_levels_change_signature_size():
    coarse = perf_signature("extract", "table", "small", level="coarse")
    mid = perf_signature("extract", "table", "small", level="mid")
    fine = perf_signature("extract", "table", "small", level="fine")
    assert coarse.preds < mid.preds < fine.preds  # chain, like base PFTI
