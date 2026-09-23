"""Tests for the changes made after Prof. Hailu's review (offline, no Ollama).

Exp 1: non-circular metrics, held-call accounting, V2 scenarios (adversarial
false positives + reasoning failures), world events, trace saving.
Exp 2: reproduction-filtered Who&When design (skipped without the data).
"""
import json
import os

import pytest

from pfti.bench.core_matrix import run_scenario
from pfti.bench.providers import FakeLLM, strip_script
from pfti.bench.scenarios_llm import SCENARIOS, SCENARIOS_V2, SCENARIO_SETS


def _runs(scenarios, mode):
    return [run_scenario(sc, mode, FakeLLM(), "fake", use_fake=True)
            for sc in scenarios]


# ---------------------------------------------------------------- Exp 1 --

def test_v1_scenarios_unchanged():
    # the original five are untouched, so old results stay comparable
    assert [s["name"] for s in SCENARIOS] == [
        "perm_denied", "notfound_dir", "rate_limit", "quota", "missing_table"]
    assert SCENARIO_SETS["all"] == SCENARIOS + SCENARIOS_V2


def test_held_calls_split_into_true_and_false_positives():
    for r in _runs(SCENARIO_SETS["all"], "inject"):
        assert r["held_tp"] + r["held_fp"] == r["warnings_delivered"]


def test_repeated_peer_metric_is_bounded_by_ignored_warnings_under_inject():
    # documents the circularity found in review: under inject a matcher-
    # defined repeat can only happen when a warning was ignored
    for r in _runs(SCENARIO_SETS["all"], "inject"):
        assert r["repeated_peer_failures"] <= r["ignored_warnings"]


def test_oracle_defined_peer_metric_counts_attempts_in_every_mode():
    off = _runs(SCENARIOS, "off")
    inj = _runs(SCENARIOS, "inject")
    # counted at proposal time, so it is defined for off too
    assert sum(r["peer_doomed_attempts"] for r in off) > 0
    # executed <= attempted
    for r in off + inj:
        assert r["peer_doomed_executed"] <= r["peer_doomed_attempts"]
        assert r["peer_doomed_attempts"] <= r["doomed_attempts"]


def test_adversarial_scenarios_produce_false_positive_warnings():
    adv = [s for s in SCENARIOS_V2 if s["kind"] == "adversarial_fp"]
    assert len(adv) == 3
    for sc, r in zip(adv, _runs(adv, "inject")):
        assert r["held_fp"] >= 1, sc["name"]
    # the fake always heeds warnings, so a wrong warning must cost it
    off = sum(r["success"] for r in _runs(adv, "off"))
    inj = sum(r["success"] for r in _runs(adv, "inject"))
    assert inj < off


def test_state_aware_signature_does_not_warn_after_visible_change():
    ctrl = [s for s in SCENARIOS_V2 if s["name"] == "stale_perm_control"]
    r = _runs(ctrl, "inject")[0]
    assert r["warnings_delivered"] == 0 and r["success"]


def test_reasoning_scenarios_get_correct_warnings():
    rs = [s for s in SCENARIOS_V2 if s["kind"] == "reasoning"]
    assert len(rs) == 2
    for r in _runs(rs, "inject"):
        assert r["held_tp"] >= 1 and r["held_fp"] == 0


def test_v2_task_text_hides_script():
    for sc in SCENARIOS_V2:
        for a in sc["agents"]:
            assert "PFTI_SCRIPT" not in strip_script(a["task"])


def test_run_all_saves_traces_and_richer_aggregates(tmp_path):
    from pfti.bench import run_all, trace_report
    out = tmp_path / "res"
    run_all.main(["--fake", "--models", "m1", "--scenarios", "v2",
                  "--save-traces", "--no-routing", "--out", str(out)])
    core = json.loads((out / "core_matrix.json").read_text())
    cell = next(c for c in core if c["mode"] == "inject")
    for k in ("held_tp", "held_fp", "peer_doomed_attempts", "runs"):
        assert k in cell
    traces = list((out / "traces").glob("*.json"))
    assert len(traces) == 3 * len(SCENARIOS_V2)
    t = json.loads(traces[0].read_text())
    assert "events" in t and "transcripts" in t
    md = trace_report.build(out / "traces")
    assert "FALSE-positive" in md


# ---------------------------------------------------------------- Exp 2 --

HERE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
HAVE_DATA = os.path.isdir(os.environ.get("WAW_DIR", os.path.join(HERE, "whoandwhen_data")))
needs_data = pytest.mark.skipif(not HAVE_DATA, reason="whoandwhen_data/ not present")


@needs_data
def test_repro_pool_is_larger_than_first_study():
    from pfti.eval.waw_repro import find_repro_cases
    cases = find_repro_cases()
    assert len(cases) >= 80
    assert len({c["log"] for c in cases}) >= 50
    assert all(c["code_index"] < c["later_index"] for c in cases)


@needs_data
def test_peer_warning_names_the_real_error_and_a_peer():
    from pfti.eval.waw_repro import find_repro_cases, peer_warning, build_messages, PEER_NAME
    c = find_repro_cases()[0]
    w = peer_warning(c)
    assert "PEER FAILURE WARNING" in w and PEER_NAME in w
    assert f"error_class={c['later_ec'].lower()}" in w
    off = build_messages(c, "off")
    inj = build_messages(c, "inject")
    # identical except for the one inserted warning, placed before the
    # final instruction
    assert len(inj) == len(off) + 1
    assert inj[:-2] == off[:-1] and inj[-1] == off[-1]
    assert inj[-2]["content"] == w
    # the warned-about failure happens AFTER the replayed context
    assert c["later_index"] > c["code_index"]


@needs_data
def test_fake_study_end_to_end_and_resume(tmp_path):
    from pfti.eval import waw_repro, waw_repro_analyze
    out = str(tmp_path / "r")
    r1 = waw_repro.run(["fake"], out, k_screen=3, k_test=2, min_repro=1,
                       max_cases=12, use_fake=True, verbose=False)
    n1 = sum(1 for _ in open(os.path.join(out, "results.jsonl")))
    # resume: nothing is re-run
    r2 = waw_repro.run(["fake"], out, k_screen=3, k_test=2, min_repro=1,
                       max_cases=12, use_fake=True, verbose=False)
    n2 = sum(1 for _ in open(os.path.join(out, "results.jsonl")))
    assert n1 == n2 > 0
    md, R = waw_repro_analyze.analyze(out)
    assert "Phase 2" in md
    assert R["test"]["ALL"]["inj"] < R["test"]["ALL"]["off"]


@needs_data
def test_execute_code_handles_non_ascii():
    from pfti.eval.whoandwhen_llm import execute_code
    out = execute_code("python", "print('≈ approx')")
    assert "execution succeeded" in out
