"""Bench harness invariants, validated with the deterministic fake client
(no Ollama needed) so CI proves the before/after pipeline is wired right."""
from pfti.bench.core_matrix import run_scenario, shadow_prf
from pfti.bench.scenarios_llm import SCENARIOS
from pfti.bench.providers import FakeLLM, strip_script
from pfti.bench import routing_matrix as RM


def _run(mode):
    client = FakeLLM()
    results = [run_scenario(sc, mode, client, "fake", use_fake=True)
               for sc in SCENARIOS]
    succ = sum(r["success"] for r in results)
    peer = sum(r["repeated_peer_failures"] for r in results)
    warn = sum(r["warnings_delivered"] for r in results)
    return succ, peer, warn, results


def test_inject_beats_off_on_success():
    off_succ, off_peer, off_warn, _ = _run("off")
    inj_succ, inj_peer, inj_warn, _ = _run("inject")
    # PFTI must strictly help: more successes, fewer repeated peer failures
    assert inj_succ > off_succ
    assert inj_peer < off_peer
    # off never injects; inject does
    assert off_warn == 0 and inj_warn > 0


def test_off_and_shadow_behave_identically():
    # shadow is pure measurement: same behavior as off, no warnings delivered
    off_succ, off_peer, off_warn, _ = _run("off")
    sh_succ, sh_peer, sh_warn, _ = _run("shadow")
    assert (off_succ, off_peer, off_warn) == (sh_succ, sh_peer, sh_warn)


def test_shadow_reports_matcher_precision_recall():
    _, _, _, results = _run("shadow")
    events = [e for r in results for e in r["events"]]
    prf = shadow_prf(events)
    # matcher should be precise on the controlled testbed at G-mid
    assert prf["precision"] >= 0.99
    assert 0.0 <= prf["recall"] <= 1.0


def test_real_model_task_text_hides_the_script():
    # a real model must never see the embedded fake script
    for sc in SCENARIOS:
        for a in sc["agents"]:
            assert "PFTI_SCRIPT" not in strip_script(a["task"])


def test_routing_fake_good_corner():
    models = RM.build_fake_models()
    rows = {r["policy"]: r for r in RM.run_routing(models)}
    routed = rows["routed"]
    cheapest = min(models, key=lambda m: models[m].cost)
    # routed should beat the cheapest model's accuracy
    assert routed["success_rate"] >= rows[cheapest]["success_rate"]
