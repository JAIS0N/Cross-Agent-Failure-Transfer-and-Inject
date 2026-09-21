"""Unit tests: Day-2 acceptance tests + control-loop invariants."""
import glob
import os
import random

import pytest

from pfti.core import FailureStore, Matcher, Injector, PFTIHook, TraceLogger
from pfti.core.signature import Signature, extract
from pfti.envs.tools import World, ToolError, TOOLS, would_fail
from pfti.eval.runner import load_scenario, run_episode
from pfti.eval.shadow import shadow_stats

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _world():
    return World(dirs=["/", "/data", "/tmp"],
                 perm={"/data": "ro", "/tmp": "rw"})


# ---------- Day-2 acceptance tests (plan §2.2) ----------

def test_same_gmid_when_only_content_differs():
    w = _world()
    s1 = extract("write_file", {"path": "/data/a.txt", "content": "hello you"}, w)
    s2 = extract("write_file", {"path": "/data/a.txt", "content": "different"}, w)
    assert s1.project("mid") == s2.project("mid")


def test_different_gmid_across_dirs_with_different_perms():
    w = _world()
    s1 = extract("write_file", {"path": "/data/a.txt", "content": "x"}, w)
    s2 = extract("write_file", {"path": "/tmp/a.txt", "content": "x"}, w)
    assert s1.project("mid") != s2.project("mid")


def test_projection_commutes_random_calls():
    w = _world()
    random.seed(0)
    for _ in range(100):
        tool = random.choice(["write_file", "read_file", "call_api", "query_db"])
        args = {"write_file": {"path": f"/data/f{random.randint(0,9)}.txt",
                               "content": "z" * random.randint(1, 20000)},
                "read_file": {"path": f"/tmp/g{random.randint(0,9)}"},
                "call_api": {"endpoint": f"/e{random.randint(0,3)}"},
                "query_db": {"table": "t", "query": "select 1"}}[tool]
        fine = extract(tool, args, w)
        for lv in ("coarse", "mid"):
            assert fine.project(lv) == fine.project("fine").project(lv)
        assert fine.project("coarse").preds <= fine.project("mid").preds \
            <= fine.project("fine").preds


# ---------- matcher invariants ----------

def _fail_then_check(matcher, agent2="Coder", path2="/data/other.txt"):
    w = _world()
    store = FailureStore()
    sig = extract("write_file", {"path": "/data/rpt.csv", "content": "x"}, w)
    store.add(sig, "write_file", "FileOps", "PermissionDenied", "ro dir")
    sig2 = extract("write_file", {"path": path2, "content": "y"}, w)
    return matcher.check(sig2, store, exclude_agent=agent2)


def test_subsumption_mid_catches_different_file_same_broken_dir():
    assert _fail_then_check(Matcher("mid", "subsumption"))


def test_fine_equality_misses_different_file():
    assert not _fail_then_check(Matcher("fine", "equality"))


def test_self_exclusion():
    hits = _fail_then_check(Matcher("mid", "subsumption"), agent2="FileOps")
    assert hits == []  # never warned about your own record


def test_no_match_on_healthy_dir():
    hits = _fail_then_check(Matcher("mid", "subsumption"), path2="/tmp/ok.txt")
    assert hits == []  # silence by default


def test_coarse_recall_geq_mid_geq_fine():
    got = {lv: bool(_fail_then_check(Matcher(lv, "subsumption")))
           for lv in ("coarse", "mid", "fine")}
    assert got["coarse"] >= got["mid"] >= got["fine"]  # Lemma 1 direction


# ---------- store ----------

def test_retry_exhaustion_recorded_once():
    w = _world()
    store = FailureStore()
    sig = extract("write_file", {"path": "/data/a", "content": "x"}, w)
    store.add(sig, "write_file", "A", "PermissionDenied")
    store.add(sig, "write_file", "A", "PermissionDenied")
    assert len(store.records) == 1 and store.records[0].retry_exhausted


def test_episode_scope_reset():
    store = FailureStore(scope="episode")
    sig = Signature(frozenset({("tool", "x")}))
    store.add(sig, "x", "A", "NotFound")
    store.reset_episode()
    assert store.records == []


# ---------- end-to-end on scenarios ----------

def _scenarios():
    ps = sorted(glob.glob(os.path.join(HERE, "envs/scenarios/*.yaml")))
    assert len(ps) == 5
    return ps


def test_pfti_beats_vanilla_end_to_end():
    for p in _scenarios():
        scen = load_scenario(p)
        base = run_episode(scen, mode="off")
        pfti = run_episode(scen, mode="inject", level="mid")
        assert pfti["success"], f"{scen['name']} should succeed with PFTI"
        assert pfti["repeated_peer_failures"] == 0
        assert base["repeated_peer_failures"] >= pfti["repeated_peer_failures"]


def test_shadow_mode_injects_nothing_but_measures():
    scen = load_scenario(_scenarios()[0])
    shadow = run_episode(scen, mode="shadow", level="mid")
    base = run_episode(scen, mode="off")
    assert shadow["failures"] == base["failures"]  # no behavioral effect
    s = shadow_stats(shadow["trace"])
    assert s["TP"] >= 1 and s["precision"] == 1.0  # oracle-confirmed warning


def test_oracle_matches_tool_behavior():
    w = _world()
    err = would_fail("write_file", {"path": "/data/x", "content": "c"}, w)
    assert err and err[0] == "PermissionDenied"
    with pytest.raises(ToolError):
        TOOLS["write_file"](w, path="/data/x", content="c")
    assert would_fail("write_file", {"path": "/tmp/x", "content": "c"}, w) is None
