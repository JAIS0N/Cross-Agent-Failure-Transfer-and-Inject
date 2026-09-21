"""Who&When real-LLM study plumbing, validated offline on the REAL logs with
the deterministic fake (no Ollama). Skips if whoandwhen_data/ is absent."""
import os

import pytest

from pfti.eval.whoandwhen import DATA
from pfti.eval import whoandwhen_llm as W

pytestmark = pytest.mark.skipif(not os.path.isdir(DATA),
                                reason="whoandwhen_data/ not present")


def test_finds_real_code_exec_transfer_cases():
    cases = W.find_transfer_points()
    assert len(cases) > 0
    for c in cases:
        assert dict(c["later_sig"].preds)["tool"] == "code_exec"
        assert c["code_index"] < c["later_index"]
        assert W.extract_code(c["history"][c["code_index"]]["content"])


def test_messages_bounded_and_identical_except_warning():
    for c in W.find_transfer_points()[:10]:
        off = W.build_messages(c)
        inj = W.build_messages(c, warning_rec=c["peer_record"])
        assert len(inj) == len(off) + 1          # only the warning differs
        assert "PEER FAILURE WARNING" in inj[-2]["content"]
        assert not any("PEER FAILURE WARNING" in m["content"] for m in off)
        hist_chars = sum(len(m["content"]) for m in off[1:])
        assert hist_chars <= W.MAX_HISTORY_CHARS + 3000  # first turn + marker


def test_execution_grading_is_real_and_strict():
    out = W.execute_code("python", "print(undefined_variable_xyz)")
    assert "exitcode: 1" in out and "NameError" in out
    ok = W.execute_code("python", "print(2 + 2)")
    assert "exitcode: 0" in ok
    c = next(c for c in W.find_transfer_points() if c["later_ec"] == "NameError")
    assert W.grade_execution(out, c["later_agent"], c["later_sig"]) == (True, True)
    assert W.grade_execution(ok, c["later_agent"], c["later_sig"]) == (False, False)
    other = W.execute_code("python", "{}['k']")          # KeyError != NameError
    assert W.grade_execution(other, c["later_agent"], c["later_sig"]) == (True, False)


def test_unsafe_code_is_never_executed():
    for bad in ("import shutil; shutil.rmtree('x')", "import os; os.remove('f')",
                "rm -rf /tmp/x", "import subprocess; subprocess.run(['ls'])",
                "os.system('del /q *')"):
        assert W.is_unsafe(bad)
    for fine in ("import pandas as pd\nprint(pd.__version__)",
                 "print(sum(range(10)))", "open('out.txt','w').write('x')"):
        assert not W.is_unsafe(fine)


def test_fake_inject_beats_off_end_to_end():
    res = W.run_study(["fake"], use_fake=True, limit=12)
    s = res["summaries"][0]
    assert s["off_fail_rate"] > s["inject_fail_rate"]
    assert s["off_same_error_rate"] >= s["inject_same_error_rate"]
