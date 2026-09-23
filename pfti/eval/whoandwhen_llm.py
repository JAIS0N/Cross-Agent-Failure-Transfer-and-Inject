"""Who&When real-data, real-LLM before/after study (scoped).

whoandwhen.py already answers: "for what fraction of real failure logs does
a LATER step's signature match an EARLIER failure's signature" (transfer
opportunity -- a moment where PFTI's peer warning would have fired). This
module goes one step further for the subset where that later match is a
real *code-execution* failure (a genuine Python/shell exception or nonzero
exitcode from AutoGen's Computer_terminal):

  1. Replay the REAL conversation history, verbatim, up to (not including)
     the assistant step that proposed the code which failed.
  2. Ask a real LLM -- playing that same agent, with its real system
     prompt -- to propose the next code block, TWICE:
       off    - no warning (vanilla)
       inject - the exact same PFTI peer-failure warning template used
                everywhere else in this codebase (core/inject.py), built
                from the real EARLIER failure in this same trajectory
                whose signature matched.
  3. Actually EXECUTE both regenerated code blocks as a subprocess (short
     timeout, isolated temp dir) and classify the real output with the
     SAME error-class extractor whoandwhen.py uses on the original data.
  4. "Repeated" = the regenerated code's real execution output fails with
     the same tool + error class as the ORIGINAL historical failure (see
     grade_execution for why the leveled Matcher is not used for this).

  KNOWN LIMITATIONS (found in review, kept for reproducibility of the first
  run; the redesigned study is waw_repro.py):
    * find_transfer_points uses Matcher(level='mid'), which reduces a
      Who&When signature to (tool=code_exec), so the "matching" earlier
      failure usually has a DIFFERENT error class (31 of 36 cases).
    * the earlier failure is from the same trajectory and is almost always
      still inside the replayed context (34 of 36), and exclude_agent=None,
      so the warning is mostly redundant within-trajectory memory rather
      than peer transfer.

This is real code, actually executed, replaying a task a real agent
genuinely failed on and a human annotator confirmed -- not a synthetic
scenario, and not just text pattern-matching on the model's prose.

SAFETY: this executes model-generated code as a local subprocess with a
short timeout in an isolated temp directory. It should only be run
somewhere you're comfortable running short LLM-generated Python/shell
snippets (this module never runs on Anthropic's infrastructure; it runs
wherever you invoke it, alongside your local Ollama).
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

from ..core.matcher import Matcher
from ..core.store import FailureStore, Record
from ..core import inject as inject_mod
from .whoandwhen import load_logs, step_signature

CODE_RE = re.compile(r"```(python|sh|bash)\s*\n(.*?)```", re.DOTALL)


def extract_code(content: str):
    """Last fenced python/sh/bash code block in `content`, or None."""
    if not content:
        return None
    matches = CODE_RE.findall(content)
    if not matches:
        return None
    lang, code = matches[-1]
    code = code.strip()
    return (lang, code) if code else None


def find_transfer_points(only_op: str | None = "code_exec"):
    """Every real transfer-opportunity EVENT (not just per-log) where a
    later failstep's signature matches an earlier one in the same real
    trajectory, restricted to `only_op` (default: code_exec, so we can
    actually re-execute), and where the failing step has a preceding
    assistant turn with an extractable code block to regenerate."""
    matcher = Matcher(level="mid", mode="subsumption")
    cases = []
    for path, d in load_logs():
        hist = d.get("history", [])
        sysprompts = d.get("system_prompt", {})
        fail_sigs = []
        for i, step in enumerate(hist):
            res = step_signature(step)
            if res is None:
                continue
            sig, ec = res
            op = dict(sig.preds).get("tool", "?")
            fail_sigs.append((i, sig, ec, op, step))

        store = FailureStore()
        for (i, sig, ec, op, step) in fail_sigs:
            hits = matcher.check(sig, store, exclude_agent=None)
            if hits and (only_op is None or op == only_op):
                code_idx = None
                for j in range(i - 1, -1, -1):
                    if extract_code(hist[j].get("content", "")):
                        code_idx = j
                        break
                if code_idx is not None:
                    cases.append({
                        "log_path": path,
                        "question": d.get("question", ""),
                        "history": hist,
                        "system_prompts": sysprompts,
                        "later_index": i,
                        "later_agent": step.get("name", "unknown"),
                        "later_ec": ec,
                        "later_sig": sig,
                        "code_index": code_idx,
                        "code_agent": hist[code_idx].get("name", "unknown"),
                        "peer_record": hits[0],
                    })
            content = str(step.get("content", ""))[:160]
            store.add(sig, op, dict(sig.preds).get("agent", "?"), ec,
                      content, os.path.basename(path), i)
    return cases


MAX_SYS_CHARS = 3000       # per-agent system prompts can be very long
MAX_HISTORY_CHARS = 8000   # ~2k tokens: prompt + 700 out fits a 4k context
MAX_TURN_CHARS = 2500      # a single huge turn (e.g. a web page dump)


def _clip(s: str, n: int) -> str:
    s = str(s)
    return s if len(s) <= n else s[: n // 2] + "\n[... clipped ...]\n" + s[-n // 2:]


def build_messages(case: dict, warning_rec: Record | None = None,
                    level: str = "mid"):
    """Replay the real history up to the failing code proposal. Long logs
    are bounded so they fit a small model's context: the first turn (the
    task statement) is always kept, then as many of the MOST RECENT turns
    as fit, with an explicit marker for what was omitted -- identical for
    off and inject, so the comparison stays fair."""
    hist, upto, agent = case["history"], case["code_index"], case["code_agent"]
    sysprompt = case["system_prompts"].get(
        agent, f"You are {agent}, an agent in a multi-agent team working "
               f"on a task together with other named agents.")
    msgs = [{"role": "system", "content": _clip(sysprompt, MAX_SYS_CHARS)}]

    def as_msg(step):
        role = "assistant" if step.get("name") == agent else "user"
        body = _clip(step.get("content", ""), MAX_TURN_CHARS)
        return {"role": role, "content": f"[{step.get('name', '?')}]: {body}"}

    prior = hist[:upto]
    if prior:
        first = as_msg(prior[0])
        budget = MAX_HISTORY_CHARS - len(first["content"])
        tail = []
        for step in reversed(prior[1:]):
            m = as_msg(step)
            if len(m["content"]) > budget:
                break
            tail.append(m)
            budget -= len(m["content"])
        tail.reverse()
        omitted = len(prior) - 1 - len(tail)
        msgs.append(first)
        if omitted > 0:
            msgs.append({"role": "user",
                         "content": f"[... {omitted} earlier turns omitted ...]"})
        msgs.extend(tail)
    if warning_rec is not None:
        msgs.append({"role": "user", "content": inject_mod.render(warning_rec, level)})
    msgs.append({"role": "user", "content": (
        "Continue the task. Propose your next step as a single fenced "
        "python or sh code block (```python or ```sh) to execute. Keep it "
        "self-contained and runnable as-is.")})
    return msgs


_UNSAFE = re.compile(
    r"(rm\s+-[a-z]*r|rmdir|shutil\.rmtree|os\.remove|os\.unlink|os\.rmdir|"
    r"Path\([^)]*\)\.unlink|\.unlink\(|os\.system|subprocess|format\s+[a-z]:|"
    r"del\s+/|rd\s+/s|mkfs|shutdown|reg\s+delete|chmod|chown|:\(\)\s*\{)",
    re.I)


def is_unsafe(code: str) -> bool:
    """Conservative denylist: never execute model code that deletes files,
    spawns shells, or touches system state. Such cases are recorded as
    'unsafe' and excluded from rates (identically for off and inject)."""
    return bool(_UNSAFE.search(code or ""))


def execute_code(lang: str, code: str, timeout: int = 12) -> str:
    """Actually run model-generated code in an isolated temp dir with a
    short timeout, and format the result exactly like the real dataset's
    Computer_terminal tool-output, so the same error-class extractor
    applies unchanged."""
    with tempfile.TemporaryDirectory() as td:
        ext = "py" if lang == "python" else "sh"
        path = os.path.join(td, f"snippet.{ext}")
        # utf-8 explicitly: on Windows the default (cp1252) crashes on
        # characters like '\u2248' in model code (1 of 144 rows in the
        # first run was lost this way).
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        cmd = [sys.executable, path] if lang == "python" else ["bash", path]
        try:
            p = subprocess.run(cmd, cwd=td, capture_output=True, text=True,
                               timeout=timeout, encoding="utf-8",
                               errors="replace",
                               env={**os.environ, "PYTHONIOENCODING": "utf-8",
                                    "PYTHONUTF8": "1"})
            rc, out, err = p.returncode, p.stdout, p.stderr
        except subprocess.TimeoutExpired:
            rc, out, err = 124, "", "TimeoutExpired: code did not finish in time"
        except Exception as e:  # e.g. bash missing
            rc, out, err = 1, "", f"{type(e).__name__}: {e}"
    status = "execution succeeded" if rc == 0 else "execution failed"
    return f"exitcode: {rc} ({status})\nCode output: {out}\n{err}"


def grade_execution(exec_out: str, later_agent: str, later_sig):
    """Grade the REAL execution output of a regenerated code block.

    Returns (failed, same_error):
      failed     -- the regenerated code produced any failure signature
      same_error -- it failed with the SAME tool + error_class as the real
                    historical failure (strict subset check on those two
                    dims of the un-projected signature).

    Why not Matcher at level='mid': signature.DIM_LEVEL does not list the
    Who&When dimensions (error_class, agent, artifact_ext, missing_module),
    so project('mid') reduces a Who&When signature to just (tool=...). At
    that level any code_exec failure 'matches' any other -- too lenient to
    count as a repeat. We grade on the real error class instead."""
    res = step_signature({"content": exec_out, "name": later_agent})
    if res is None:
        return False, False  # no failure signature: the code ran cleanly
    new_sig, _ = res
    key = {p for p in later_sig.preds if p[0] in ("tool", "error_class")}
    return True, key <= new_sig.preds


def _fake_turn(later_ec: str, warned: bool) -> str:
    """Deterministic offline stand-in: 'off' reproduces a real error of the
    case's own class where we know how to synthesize one; 'inject'
    (warned=True) always returns working code. No Ollama needed -- proves
    the build_messages -> execute -> classify -> match pipeline end to end
    on real trajectories before spending real model calls."""
    if warned:
        return "```python\nprint('ok')\n```"
    bugs = {
        "NameError": "print(undefined_variable_xyz)",
        "TypeError": "1 + 'a'",
        "KeyError": "{}['missing_key']",
        "AttributeError": "None.some_attribute",
        "FileNotFoundError": "open('/nonexistent/path/xyz.txt')",
        "ExecutionError": "import sys; sys.exit(1)",
        "ValueError": "int('not-a-number')",
    }
    code = bugs.get(later_ec, "raise Exception('synthetic failure')")
    return f"```python\n{code}\n```"


def run_turn(client, model: str, messages: list, use_fake: bool = False,
             later_ec: str = "", warned: bool = False) -> str:
    if use_fake:
        return _fake_turn(later_ec, warned)
    resp = client.chat.completions.create(model=model, messages=messages,
                                          temperature=0.2, max_tokens=700)
    return resp.choices[0].message.content or ""


def run_case(case: dict, client, model: str, use_fake: bool = False,
             level: str = "mid") -> dict:
    result = {"log": os.path.basename(case["log_path"]), "model": model,
              "code_agent": case["code_agent"], "later_ec": case["later_ec"]}
    for cond, warn_rec in (("off", None), ("inject", case["peer_record"])):
        messages = build_messages(case, warning_rec=warn_rec, level=level)
        resp = run_turn(client, model, messages, use_fake=use_fake,
                        later_ec=case["later_ec"], warned=warn_rec is not None)
        extracted = extract_code(resp)
        result[f"{cond}_mentions_warning"] = (
            warn_rec is not None and _acknowledges_warning(resp))
        if extracted is None:
            # no parseable code block: recorded, excluded from rates
            result[f"{cond}_parsed"] = False
            result[f"{cond}_failed"] = None
            result[f"{cond}_same_error"] = None
            result[f"{cond}_exec"] = None
            continue
        lang, code = extracted
        if lang != "python":
            # shell blocks: bash is usually absent on Windows, so executing
            # them would fake a failure. Recorded as unrunnable, excluded
            # from rates identically in both conditions.
            result[f"{cond}_parsed"] = False
            result[f"{cond}_unrunnable"] = True
            result[f"{cond}_failed"] = None
            result[f"{cond}_same_error"] = None
            result[f"{cond}_exec"] = None
            result[f"{cond}_code"] = code[:500]
            continue
        if is_unsafe(code):
            result[f"{cond}_parsed"] = False
            result[f"{cond}_unsafe"] = True
            result[f"{cond}_failed"] = None
            result[f"{cond}_same_error"] = None
            result[f"{cond}_exec"] = None
            result[f"{cond}_code"] = code[:500]
            continue
        exec_out = execute_code(lang, code)
        failed, same = grade_execution(exec_out, case["later_agent"],
                                       case["later_sig"])
        result[f"{cond}_parsed"] = True
        result[f"{cond}_failed"] = failed
        result[f"{cond}_same_error"] = same
        result[f"{cond}_exec"] = exec_out[:300]
        result[f"{cond}_code"] = code[:500]
    return result


_ACK = re.compile(r"(warning|previous(ly)? fail|peer|avoid|instead|"
                  r"verify|check (if|whether|that)|try/except|try:)", re.I)


def _acknowledges_warning(text: str) -> bool:
    """Cheap signal: does the model's reply engage with the warning at all?"""
    return bool(_ACK.search(text or ""))


def summarize(rows: list[dict], model: str) -> dict:
    s = {"model": model}
    for cond in ("off", "inject"):
        rs = [r for r in rows if r["model"] == model]
        parsed = [r for r in rs if r.get(f"{cond}_parsed")]
        s[f"{cond}_n"] = len(rs)
        s[f"{cond}_parsed"] = len(parsed)
        s[f"{cond}_unsafe_skipped"] = sum(bool(r.get(f"{cond}_unsafe"))
                                          for r in rs)
        s[f"{cond}_fail_rate"] = (sum(r[f"{cond}_failed"] for r in parsed)
                                  / len(parsed)) if parsed else None
        s[f"{cond}_same_error_rate"] = (
            sum(r[f"{cond}_same_error"] for r in parsed) / len(parsed)
            if parsed else None)
    s["inject_ack_rate"] = (sum(r["inject_mentions_warning"] for r in rs)
                            / len(rs)) if rs else None
    return s


def run_study(models: list[str], client=None, use_fake: bool = False,
              level: str = "mid", limit: int | None = None,
              repeats: int = 1):
    cases = find_transfer_points(only_op="code_exec")
    if limit:
        cases = cases[:limit]
    rows, summaries = [], []
    for model in models:
        for rep in range(repeats):
            for k, case in enumerate(cases):
                print(f"    [{model}] rep {rep + 1}/{repeats} case "
                      f"{k + 1}/{len(cases)} {os.path.basename(case['log_path'])}",
                      flush=True)
                try:
                    r = run_case(case, client, model, use_fake=use_fake,
                                 level=level)
                except Exception as e:  # model missing / server down
                    r = {"log": os.path.basename(case["log_path"]),
                         "model": model, "error": f"{type(e).__name__}: {e}",
                         "off_parsed": False, "inject_parsed": False,
                         "inject_mentions_warning": False}
                r["repeat"] = rep
                rows.append(r)
        s = summarize(rows, model)
        summaries.append(s)
        f = lambda v: "  n/a" if v is None else f"{v:5.0%}"
        print(f"  {model:<14} cases={len(cases)} x{repeats}  "
              f"fail off={f(s['off_fail_rate'])} inj={f(s['inject_fail_rate'])}  "
              f"same-err off={f(s['off_same_error_rate'])} "
              f"inj={f(s['inject_same_error_rate'])}  "
              f"parsed off={s['off_parsed']}/{s['off_n']} "
              f"inj={s['inject_parsed']}/{s['inject_n']}")
    return {"n_cases": len(cases), "repeats": repeats,
            "summaries": summaries, "rows": rows}
