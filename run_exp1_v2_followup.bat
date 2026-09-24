@echo off
REM ============================================================
REM  Follow-up to run_exp1_v2.bat (after reading its traces):
REM   A) the 2 reasoning scenarios, now solvable (all 4 models)
REM   B) llama3.1:8b on all 11 scenarios with --parse-text-calls
REM      (llama writes tool calls as text after warnings; this
REM      checks whether its regressions are only that formatting
REM      problem). Applied identically in off/shadow/inject.
REM  Time: roughly 30-45 min.  Log: bench_v2_followup_log.txt
REM ============================================================
setlocal
set HERE=%~dp0
set PY=%HERE%.venv\Scripts\python.exe
set LOG=%HERE%bench_v2_followup_log.txt
cd /d "%HERE%"
echo ===== Exp1 v2 follow-up started %DATE% %TIME% ===== > "%LOG%"
if not exist "%PY%" ( echo .venv not found & pause & exit /b 1 )

echo [1/4] Offline tests...
"%PY%" -m pytest pfti\tests\test_bench.py pfti\tests\test_review_fixes.py -q >> "%LOG%" 2>&1
if errorlevel 1 ( echo Tests FAILED - see bench_v2_followup_log.txt & pause & exit /b 1 )
echo   passed.

echo [2/4] A) Reasoning scenarios, all 4 models...
"%PY%" -u -m pfti.bench.run_all --scenarios reasoning --repeats 1 --save-traces --no-routing --out bench_results_v2_reasoning >> "%LOG%" 2>&1

echo [3/4] B) llama3.1:8b, all scenarios, text tool calls accepted...
"%PY%" -u -m pfti.bench.run_all --models llama3.1:8b --scenarios all --repeats 1 --save-traces --no-routing --parse-text-calls --out bench_results_v2_llama_textcalls >> "%LOG%" 2>&1

echo [4/4] Trace reports...
"%PY%" -m pfti.bench.trace_report bench_results_v2_reasoning >> "%LOG%" 2>&1
"%PY%" -m pfti.bench.trace_report bench_results_v2_llama_textcalls >> "%LOG%" 2>&1
(echo ===== DONE %DATE% %TIME% ===== ) >> "%LOG%"
echo.
echo Done. Results: bench_results_v2_reasoning\  and  bench_results_v2_llama_textcalls\
pause
endlocal
