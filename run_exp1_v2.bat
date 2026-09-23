@echo off
REM ============================================================
REM  Experiment 1, revised after Prof. Hailu's review.
REM  All 11 scenarios (5 original + 3 adversarial false-positive
REM  + 1 control + 2 reasoning-bug), off/shadow/inject, 4 models,
REM  1 repeat (temperature 0: repeats add ~nothing), full traces
REM  saved, routing skipped. Then the "what happens after a
REM  warning" harm report is built from the traces.
REM  Time: roughly 45-60 min.  Do NOT run at the same time as
REM  run_waw_repro.bat (both need the GPU).
REM  Log: bench_v2_log.txt   Output: bench_results_v2\
REM ============================================================
setlocal
set HERE=%~dp0
set PY=%HERE%.venv\Scripts\python.exe
set LOG=%HERE%bench_v2_log.txt
cd /d "%HERE%"
echo ===== Exp1 v2 started %DATE% %TIME% ===== > "%LOG%"
if not exist "%PY%" ( echo .venv not found & pause & exit /b 1 )

echo [1/4] Installing openai (if missing)...
"%PY%" -m pip install --quiet openai >> "%LOG%" 2>&1

echo [2/4] Offline tests (no Ollama)...
"%PY%" -m pytest pfti\tests\test_bench.py pfti\tests\test_review_fixes.py -q >> "%LOG%" 2>&1
if errorlevel 1 ( echo Tests FAILED - see bench_v2_log.txt & pause & exit /b 1 )
echo   passed.

echo [3/4] Running the real benchmark (all scenarios, traces on)...
"%PY%" -u -m pfti.bench.run_all --scenarios all --repeats 1 --save-traces --no-routing --out bench_results_v2 >> "%LOG%" 2>&1

echo [4/4] Building the trace (harm) report...
"%PY%" -m pfti.bench.trace_report bench_results_v2 >> "%LOG%" 2>&1
(echo ===== DONE %DATE% %TIME% ===== ) >> "%LOG%"
echo.
echo Done. Open bench_results_v2\summary.md and bench_results_v2\trace_report.md
pause
endlocal
