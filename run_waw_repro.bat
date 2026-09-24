@echo off
REM ============================================================
REM  Experiment 2, redesigned after Prof. Hailu's review:
REM  reproduction-filtered Who^&When study.
REM   Phase 1: 6 samples/case (T=0.7) under OFF for all ~90
REM            code-failure moments; keep cases that repeat the
REM            historical error at least 2/6 times.
REM   Phase 2: fresh 6 OFF + 6 INJECT samples on kept cases.
REM  Long run: roughly 2 h per model (6-9 h for all 4) - start it
REM  before bed. It RESUMES: if it stops, just run this file again.
REM  To test one model first:  edit MODELS below.
REM  Executes short model-written Python in a temp folder (12 s
REM  timeout); deleting/shell/system code is never executed.
REM  Log: waw_repro_log.txt   Output: waw_repro_results\
REM ============================================================
setlocal
set HERE=%~dp0
set PY=%HERE%.venv\Scripts\python.exe
set LOG=%HERE%waw_repro_log.txt
set PYTHONIOENCODING=utf-8
set MODELS=qwen2.5:3b,qwen2.5:7b,qwen2.5:14b,llama3.1:8b
cd /d "%HERE%"
echo ===== WaW repro started %DATE% %TIME% ===== >> "%LOG%"
if not exist "%PY%" ( echo .venv not found & pause & exit /b 1 )

echo [1/3] Installing analysis + task libraries (statsmodels, scipy, pandas...)...
"%PY%" -m pip install --quiet openai statsmodels scipy pandas requests openpyxl beautifulsoup4 >> "%LOG%" 2>&1

echo [2/3] Offline tests on the real logs (no Ollama)...
"%PY%" -m pytest pfti\tests\test_review_fixes.py pfti\tests\test_whoandwhen_llm.py -q >> "%LOG%" 2>&1
if errorlevel 1 ( echo Tests FAILED - see waw_repro_log.txt & pause & exit /b 1 )
echo   passed.

echo [3/3] Running the study (resumable)...  models: %MODELS%
"%PY%" -u -m pfti.eval.waw_repro --models %MODELS% --out waw_repro_results >> "%LOG%" 2>&1
(echo ===== DONE %DATE% %TIME% ===== ) >> "%LOG%"
echo.
echo Done. Open waw_repro_results\summary.md and waw_repro_cases.png
pause
endlocal
