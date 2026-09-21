@echo off
REM ============================================================
REM  PFTI real-LLM before/after benchmark (repeats=3), run IN
REM  PLACE inside your actual project repo, using its own .venv
REM  (so it reuses matplotlib/pytest/pyyaml already installed
REM  there instead of touching system Python).
REM  Run this by double-clicking it from this same folder
REM  (the one containing .venv\ and pfti\).
REM  Everything is logged to bench_log_repo.txt.
REM ============================================================
setlocal
set HERE=%~dp0
set PY=%HERE%.venv\Scripts\python.exe
set LOG=%HERE%bench_log_repo.txt
echo ===== PFTI repo bench run started %DATE% %TIME% ===== > "%LOG%"

echo [1/4] Checking venv Python...
(echo [1/4] Checking venv Python...) >> "%LOG%"
if not exist "%PY%" (
  echo Could not find %PY% >> "%LOG%"
  echo Could not find .venv\Scripts\python.exe next to this script.
  echo Make sure you double-click this from inside the Systemsproject folder.
  pause
  exit /b 1
)
"%PY%" --version >> "%LOG%" 2>&1

echo [2/4] Installing openai into the existing venv (matplotlib/pytest/pyyaml already present)...
(echo [2/4] Installing openai...) >> "%LOG%"
"%PY%" -m pip install --quiet openai >> "%LOG%" 2>&1
"%PY%" -m pip show openai >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Could not install openai into the venv. See bench_log_repo.txt.
  pause
  exit /b 1
)

echo [3/4] Sanity-checking the merged bench module (offline, no Ollama needed)...
(echo [3/4] Running pytest pfti/tests/test_bench.py ...) >> "%LOG%"
"%PY%" -m pytest "%HERE%pfti\tests\test_bench.py" -q >> "%LOG%" 2>&1
if errorlevel 1 (
  echo pytest sanity check FAILED -- see bench_log_repo.txt before trusting real results.
  echo Not running the real matrix.
  pause
  exit /b 1
)
echo   pytest sanity check passed.

echo [4/4] Checking Ollama and running the REAL matrix with --repeats 3 (this takes a while)...
(echo [4/4] Running benchmark matrix, repeats=3 ...) >> "%LOG%"
where ollama >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Ollama not found on PATH. It was working for the earlier run -- if this
  echo fails, make sure Ollama is installed and on PATH, then re-run.
  pause
  exit /b 1
)
"%PY%" -m pfti.bench.run_all --out bench_results --repeats 3 >> "%LOG%" 2>&1
(echo ===== DONE %DATE% %TIME% ===== ) >> "%LOG%"

echo.
echo Done. See bench_log_repo.txt for the full log and bench_results\ for output.
echo (This window will stay open -- press any key to close it.)
pause
endlocal
