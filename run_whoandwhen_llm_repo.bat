@echo off
REM ============================================================
REM  Who&When real-LLM before/after study (PFTI off vs inject on
REM  36 REAL code-execution failure moments from Who&When).
REM  Run AFTER run_pfti_bench_repo.bat has finished (both use the
REM  same Ollama models / GPU).
REM  NOTE: this executes short model-generated Python snippets
REM  in a temp folder with a 12s timeout. Snippets that delete
REM  files, spawn shells or change system state are skipped.
REM  Log: waw_log.txt   Output: waw_results\
REM ============================================================
setlocal
set HERE=%~dp0
set PY=%HERE%.venv\Scripts\python.exe
set LOG=%HERE%waw_log.txt
echo ===== Who^&When LLM run started %DATE% %TIME% ===== > "%LOG%"

if not exist "%PY%" (
  echo Could not find .venv\Scripts\python.exe next to this script.
  pause
  exit /b 1
)

echo [0/3] Installing the libraries the original Who^&When agents used (pandas, requests, openpyxl, bs4)...
(echo [0/3] pip install pandas requests openpyxl beautifulsoup4) >> "%LOG%"
"%PY%" -m pip install --quiet pandas requests openpyxl beautifulsoup4 >> "%LOG%" 2>&1

echo [1/3] Offline sanity check on the real logs (no Ollama)...
(echo [1/3] pytest test_whoandwhen_llm.py) >> "%LOG%"
"%PY%" -m pytest "%HERE%pfti\tests\test_whoandwhen_llm.py" -q >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Sanity check FAILED -- see waw_log.txt. Not running the real study.
  pause
  exit /b 1
)
echo   passed.

echo [2/3] Checking Ollama...
where ollama >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Ollama not found on PATH.
  pause
  exit /b 1
)

echo [3/3] Running the REAL study (4 models x 36 cases x off/inject)...
(echo [3/3] run_whoandwhen_llm) >> "%LOG%"
cd /d "%HERE%"
"%PY%" -u -m pfti.eval.run_whoandwhen_llm --out waw_results >> "%LOG%" 2>&1
(echo ===== DONE %DATE% %TIME% ===== ) >> "%LOG%"

echo.
echo Done. See waw_log.txt and waw_results\ (summary.md, waw_before_after.png).
pause
endlocal
