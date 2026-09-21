@echo off
REM Commit ONLY the new benchmark files and push to GitHub (origin/main).
REM Your other local changes (if any) are left untouched and uncommitted.
REM Log: git_push_log.txt
setlocal
set HERE=%~dp0
set LOG=%HERE%git_push_log.txt
cd /d "%HERE%"
echo ===== git push run %DATE% %TIME% ===== > "%LOG%"

where git >> "%LOG%" 2>&1
if errorlevel 1 (
  echo git not found on PATH. >> "%LOG%"
  echo git was not found on PATH. Install Git for Windows, or run these steps in GitHub Desktop.
  pause
  exit /b 1
)

echo [1/5] Current state...
(echo [1/5] status before) >> "%LOG%"
git status --short >> "%LOG%" 2>&1
git log --oneline -3 >> "%LOG%" 2>&1

echo [2/5] Staging the new files only...
(echo [2/5] add) >> "%LOG%"
git add .gitignore >> "%LOG%" 2>&1
git add pfti/bench/__init__.py pfti/bench/providers.py pfti/bench/scenarios_llm.py pfti/bench/core_matrix.py pfti/bench/routing_matrix.py pfti/bench/run_all.py pfti/bench/report.py pfti/bench/RUN_BENCH.md >> "%LOG%" 2>&1
git add pfti/eval/whoandwhen_llm.py pfti/eval/run_whoandwhen_llm.py >> "%LOG%" 2>&1
git add pfti/tests/test_bench.py pfti/tests/test_whoandwhen_llm.py >> "%LOG%" 2>&1
git add run_pfti_bench_repo.bat run_whoandwhen_llm_repo.bat push_to_github.bat >> "%LOG%" 2>&1
git add bench_results >> "%LOG%" 2>&1
(echo --- staged:) >> "%LOG%"
git diff --cached --stat >> "%LOG%" 2>&1

echo [3/5] Committing...
(echo [3/5] commit) >> "%LOG%"
git commit -F _commit_msg.txt >> "%LOG%" 2>&1
if errorlevel 1 (
  echo Commit failed or nothing to commit - see git_push_log.txt
  pause
  exit /b 1
)
del _commit_msg.txt

echo [4/5] Pushing to origin main (a GitHub sign-in window may appear)...
(echo [4/5] push) >> "%LOG%"
git push origin main >> "%LOG%" 2>&1
if errorlevel 1 (
  echo PUSH FAILED - see git_push_log.txt. The commit is saved locally.
  pause
  exit /b 1
)

echo [5/5] Done.
(echo [5/5] after) >> "%LOG%"
git log --oneline -3 >> "%LOG%" 2>&1
git status --short >> "%LOG%" 2>&1
(echo ===== DONE =====) >> "%LOG%"
echo Pushed. See git_push_log.txt
pause
endlocal
