@echo off
REM Re-analysis of the EXISTING results (no Ollama, ~1 minute):
REM   Exp1 from bench_results\core_matrix.json
REM   Exp2 from waw_results\waw_llm.json
REM Output: review_reanalysis\exp1_reanalysis.md, exp2_reanalysis.md
setlocal
set HERE=%~dp0
set PY=%HERE%.venv\Scripts\python.exe
cd /d "%HERE%"
"%PY%" -m pip install --quiet statsmodels scipy pandas
"%PY%" -m pfti.eval.reanalyze_exp1
"%PY%" -m pfti.eval.reanalyze_exp2
echo.
echo Done. See review_reanalysis\
pause
endlocal
