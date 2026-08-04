@echo off
REM Wrapper for the scheduled chunk sync on Windows.
REM
REM Task Scheduler does not capture stdout, and quoting a redirect inside a task
REM argument is fragile — so the redirect lives here instead.
REM
REM cd to the repo root (this file sits in deploy\) so run_sync.py finds .env.

cd /d "%~dp0.."
".venv\Scripts\python.exe" run_sync.py >> "chunksync.log" 2>&1
