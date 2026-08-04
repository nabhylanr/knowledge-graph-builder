@echo off
REM Wrapper for the long-lived chunk-sync listener on Windows.
REM
REM Unlike chunksync.cmd this never exits on its own — it is registered with an
REM at-startup trigger and restarted by Task Scheduler if it dies.

cd /d "%~dp0.."
".venv\Scripts\python.exe" run_listen.py >> "chunksync-listener.log" 2>&1
