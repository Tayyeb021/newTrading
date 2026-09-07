@echo off
rem Unattended shadow week: live MT5 data, paper execution, NO real orders.
rem
rem Started by Windows Task Scheduler (task "TradingShadowWeek") at the Sunday
rem open, 2026-09-06 21:00 UTC, and runs until the Friday close, 2026-09-11
rem 21:00 UTC. The end is an absolute time so that if the task restarts a
rem crashed run, the restart still ends at the close instead of running on.
rem
rem Self-healing: the scheduled task re-fires every 15 minutes with
rem MultipleInstances=IgnoreNew, so a running instance is left alone and a dead
rem one is replaced. This exists because on 2026-09-07 the first run was killed
rem by a console interrupt (exit 3221225786, STATUS_CONTROL_C_EXIT) after eleven
rem hours, and Task Scheduler's "restart on failure" does NOT fire on a
rem non-zero exit code - only on a failure to start. Eleven hours of record were
rem lost before anyone noticed.
rem
rem Stop it early:   create the file  state\SHADOW_KILL   (the runner halts and
rem                  refuses every trade until the file is removed), or end the
rem                  task:  schtasks /End /TN TradingShadowWeek
rem Watch it:        state\shadow_week.log     (this file)
rem                  state\shadow_journal.jsonl (every decision, fill, breach)
rem                  state\shadow_reports.md    (daily digest, 21:30 UTC)
rem
rem The MetaTrader 5 terminal must be running and logged in, in the same Windows
rem session as this task (the task is registered "run only when user is logged
rem on" for exactly that reason - disconnect RDP, do not sign out).

cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
if not exist state mkdir state

echo ==== %DATE% %TIME% shadow week start ==== >> state\shadow_week.log
"C:\Users\azureuser\AppData\Local\Programs\Python\Python312\python.exe" scripts\shadow.py --until 2026-09-11T21:00Z --poll 10 --quiet >> state\shadow_week.log 2>&1
set RC=%ERRORLEVEL%
echo ==== %DATE% %TIME% shadow week end, exit %RC% ==== >> state\shadow_week.log
exit /b %RC%
