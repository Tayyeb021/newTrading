@echo off
rem Digest of the shadow run, appended to state\shadow_reports.md.
rem   shadow_report.cmd        last 24 hours   (task "TradingShadowReport", daily 21:30 UTC)
rem   shadow_report.cmd 168    the whole week  (task "TradingShadowWeekReport", Friday 21:40 UTC)

cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
if not exist state mkdir state
set HOURS=%1
if "%HOURS%"=="" set HOURS=24

"C:\Users\azureuser\AppData\Local\Programs\Python\Python312\python.exe" scripts\shadow_report.py --hours %HOURS% >> state\shadow_report.log 2>&1
exit /b %ERRORLEVEL%
