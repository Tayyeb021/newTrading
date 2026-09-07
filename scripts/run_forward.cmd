@echo off
rem The forward paper record: the monthly book on Interactive Brokers.
rem
rem Shadow mode - IB supplies prices, contracts and the account; fills are
rem simulated in-process. No order reaches IB. To send orders to the IB paper
rem account instead, add --live-paper below; that is the operator's decision.
rem
rem Requires IB Gateway running and logged in on port 4002, in this Windows
rem session. IB forces a daily re-login, so if the record shows a gap that is
rem almost always why. Disconnect RDP, do not sign out.
rem
rem Stop it:   create  state\FORWARD_KILL   (the runner halts and stays halted)
rem            or  schtasks /End /TN TradingForwardRecord
rem Watch it:  state\forward_record.log
rem            state\forward_journal.jsonl     every decision, at the moment taken
rem            python scripts\run_forward.py --report

cd /d "%~dp0.."
set PYTHONUNBUFFERED=1
if not exist state mkdir state

echo ==== %DATE% %TIME% forward record start ==== >> state\forward_record.log
"C:\Users\azureuser\AppData\Local\Programs\Python\Python312\python.exe" scripts\run_forward.py --poll 3600 >> state\forward_record.log 2>&1
set RC=%ERRORLEVEL%
echo ==== %DATE% %TIME% forward record end, exit %RC% ==== >> state\forward_record.log
exit /b %RC%
