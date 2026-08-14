# Scheduling the daily pipeline (Windows Task Scheduler)

`etl/pipeline.py` is meant to run once per day, after US market close, so the
day's bar is final. It is not scheduled automatically — set this up yourself
with Task Scheduler.

## Option A: one-line `schtasks` command

Run this from an elevated PowerShell/cmd prompt, adjusting the venv path if
your install differs (market close is 4:00pm ET / 4:30pm covers most
settlement delay; adjust `/st` for your local timezone):

```
schtasks /create /tn "PaperTradingETL" /tr "C:\Users\Leigh\paper-trading-bot\venv\Scripts\python.exe C:\Users\Leigh\paper-trading-bot\etl\pipeline.py" /sc daily /st 16:30 /sd 01/01/2026
```

## Option B: Task Scheduler GUI

1. Open Task Scheduler -> Create Task.
2. **General**: name it `PaperTradingETL`, run whether user is logged on or not.
3. **Triggers**: New -> Daily, start time after market close (e.g. 4:30 PM local).
4. **Actions**: New -> Start a program:
   - Program/script: `C:\Users\Leigh\paper-trading-bot\venv\Scripts\python.exe`
   - Arguments: `etl\pipeline.py`
   - Start in: `C:\Users\Leigh\paper-trading-bot`
5. **Conditions**: uncheck "Start the task only if the computer is on AC power" if this runs on a laptop.

## Verifying it ran

Check `logs/pipeline_<date>.log` for the day's run, or check Task Scheduler's
History tab for the task.

## Removing the task

```
schtasks /delete /tn "PaperTradingETL" /f
```
