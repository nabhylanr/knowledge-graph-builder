<#
.SYNOPSIS
    Registers the chunk-sync poll as a Windows Scheduled Task — the Windows
    equivalent of the launchd agent used on macOS.

.DESCRIPTION
    Runs deploy\chunksync.cmd every N minutes. It only downloads pending chunk
    uploads into chunks_data\; it never starts a KG build.

    This is the safety net. Paired with install_chunksync_listener.ps1 it is what
    catches uploads that arrived while the listener was down — Realtime does not
    replay those. Keep it installed even when the listener is running.

.PARAMETER IntervalMinutes
    How often to poll. Default 10 — fine as a standalone mechanism. With the
    listener installed the poll is pure backstop, so 30-60 is plenty.

.PARAMETER AsSystem
    Register under SYSTEM so the task runs even when nobody is logged on.
    Without this the task only runs while the registering user has a session.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\install_chunksync_task.ps1
    powershell -ExecutionPolicy Bypass -File deploy\install_chunksync_task.ps1 -AsSystem -IntervalMinutes 5

.NOTES
    Remove with:  Unregister-ScheduledTask -TaskName kg-chunksync -Confirm:$false
    Run once now: Start-ScheduledTask   -TaskName kg-chunksync
    Check state:  Get-ScheduledTaskInfo -TaskName kg-chunksync
#>

[CmdletBinding()]
param(
    [int]$IntervalMinutes = 10,
    [switch]$AsSystem
)

$ErrorActionPreference = 'Stop'

$TaskName = 'kg-chunksync'
$repo    = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $PSScriptRoot 'chunksync.cmd'
$python  = Join-Path $repo '.venv\Scripts\python.exe'

if (-not (Test-Path $wrapper)) { throw "Not found: $wrapper" }
if (-not (Test-Path $python))  { throw "No virtualenv at $python — create it and pip install -r requirements.txt first." }
if (-not (Test-Path (Join-Path $repo '.env'))) {
    Write-Warning "No .env in $repo — run_sync.py will exit telling you SUPABASE_URL/SUPABASE_SERVICE_KEY are missing."
}

$action = New-ScheduledTaskAction -Execute $wrapper -WorkingDirectory $repo

# -Once + RepetitionInterval with an effectively unbounded duration is the
# standard way to express "every N minutes, forever" in Task Scheduler.
$trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) `
    -RepetitionInterval (New-TimeSpan -Minutes $IntervalMinutes) `
    -RepetitionDuration ([TimeSpan]::MaxValue)

# StartWhenAvailable is the important one: it runs the poll as soon as possible
# after a start that was missed because the machine was off or asleep. Combined
# with the manifest table, nothing queued is ever lost.
# IgnoreNew stops a slow run from stacking on top of itself.
$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 1)

$principal = if ($AsSystem) {
    New-ScheduledTaskPrincipal -UserId 'SYSTEM' -LogonType ServiceAccount -RunLevel Highest
} else {
    New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive
}

if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
    Write-Host "Replacing existing task '$TaskName'..."
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
}

Register-ScheduledTask -TaskName $TaskName `
    -Action $action -Trigger $trigger -Settings $settings -Principal $principal `
    -Description "Pull pending chunk uploads from Supabase into chunks_data. Downloads only — never starts a KG build." | Out-Null

Write-Host "Registered '$TaskName': every $IntervalMinutes minute(s), running $wrapper"
Write-Host "Logs: $(Join-Path $repo 'chunksync.log')"
Write-Host "Verify now with:  Start-ScheduledTask -TaskName $TaskName"
