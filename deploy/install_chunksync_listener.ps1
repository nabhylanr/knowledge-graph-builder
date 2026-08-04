<#
.SYNOPSIS
    Registers the Realtime chunk-sync listener as a long-lived Scheduled Task.

.DESCRIPTION
    This is the FAST half of the hand-off: it holds a WebSocket to Supabase and
    pulls an upload within about a second of it landing.

    It is not the delivery guarantee. Realtime never replays an event fired while
    this process was down, so install_chunksync_task.ps1 (the periodic poll) must
    stay installed alongside it as the safety net. Losing this daemon costs
    latency, never data.

    Task Scheduler is used rather than a real service wrapper (NSSM/winsw) so
    there is nothing extra to install: an at-startup trigger with no execution
    time limit and restart-on-failure behaves the same for this purpose.

.PARAMETER AsSystem
    Register under SYSTEM with an at-startup trigger, so it runs with no user
    logged on. Without it the task triggers at logon for the current user.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File deploy\install_chunksync_listener.ps1
    powershell -ExecutionPolicy Bypass -File deploy\install_chunksync_listener.ps1 -AsSystem

.NOTES
    Start now:   Start-ScheduledTask       -TaskName kg-chunksync-listener
    Check:       Get-ScheduledTaskInfo     -TaskName kg-chunksync-listener
    Remove:      Unregister-ScheduledTask  -TaskName kg-chunksync-listener -Confirm:$false
    Logs:        chunksync-listener.log in the repo root
#>

[CmdletBinding()]
param(
    [switch]$AsSystem
)

$ErrorActionPreference = 'Stop'

$TaskName = 'kg-chunksync-listener'
$repo    = Split-Path -Parent $PSScriptRoot
$wrapper = Join-Path $PSScriptRoot 'chunksync_listener.cmd'
$python  = Join-Path $repo '.venv\Scripts\python.exe'

if (-not (Test-Path $wrapper)) { throw "Not found: $wrapper" }
if (-not (Test-Path $python))  { throw "No virtualenv at $python — create it and pip install -r requirements.txt first." }

if (-not (Get-ScheduledTask -TaskName 'kg-chunksync' -ErrorAction SilentlyContinue)) {
    Write-Warning "The periodic poll task 'kg-chunksync' is not installed. The listener alone loses any upload that arrives while it is down — install install_chunksync_task.ps1 as well."
}

$action = New-ScheduledTaskAction -Execute $wrapper -WorkingDirectory $repo

$trigger = if ($AsSystem) { New-ScheduledTaskTrigger -AtStartup } else { New-ScheduledTaskTrigger -AtLogOn }

# ExecutionTimeLimit 0 = never kill it; this task is meant to run forever.
# RestartCount/Interval bring it back if the process dies.
$settings = New-ScheduledTaskSettingsSet `
    -MultipleInstances IgnoreNew `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -RestartCount 999 `
    -RestartInterval (New-TimeSpan -Minutes 1) `
    -ExecutionTimeLimit (New-TimeSpan -Seconds 0)

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
    -Description "Realtime listener: pulls chunk uploads seconds after they land. Downloads only — never starts a KG build. Paired with the kg-chunksync poll." | Out-Null

Write-Host "Registered '$TaskName' ($(if ($AsSystem) {'at startup, as SYSTEM'} else {'at logon'}))."
Write-Host "Logs: $(Join-Path $repo 'chunksync-listener.log')"
Write-Host "Start it now with:  Start-ScheduledTask -TaskName $TaskName"
