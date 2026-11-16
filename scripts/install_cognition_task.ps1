# Registers "INFINITO Motor cognitivo": every hour, `python -m cognition tick` in this repository.
# The tick does the day's review once it is due (and catches up a day the computer was off), or an observer check.
# It has no end date: pause it with `python -m cognition pause`, or remove the task with
#   Unregister-ScheduledTask -TaskName "INFINITO Motor cognitivo" -Confirm:$false
# With -NightHour (say 4, the first of the night_hours in config.json) a second task, "INFINITO Motor cognitivo
# (noche)", starts the night at that hour and wakes the computer from sleep for it (not from off or hibernation).
param([string]$Repo = (Resolve-Path "$PSScriptRoot\..").Path, [int]$NightHour = -1)

$python = (Get-Command pythonw.exe -ErrorAction SilentlyContinue).Source
if (-not $python) { $python = (Get-Command python.exe).Source }
$action = New-ScheduledTaskAction -Execute $python -Argument "-m cognition tick" -WorkingDirectory $Repo
# A daily trigger that repeats every hour for a day: it starts again each day, so it never runs out.
$trigger = New-ScheduledTaskTrigger -Daily -At "00:05"
$trigger.Repetition = (New-ScheduledTaskTrigger -Once -At "00:05" -RepetitionInterval (New-TimeSpan -Hours 1) `
    -RepetitionDuration (New-TimeSpan -Days 1)).Repetition
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName "INFINITO Motor cognitivo" -Action $action -Trigger $trigger -Settings $settings `
    -Description "Motor cognitivo de INFINITO: revisión diaria y observador cada hora (python -m cognition tick)." `
    -Force | Out-Null
Write-Output "Registered: INFINITO Motor cognitivo, every hour, in $Repo"
if ($NightHour -ge 0) {
    $night = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -WakeToRun -ExecutionTimeLimit (New-TimeSpan -Hours 6) -MultipleInstances IgnoreNew
    Register-ScheduledTask -TaskName "INFINITO Motor cognitivo (noche)" -Action $action `
        -Trigger (New-ScheduledTaskTrigger -Daily -At ("{0:00}:00" -f $NightHour)) -Settings $night `
        -Description "Motor cognitivo de INFINITO: empieza la noche (revisión y repaso) y despierta el equipo." `
        -Force | Out-Null
    Write-Output "Registered: INFINITO Motor cognitivo (noche), daily at ${NightHour}:00, waking the computer"
}
