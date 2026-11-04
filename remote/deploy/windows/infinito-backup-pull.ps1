<#
Keeps a copy of an INFINITO server's sealed memories on this computer, so they survive losing the server.
Nobody can read them without each person's password. Run it once with -Install to leave it scheduled:

    powershell -ExecutionPolicy Bypass -File infinito-backup-pull.ps1 -Server root@1.2.3.4 -Install

It then runs every day and at each sign-in, brings the copies this computer does not have, checks each
one, and keeps the newest 60. It writes pull.log in the folder, and warns on screen if the newest copy is
over 3 days old, because a backup that stopped working in silence is the one that fails when needed.
#>
param(
    [Parameter(Mandatory = $true)][string]$Server,
    [string]$Key = "$env:USERPROFILE\.ssh\lain_vps",
    [string]$Folder = "$env:USERPROFILE\.infinito\copias-servidor",
    [string]$RemoteFolder = '/var/lib/infinito-remote-backups',
    [string]$Checker = "$env:USERPROFILE\.infinito\venv\Scripts\infinito-remote.exe",
    [int]$Keep = 60,
    [int]$WarnAfterDays = 3,
    [switch]$Install
)
$ErrorActionPreference = 'Stop'
$taskName = 'INFINITO copias de seguridad'

if ($Install) {
    $home_ = "$env:USERPROFILE\.infinito"
    New-Item -ItemType Directory -Force -Path $home_ | Out-Null
    $installed = Join-Path $home_ 'infinito-backup-pull.ps1'
    if ($PSCommandPath -ne $installed) { Copy-Item -Force -LiteralPath $PSCommandPath -Destination $installed }
    $arguments = "-NoProfile -NonInteractive -ExecutionPolicy Bypass -WindowStyle Hidden -File `"$installed`" -Server $Server -Key `"$Key`" -Folder `"$Folder`""
    $action = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $arguments
    $triggers = @((New-ScheduledTaskTrigger -Daily -At '21:00'), (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME))
    $settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries
    Register-ScheduledTask -TaskName $taskName -Action $action -Trigger $triggers -Settings $settings -Force | Out-Null
    Write-Output "Scheduled '$taskName': every day at 21:00 and at each sign-in. Copies go to $Folder."
    return
}

New-Item -ItemType Directory -Force -Path $Folder | Out-Null
$log = Join-Path $Folder 'pull.log'
function Write-Log([string]$Text) {
    Add-Content -LiteralPath $log -Value "$((Get-Date).ToString('yyyy-MM-dd HH:mm')) $Text" -Encoding UTF8
}
$pattern = '^infinito-\d{8}-\d{6}\.tar\.gz$'
$ssh = @('-i', $Key, '-o', 'IdentitiesOnly=yes', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=20')

try {
    $names = @(& ssh.exe @ssh $Server "ls -1 $RemoteFolder" 2>$null)
    if ($LASTEXITCODE -ne 0) { throw "the server did not answer" }
    $brought = 0
    foreach ($name in ($names | Where-Object { $_ -match $pattern })) {
        $target = Join-Path $Folder $name
        if (Test-Path -LiteralPath $target) { continue }
        $partial = "$target.part"
        & scp.exe -q @ssh "${Server}:$RemoteFolder/$name" $partial 2>$null
        if ($LASTEXITCODE -ne 0) { throw "could not bring $name" }
        if (Test-Path -LiteralPath $Checker) {
            & $Checker verify-backup $partial *> $null
            if ($LASTEXITCODE -ne 0) {
                Remove-Item -LiteralPath $partial -Force
                throw "$name came damaged"
            }
        }
        Move-Item -LiteralPath $partial -Destination $target -Force
        $brought++
    }
    $copies = @(Get-ChildItem -LiteralPath $Folder -File | Where-Object { $_.Name -match $pattern } | Sort-Object Name)
    if ($copies.Count -gt $Keep) { $copies[0..($copies.Count - $Keep - 1)] | Remove-Item -Force }
    Write-Log "ok: $brought new, $([Math]::Min($copies.Count, $Keep)) kept"
} catch {
    Write-Log "FAILED: $_"
}

$newest = Get-ChildItem -LiteralPath $Folder -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match $pattern } | Sort-Object Name | Select-Object -Last 1
$made = if ($newest) { [datetime]::ParseExact($newest.Name.Substring(9, 15), 'yyyyMMdd-HHmmss', $null) } else { $null }
if (-not $made -or ((Get-Date).ToUniversalTime() - $made).TotalDays -gt $WarnAfterDays) {
    $warning = "INFINITO: tu copia de seguridad mas reciente tiene mas de $WarnAfterDays dias. Mira $log"
    Write-Log $warning
    try { & msg.exe $env:USERNAME $warning } catch { }
}
