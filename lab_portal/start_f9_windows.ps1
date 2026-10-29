<#
INFINITO F9 — instala y ejecuta el trabajador de cálculo en Windows.
Ejecutar en PowerShell desde la raíz del repositorio, en la rama exacta
que ha desplegado el VPS. No guarda ni imprime el token del trabajador.
Ejemplo:
  powershell -ExecutionPolicy Bypass -File lab_portal/start_f9_windows.ps1
#>
param(
  [string]$Server = "https://lab.178-105-103-4.sslip.io",
  [switch]$InstallOnly,
  [switch]$Once,
  [switch]$SkipInstall
)

$ErrorActionPreference = "Stop"
$repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $repo
Write-Host "INFINITO F9 · PC Windows · CPU / 32 GB RAM" -ForegroundColor Cyan
Write-Host "Carpeta: $repo"
$head = (& git rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0 -or $head.Length -ne 40) {
  throw "No hay un checkout Git válido. Clona la rama experiment/f9-windows-remote-worker."
}
Write-Host ("Revisión Git: " + $head.Substring(0, 12))
$python = Join-Path $repo ".venv-f9\Scripts\python.exe"
if (!(Test-Path $python)) {
  Write-Host "Creando Python 3.11 aislado ..."
  & py -3.11 -m venv (Join-Path $repo ".venv-f9")
  if ($LASTEXITCODE -ne 0) { throw "Se necesita Python 3.11 y el comando py." }
}
if (!$SkipInstall) {
  Write-Host "Instalando PyTorch CPU y MiniGrid de forma aislada..."
  & $python -m pip install --upgrade pip
  if ($LASTEXITCODE -ne 0) { throw "pip upgrade falló" }
  & $python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  if ($LASTEXITCODE -ne 0) { throw "No se pudo instalar PyTorch CPU" }
  & $python -m pip install -r (Join-Path $repo "observatory\requirements_f9.txt")
  if ($LASTEXITCODE -ne 0) { throw "Instalación F9 fallida" }
}
& $python -c "import torch, minigrid, stable_baselines3, sb3_contrib; print('F9 OK: Torch',torch.__version__, 'MiniGrid',minigrid.__version__)"
if ($LASTEXITCODE -ne 0) { throw "Faltan dependencias. Ejecuta el script sin -SkipInstall." }
if ($InstallOnly) {
  Write-Host "Dependencias instaladas. No se inició el worker."
  exit 0
}
if ([string]::IsNullOrWhiteSpace($env:INFINITO_WORKER_TOKEN)) {
  $secure = Read-Host "Pega la clave privada del trabajador (no la clave del navegador)" -AsSecureString
  $bstr = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
  try {
    $env:INFINITO_WORKER_TOKEN = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($bstr)
  } finally {
    [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($bstr)
  }
}
if ($env:INFINITO_WORKER_TOKEN.Length -lt 32) {
  throw "La clave debe tener al menos 32 caracteres y coincidir con LAB_WORKER_TOKEN del VPS."
}
Write-Host "Buscando experimentos desde $Server (saliente por HTTPS; Ctrl+C para terminar)"
$argsWorker = @("-m", "lab_portal.remote_worker", "--server", $Server)
if ($Once) { $argsWorker += "--once" }
try {
  & $python @argsWorker
} finally {
  Remove-Item Env:INFINITO_WORKER_TOKEN -ErrorAction SilentlyContinue
}
