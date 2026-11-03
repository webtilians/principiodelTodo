# Envía la clave de OpenAI de la demo al servidor. Se pide aquí, oculta, viaja solo dentro de la
# conexión SSH y no se guarda en ningún archivo del PC. Usa la misma llave SSH que vps.ps1 de LAIN.
#   powershell -ExecutionPolicy Bypass -File demo\deploy\vps\clave-openai.ps1
#   powershell -ExecutionPolicy Bypass -File demo\deploy\vps\clave-openai.ps1 -Off   (la quita)
param([switch]$Off, [string]$Server = '178.105.103.4')
$ErrorActionPreference = 'Stop'
$sshKey = Join-Path $env:USERPROFILE '.ssh\lain_vps'
if (-not (Test-Path -LiteralPath $sshKey)) { throw "Falta la llave SSH $sshKey (la crea vps.ps1 -Key de LAIN)." }
$options = @('-i', $sshKey, '-o', 'IdentitiesOnly=yes', '-o', 'StrictHostKeyChecking=accept-new')

if ($Off) {
    & ssh.exe @options "root@$Server" 'infinito-demo-set-key --off'
    exit $LASTEXITCODE
}
$secure = Read-Host 'Pega tu clave de OpenAI (clic derecho o Ctrl+V) y pulsa Enter' -AsSecureString
$pointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
try { $plain = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($pointer) }
finally { [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($pointer) }
# Plain ASCII: a UTF-8 output encoding would prefix a byte order mark.
$OutputEncoding = New-Object Text.ASCIIEncoding
$plain | & ssh.exe @options "root@$Server" 'infinito-demo-set-key'
$plain = $null
if ($LASTEXITCODE -ne 0) { throw 'No se guardó la clave (mira el mensaje de arriba).' }
