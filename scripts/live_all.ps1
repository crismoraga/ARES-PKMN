# ARES en vivo: arranca el panel (scripts.live_dashboard) en segundo plano, abre el navegador e imprime
# los comandos para lanzar las partidas que publican en artifacts/live/<run_id>/.
#
#   powershell -ExecutionPolicy Bypass -File scripts\live_all.ps1            # panel + navegador
#   powershell -ExecutionPolicy Bypass -File scripts\live_all.ps1 -Demo      # + ejecuciones simuladas
#   powershell -ExecutionPolicy Bypass -File scripts\live_all.ps1 -NoBrowser -Port 8771
param(
    [int]$Port = 8770,
    [switch]$NoBrowser,
    [switch]$Demo
)
$ErrorActionPreference = 'Stop'
$Root = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $Root '.venv\Scripts\python.exe'
if (-not (Test-Path $Python)) { $Python = 'python' }
$Url = "http://127.0.0.1:$Port"
$LogDir = Join-Path $Root 'artifacts\dashboard'
New-Item -ItemType Directory -Force $LogDir | Out-Null

function Get-DashboardHealth {
    try { return Invoke-RestMethod -Uri "$Url/api/health" -TimeoutSec 2 } catch { return $null }
}

$health = Get-DashboardHealth
if ($health) {
    Write-Host "El panel ya está en marcha en $Url (PID $($health.pid))."
} else {
    $log = Join-Path $LogDir 'server.log'
    # No -Redirect*: redirecting makes the child inherit this console's handles (a caller piping our output
    # would then wait for the server to exit). The server tees stdout/stderr into --log-file itself.
    Start-Process -FilePath $Python -WorkingDirectory $Root -WindowStyle Hidden `
        -ArgumentList @('-m', 'scripts.live_dashboard', '--no-browser', '--port', "$Port", '--log-file', $log) | Out-Null
    for ($i = 0; $i -lt 40 -and -not $health; $i++) { Start-Sleep -Milliseconds 250; $health = Get-DashboardHealth }
    if (-not $health) { Write-Error "El panel no respondió en $Url; revisa $log" }
    Write-Host "Panel en marcha: $Url (PID $($health.pid), registro $log)"
}

if ($Demo) {
    Start-Process -FilePath $Python -WorkingDirectory $Root -WindowStyle Hidden `
        -ArgumentList @('-m', 'scripts.live_demo_publisher', '--clean') | Out-Null
    Write-Host 'Publicador de ejecuciones SIMULADAS en marcha (desmarca «simulados» en el panel para ocultarlas).'
}

if (-not $NoBrowser) { Start-Process $Url }

Write-Host ''
Write-Host 'Lanza las partidas en otras terminales (desde la raíz del repositorio):'
Write-Host '  Campaña Rojo Fuego : python -m scripts.run_campaign --game firered --resume'
Write-Host '  Campaña Esmeralda  : python -m scripts.run_campaign --game emerald --resume'
Write-Host '  Agente universal   : python -m scripts.ares_u_universal --game rojo   (Gen 1-5 vía GAL; ver --help)'
Write-Host '  Liga de rivales    : python -m scripts.ares_u_league train   (después: python -m scripts.ares_u_league report)'
Write-Host '  Datos simulados    : python -m scripts.live_demo_publisher --clean'
Write-Host ''
Write-Host "Detener el panel: Stop-Process -Id $($health.pid)"
