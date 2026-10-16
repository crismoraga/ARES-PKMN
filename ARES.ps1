[CmdletBinding()]
param(
    [ValidateSet('run', 'resume', 'status', 'train', 'verify', 'setup', 'demo', 'watch', 'dashboard')]
    [string]$Action = 'status',
    [ValidateSet('firered', 'emerald', 'both')]
    [string]$Game = 'both',
    [ValidateSet('native', 'synthetic')]
    [string]$Backend = 'native',
    [ValidateRange(1, 2147483647)]
    [int]$MaxFrames,
    [switch]$Background,
    [switch]$Quick,
    [string]$SnapshotPath,
    [ValidateRange(1, 65535)]
    [int]$Port = 8765
)

$ErrorActionPreference = 'Stop'
$aresRoot = $PSScriptRoot
$aresPython = Join-Path $aresRoot '.venv\Scripts\python.exe'
if ($Action -eq 'setup' -and -not (Test-Path -LiteralPath $aresPython)) {
    $aresPython = (Get-Command python -ErrorAction Stop).Source
}
if (-not (Test-Path -LiteralPath $aresPython)) {
    throw 'Primero ejecuta .\ARES.ps1 setup con Python 3.12 y uv disponibles.'
}

$aresArguments = switch ($Action) {
    'setup' { @('-m', 'scripts.setup_environment') }
    'train' {
        $aresTrainArgs = @('-m', 'scripts.train_neural_modules')
        if (Test-Path (Join-Path $aresRoot 'artifacts\checkpoints\neural.pt')) {
            $aresTrainArgs += '--resume'
        }
        $aresTrainArgs
    }
    'verify' {
        if ($Quick) { @('-m', 'scripts.verify_installation', '--quick') }
        else { @('-m', 'scripts.verify_installation') }
    }
    'demo' { @('-m', 'scripts.run_autonomous_verification_demo') }
    'dashboard' {
        $aresDashboardArgs = @('-m', 'scripts.dashboard', '--backend', $Backend, '--port', $Port.ToString())
        if ($Background) { $aresDashboardArgs += '--no-browser' }
        $aresDashboardArgs
    }
    'watch' {
        $aresWatchArgs = @('-m', 'scripts.watch_campaign', '--backend', $Backend)
        if ($PSBoundParameters.ContainsKey('SnapshotPath')) {
            $aresWatchArgs += @('--snapshot', $SnapshotPath)
        }
        $aresWatchArgs
    }
    default {
        $aresCommandArgs = @('-m', 'scripts.run_campaign', '--backend', $Backend, '--game', $Game)
        if ($Action -eq 'status') { $aresCommandArgs += '--status' }
        elseif ($Action -eq 'resume') { $aresCommandArgs += '--resume' }
        else { $aresCommandArgs += '--new-game' }
        if ($PSBoundParameters.ContainsKey('MaxFrames')) { $aresCommandArgs += @('--max-frames', $MaxFrames.ToString()) }
        $aresCommandArgs
    }
}

if ($Background) {
    if ($Action -eq 'watch') {
        throw 'El visor necesita una ventana visible; ejecútalo en primer plano con .\ARES.ps1 watch.'
    }
    if ($Action -notin @('run', 'resume', 'train', 'verify', 'demo', 'dashboard')) {
        throw '-Background se admite para run, resume, train, verify, demo y dashboard.'
    }
    $aresLogDirectory = Join-Path $aresRoot 'artifacts\processes'
    New-Item -ItemType Directory -Force -Path $aresLogDirectory | Out-Null
    $aresTimestamp = Get-Date -Format 'yyyyMMdd-HHmmss-fff'
    $aresStdout = Join-Path $aresLogDirectory "$Action-$Game-$aresTimestamp.stdout.log"
    $aresStderr = Join-Path $aresLogDirectory "$Action-$Game-$aresTimestamp.stderr.log"
    $aresProcess = Start-Process -FilePath $aresPython -ArgumentList $aresArguments -WorkingDirectory $aresRoot `
        -WindowStyle Hidden -RedirectStandardOutput $aresStdout -RedirectStandardError $aresStderr -PassThru
    Start-Sleep -Milliseconds 700
    $aresLive = Get-CimInstance Win32_Process -Filter "ProcessId = $($aresProcess.Id)"
    if ($null -eq $aresLive -or $aresLive.ExecutablePath -ne $aresPython) {
        throw "El proceso terminó durante el inicio. Revisa $aresStderr"
    }
    $aresState = @{ pid = $aresProcess.Id; action = $Action; game = $Game; backend = $Backend;
                   stdout = $aresStdout; stderr = $aresStderr; executable = $aresLive.ExecutablePath }
    if ($Action -eq 'dashboard') {
        $aresState.url = "http://127.0.0.1:$Port/"
        Set-Content -LiteralPath (Join-Path $aresLogDirectory 'dashboard.json') -Value ($aresState | ConvertTo-Json) -Encoding utf8
    } else {
        $aresState | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $aresLogDirectory 'latest.json') -Encoding utf8
    }
    $aresState | ConvertTo-Json
} else {
    Push-Location $aresRoot
    try {
        & $aresPython @aresArguments
        exit $LASTEXITCODE
    } finally {
        Pop-Location
    }
}
