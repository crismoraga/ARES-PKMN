@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: ARES-PKMN Windows Command Launcher (cmd.exe / PowerShell compatible)
:: ============================================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Locate Python executable
set "PYTHON_EXE="
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "delims=" %%I in ('where python') do (
            if not defined PYTHON_EXE set "PYTHON_EXE=%%I"
        )
    )
)

if not defined PYTHON_EXE (
    echo [ERROR] No se encontro Python 3.12 ni el entorno virtual .venv.
    echo Por favor ejecuta setup.bat primero.
    exit /b 1
)

set "ACTION=%~1"
if "%ACTION%"=="" set "ACTION=status"

:: Collect arguments after %1 (since shift does not affect %*)
set "ARGS="
:collect_args
shift
if "%~1"=="" goto done_args
set "ARGS=%ARGS% %1"
goto collect_args
:done_args

:: Dispatch actions
if /i "%ACTION%"=="setup" (
    "%PYTHON_EXE%" -m scripts.setup_environment %ARGS%
    exit /b !errorlevel!
)

if /i "%ACTION%"=="dashboard" (
    echo Iniciando estacion de monitoreo ARES en http://127.0.0.1:8765/
    "%PYTHON_EXE%" -m scripts.dashboard %ARGS%
    exit /b !errorlevel!
)

if /i "%ACTION%"=="stadium" (
    echo Iniciando estadio adversarial ARES-ADVERSARIAL...
    "%PYTHON_EXE%" -m scripts.run_stadium %ARGS%
    exit /b !errorlevel!
)

if /i "%ACTION%"=="watch" (
    echo Iniciando visor visual de campana...
    "%PYTHON_EXE%" -m scripts.watch_campaign %ARGS%
    exit /b !errorlevel!
)

if /i "%ACTION%"=="status" (
    "%PYTHON_EXE%" -m scripts.run_campaign --status %ARGS%
    exit /b !errorlevel!
)

if /i "%ACTION%"=="run" (
    echo Iniciando nueva campana autonoma...
    "%PYTHON_EXE%" -m scripts.run_campaign --new-game %ARGS%
    exit /b !errorlevel!
)

if /i "%ACTION%"=="resume" (
    echo Reanudando campana autonoma desde el ultimo checkpoint...
    "%PYTHON_EXE%" -m scripts.run_campaign --resume %ARGS%
    exit /b !errorlevel!
)

if /i "%ACTION%"=="train" (
    echo Iniciando entrenamiento neuronal...
    set "TRAIN_ARGS=-m scripts.train_neural_modules"
    if exist "%SCRIPT_DIR%artifacts\checkpoints\neural.pt" (
        set "TRAIN_ARGS=!TRAIN_ARGS! --resume"
    )
    "%PYTHON_EXE%" !TRAIN_ARGS! %ARGS%
    exit /b !errorlevel!
)

if /i "%ACTION%"=="verify" (
    echo Verificando instalacion de ARES...
    "%PYTHON_EXE%" -m scripts.verify_installation %ARGS%
    exit /b !errorlevel!
)

if /i "%ACTION%"=="demo" (
    echo Ejecutando verificacion autonoma de demostracion...
    "%PYTHON_EXE%" -m scripts.run_autonomous_verification_demo %ARGS%
    exit /b !errorlevel!
)

:: Unknown action: pass directly to run_campaign
"%PYTHON_EXE%" -m scripts.run_campaign %ACTION% %ARGS%
exit /b !errorlevel!
