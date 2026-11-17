@echo off
setlocal enabledelayedexpansion

:: ============================================================================
:: ARES-PKMN Windows Automated Setup
:: ============================================================================

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

echo ============================================================================
echo   ARES-PKMN: Instalacion y Configuracion Automatizada
echo ============================================================================
echo.

:: Step 1: Detect Python
echo [1/5] Detectando Python 3.12...
set "SYS_PYTHON="
py -3.12 -c "import sys; print(sys.executable)" >nul 2>&1
if !errorlevel! equ 0 (
    for /f "delims=" %%I in ('py -3.12 -c "import sys; print(sys.executable)"') do set "SYS_PYTHON=%%I"
) else (
    python -c "import sys; assert sys.version_info[:2] == (3, 12)" >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "delims=" %%I in ('where python') do (
            if not defined SYS_PYTHON set "SYS_PYTHON=%%I"
        )
    )
)

if not defined SYS_PYTHON (
    echo [AVISO] Python 3.12 exacto no encontrado directamente. Probando 'python'...
    where python >nul 2>&1
    if !errorlevel! equ 0 (
        for /f "delims=" %%I in ('where python') do (
            if not defined SYS_PYTHON set "SYS_PYTHON=%%I"
        )
    )
)

if not defined SYS_PYTHON (
    echo [ERROR] No se encontro Python en el sistema.
    echo Instala Python 3.12 desde https://www.python.org/downloads/ o via winget:
    echo   winget install Python.Python.3.12
    pause
    exit /b 1
)

echo   Encontrado: !SYS_PYTHON!
echo.

:: Step 2: Check or install uv
echo [2/5] Comprobando herramienta uv...
set "UV_EXE="
where uv >nul 2>&1
if !errorlevel! equ 0 (
    set "UV_EXE=uv"
    echo   uv encontrado en el PATH.
) else (
    echo   uv no encontrado; intentando instalar o usar pip...
    "!SYS_PYTHON!" -m pip install --upgrade uv >nul 2>&1
    where uv >nul 2>&1
    if !errorlevel! equ 0 set "UV_EXE=uv"
)
echo.

:: Step 3: Create virtual environment
echo [3/5] Configurando entorno virtual (.venv)...
set "VENV_DIR=%SCRIPT_DIR%.venv"
set "VENV_PYTHON=%VENV_DIR%\Scripts\python.exe"

if not exist "%VENV_PYTHON%" (
    if defined UV_EXE (
        echo   Creando .venv con uv...
        !UV_EXE! venv --python "!SYS_PYTHON!" "%VENV_DIR%"
    ) else (
        echo   Creando .venv con venv estandar...
        "!SYS_PYTHON!" -m venv "%VENV_DIR%"
    )
) else (
    echo   Entorno virtual ya existente en .venv.
)
echo.

:: Step 4: Install dependencies
echo [4/5] Instalando dependencias del proyecto...
if defined UV_EXE (
    echo   Sincronizando paquetes bloqueados con uv...
    !UV_EXE! pip sync --python "%VENV_PYTHON%" --extra-index-url https://download.pytorch.org/whl/cpu requirements.lock
    !UV_EXE! pip install --python "%VENV_PYTHON%" --no-deps --editable "%SCRIPT_DIR%"
) else (
    echo   Instalando dependencias con pip...
    "%VENV_PYTHON%" -m pip install --upgrade pip
    "%VENV_PYTHON%" -m pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock
    "%VENV_PYTHON%" -m pip install --no-deps -e "%SCRIPT_DIR%"
)
echo.

:: Step 5: Verification
echo [5/5] Verificando instalacion...
"%VENV_PYTHON%" -m scripts.verify_installation --quick
if !errorlevel! equ 0 (
    echo.
    echo ============================================================================
    echo   Instalacion completada con exito.
    echo   Para iniciar la demostracion de convencion, ejecuta: DEMO.bat
    echo   O para ver la estacion de monitoreo web: ARES.bat dashboard
    echo ============================================================================
) else (
    echo.
    echo [ADVERTENCIA] La verificacion reporto algunas observaciones.
    echo Revisa el registro arriba. Los datos estandar del Stadium estan listos.
)

pause
