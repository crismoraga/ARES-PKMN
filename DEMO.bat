@echo off
setlocal enabledelayedexpansion
title ARES-PKMN :: Estacion de Monitoreo y Demostracion en Convencion

set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

:: Locate Python
set "PYTHON_EXE="
if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
) else (
    echo [INFO] No se encontro entorno virtual .venv. Ejecutando configuracion inicial...
    call "%SCRIPT_DIR%setup.bat"
    if exist "%SCRIPT_DIR%.venv\Scripts\python.exe" (
        set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
    ) else (
        echo [ERROR] No se pudo preparar el entorno. Abortando.
        pause
        exit /b 1
    )
)

:MENU
cls
echo ============================================================================
echo   ____  ____  _____ ____             ____  _  ____  __ _  _ 
echo  / _ \^|  _ \^| ____/ ___^|           ^|  _ \^| ^|/ /  \/  ^| \ ^| ^|
echo / /_\ \ ^|_) ^|  _^| \___ \  - - - -  ^| ^|_) ^| ' /^| ^|\/^| ^|  \^| ^|
echo /  _   ^|  _ ^<^| ^|___ ___) ^|          ^|  __/^| . \^| ^|  ^| ^| ^|\  ^|
echo /_/   \_\_^| \_\_____^|____/           ^|_^|   ^|_^|\_\_^|  ^|_^|_^| \_^|
echo.
echo   SISTEMA AUTONOMO NEURO-SIMBOLICO DE AGENTES POKEMON GEN-3
echo   Estacion de Monitoreo, Analisis en Tiempo Real y Estadio Coevolutivo
echo ============================================================================
echo.
echo   La estacion de monitoreo visual estara disponible en:
echo     - Panel de Campanas y Metricas: http://127.0.0.1:8765/
echo     - Estadio Adversarial en Vivo:   http://127.0.0.1:8765/stadium
echo.
echo ----------------------------------------------------------------------------
echo   SELECCIONA MODO DE DEMOSTRACION:
echo ----------------------------------------------------------------------------
echo   [1] Demostracion Completa de Convencion (Liga en Vivo + Dashboard Web) [DEFAULT]
echo   [2] Liga Rapida de Prueba (4 Generaciones / Duelos y Visualizacion)
echo   [3] Abrir solo Estacion de Monitoreo Web (http://127.0.0.1:8765/)
echo   [4] Iniciar Campana Autonoma en Emulador Real (FireRed y Emerald)
echo   [5] Reanudar Campana desde Ultimo Checkpoint
echo   [6] Abrir Visor Grafico Nativo en Ventana (Pygame)
echo   [7] Verificacion Integral del Sistema (Tests y Verificador Z3)
echo   [0] Salir
echo.
set "CHOICE="
set /p "CHOICE=Opcion [1-7, o Enter para Modo 1]: "
if "%CHOICE%"=="" set "CHOICE=1"

if "%CHOICE%"=="1" goto MODO_COMPLETO
if "%CHOICE%"=="2" goto MODO_RAPIDO
if "%CHOICE%"=="3" goto MODO_DASHBOARD
if "%CHOICE%"=="4" goto MODO_CAMPANA
if "%CHOICE%"=="5" goto MODO_RESUME
if "%CHOICE%"=="6" goto MODO_WATCH
if "%CHOICE%"=="7" goto MODO_VERIFY
if "%CHOICE%"=="0" exit /b 0

echo Opcion no valida.
timeout /t 2 >nul
goto MENU

:MODO_COMPLETO
cls
echo [INFO] Iniciando Estacion de Monitoreo Web en segundo plano...
start "ARES Dashboard Server" /min "%PYTHON_EXE%" -m scripts.dashboard --no-browser
timeout /t 2 >nul
echo [INFO] Abriendo vista del Estadio en el navegador...
start http://127.0.0.1:8765/stadium
echo.
echo ============================================================================
echo   ARES-ADVERSARIAL: Coevolucion en Vivo (ROJO vs AZUL)
echo   (Los combates, lecciones contrafactuales y duelos se muestran en el navegador)
echo   Presiona Ctrl+C en cualquier momento para detener la liga.
echo ============================================================================
echo.
"%PYTHON_EXE%" -m scripts.run_stadium
pause
goto MENU

:MODO_RAPIDO
cls
echo [INFO] Iniciando Estacion de Monitoreo Web en segundo plano...
start "ARES Dashboard Server" /min "%PYTHON_EXE%" -m scripts.dashboard --no-browser
timeout /t 2 >nul
start http://127.0.0.1:8765/stadium
echo [INFO] Ejecutando 4 generaciones de combate coevolutivo...
"%PYTHON_EXE%" -m scripts.run_stadium --smoke
echo.
echo [INFO] Demostracion rapida completada. Revisa los resultados en http://127.0.0.1:8765/stadium
pause
goto MENU

:MODO_DASHBOARD
cls
echo [INFO] Iniciando Estacion de Monitoreo Web...
echo Abre tu navegador en: http://127.0.0.1:8765/
echo Presiona Ctrl+C para detener el servidor.
"%PYTHON_EXE%" -m scripts.dashboard
goto MENU

:MODO_CAMPANA
cls
echo [INFO] Iniciando Estacion de Monitoreo Web en segundo plano...
start "ARES Dashboard Server" /min "%PYTHON_EXE%" -m scripts.dashboard --no-browser
timeout /t 2 >nul
start http://127.0.0.1:8765/
echo [INFO] Iniciando campanas de juego autonomo en emulador nativo...
"%PYTHON_EXE%" -m scripts.run_campaign --new-game
pause
goto MENU

:MODO_RESUME
cls
echo [INFO] Iniciando Estacion de Monitoreo Web en segundo plano...
start "ARES Dashboard Server" /min "%PYTHON_EXE%" -m scripts.dashboard --no-browser
timeout /t 2 >nul
start http://127.0.0.1:8765/
echo [INFO] Reanudando campanas desde el checkpoint mas avanzado...
"%PYTHON_EXE%" -m scripts.run_campaign --resume
pause
goto MENU

:MODO_WATCH
cls
echo [INFO] Iniciando visor grafico pygame...
"%PYTHON_EXE%" -m scripts.watch_campaign
pause
goto MENU

:MODO_VERIFY
cls
echo [INFO] Verificando modulos del sistema, equivalencia Z3 y tablas Gen-3...
"%PYTHON_EXE%" -m scripts.verify_installation
pause
goto MENU
