#!/usr/bin/env bash
# ============================================================================
# ARES-PKMN Linux / macOS Convention Demonstration Launcher
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Locate Python
PYTHON_EXE=""
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/Scripts/python.exe"
else
    echo "[INFO] No se encontró entorno virtual .venv. Ejecutando ./setup.sh..."
    bash "$SCRIPT_DIR/setup.sh"
    if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
        PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
    else
        echo "[ERROR] No se pudo inicializar el entorno. Abortando."
        exit 1
    fi
fi

open_browser() {
    URL="$1"
    if command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$URL" >/dev/null 2>&1 &
    elif command -v open >/dev/null 2>&1; then
        open "$URL" >/dev/null 2>&1 &
    elif command -v sensible-browser >/dev/null 2>&1; then
        sensible-browser "$URL" >/dev/null 2>&1 &
    fi
}

start_dashboard_background() {
    echo "[INFO] Iniciando Estación de Monitoreo Web en segundo plano..."
    "$PYTHON_EXE" -m scripts.dashboard --no-browser >/dev/null 2>&1 &
    DASHBOARD_PID=$!
    trap 'kill $DASHBOARD_PID 2>/dev/null || true' EXIT
    sleep 1.5
}

while true; do
    clear || true
    echo "============================================================================"
    echo "   ____  ____  _____ ____             ____  _  ____  __ _  _ "
    echo "  / _ \|  _ \| ____/ ___|           |  _ \| |/ /  \/  | \ | |"
    echo " / /_\ \ |_) |  _| \___ \  - - - -  | |_) | ' /| |\/| |  \| |"
    echo "/  _   |  _ <| |___ ___) |          |  __/| . \| |  | | |\  |"
    echo "/_/   \_\_| \_\_____|____/           |_|   |_|\_\_|  |_|_| \_|"
    echo ""
    echo "  SISTEMA AUTÓNOMO NEURO-SIMBÓLICO DE AGENTES POKÉMON GEN-3"
    echo "  Estación de Monitoreo, Análisis en Tiempo Real y Estadio Coevolutivo"
    echo "============================================================================"
    echo ""
    echo "  La estación de monitoreo visual estará disponible en:"
    echo "    - Panel de Campañas y Métricas: http://127.0.0.1:8765/"
    echo "    - Estadio Adversarial en Vivo:   http://127.0.0.1:8765/stadium"
    echo ""
    echo "----------------------------------------------------------------------------"
    echo "  SELECCIONA MODO DE DEMOSTRACIÓN:"
    echo "----------------------------------------------------------------------------"
    echo "  [1] Demostración Completa de Convención (Liga en Vivo + Dashboard Web) [DEFAULT]"
    echo "  [2] Liga Rápida de Prueba (4 Generaciones / Duelos y Visualización)"
    echo "  [3] Abrir solo Estación de Monitoreo Web (http://127.0.0.1:8765/)"
    echo "  [4] Iniciar Campaña Autónoma en Emulador (FireRed y Emerald)"
    echo "  [5] Reanudar Campaña desde Último Checkpoint"
    echo "  [6] Abrir Visor Gráfico Nativo en Ventana (Pygame)"
    echo "  [7] Verificación Integral del Sistema (Tests y Verificador Z3)"
    echo "  [0] Salir"
    echo ""
    read -r -p "Opción [1-7, o Enter para Modo 1]: " CHOICE
    CHOICE="${CHOICE:-1}"

    case "$CHOICE" in
        1)
            start_dashboard_background
            open_browser "http://127.0.0.1:8765/stadium"
            echo ""
            echo "============================================================================"
            echo "  ARES-ADVERSARIAL: Coevolución en Vivo (ROJO vs AZUL)"
            echo "  (Combates, lecciones contrafactuales y duelos visibles en el navegador)"
            echo "  Presiona Ctrl+C en cualquier momento para detener la liga."
            echo "============================================================================"
            echo ""
            "$PYTHON_EXE" -m scripts.run_stadium || true
            read -r -p "Presiona Enter para continuar..."
            ;;
        2)
            start_dashboard_background
            open_browser "http://127.0.0.1:8765/stadium"
            echo "[INFO] Ejecutando 4 generaciones de combate coevolutivo..."
            "$PYTHON_EXE" -m scripts.run_stadium --smoke
            echo "[INFO] Demostración rápida completada. Revisa los resultados en http://127.0.0.1:8765/stadium"
            read -r -p "Presiona Enter para continuar..."
            ;;
        3)
            echo "[INFO] Iniciando Estación de Monitoreo Web..."
            open_browser "http://127.0.0.1:8765/"
            "$PYTHON_EXE" -m scripts.dashboard
            ;;
        4)
            start_dashboard_background
            open_browser "http://127.0.0.1:8765/"
            echo "[INFO] Iniciando campañas de juego autónomo..."
            "$PYTHON_EXE" -m scripts.run_campaign --new-game
            read -r -p "Presiona Enter para continuar..."
            ;;
        5)
            start_dashboard_background
            open_browser "http://127.0.0.1:8765/"
            echo "[INFO] Reanudando campañas desde el checkpoint más avanzado..."
            "$PYTHON_EXE" -m scripts.run_campaign --resume
            read -r -p "Presiona Enter para continuar..."
            ;;
        6)
            echo "[INFO] Iniciando visor gráfico pygame..."
            "$PYTHON_EXE" -m scripts.watch_campaign
            ;;
        7)
            echo "[INFO] Verificando módulos del sistema, equivalencia Z3 y tablas Gen-3..."
            "$PYTHON_EXE" -m scripts.verify_installation
            read -r -p "Presiona Enter para continuar..."
            ;;
        0)
            echo "Saliendo."
            exit 0
            ;;
        *)
            echo "Opción no válida."
            sleep 1
            ;;
    esac
done
