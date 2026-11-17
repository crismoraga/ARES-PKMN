#!/usr/bin/env bash
# ============================================================================
# ARES-PKMN Linux / macOS Command Launcher
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Locate Python in .venv or system
PYTHON_EXE=""
if [ -f "$SCRIPT_DIR/.venv/bin/python" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/bin/python"
elif [ -f "$SCRIPT_DIR/.venv/Scripts/python.exe" ]; then
    PYTHON_EXE="$SCRIPT_DIR/.venv/Scripts/python.exe"
elif command -v python3.12 >/dev/null 2>&1; then
    PYTHON_EXE="python3.12"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_EXE="python3"
elif command -v python >/dev/null 2>&1; then
    PYTHON_EXE="python"
fi

if [ -z "$PYTHON_EXE" ]; then
    echo "[ERROR] No se encontró Python en el sistema ni en .venv."
    echo "Ejecuta ./setup.sh primero."
    exit 1
fi

ACTION="${1:-status}"
shift || true

case "$ACTION" in
    setup)
        exec "$PYTHON_EXE" -m scripts.setup_environment "$@"
        ;;
    dashboard)
        echo "Iniciando estación de monitoreo ARES en http://127.0.0.1:8765/"
        exec "$PYTHON_EXE" -m scripts.dashboard "$@"
        ;;
    stadium)
        echo "Iniciando estadio adversarial ARES-ADVERSARIAL..."
        exec "$PYTHON_EXE" -m scripts.run_stadium "$@"
        ;;
    watch)
        echo "Iniciando visor visual de campaña..."
        exec "$PYTHON_EXE" -m scripts.watch_campaign "$@"
        ;;
    status)
        exec "$PYTHON_EXE" -m scripts.run_campaign --status "$@"
        ;;
    run)
        echo "Iniciando nueva campaña autónoma..."
        exec "$PYTHON_EXE" -m scripts.run_campaign --new-game "$@"
        ;;
    resume)
        echo "Reanudando campaña autónoma..."
        exec "$PYTHON_EXE" -m scripts.run_campaign --resume "$@"
        ;;
    train)
        echo "Iniciando entrenamiento neuronal..."
        TRAIN_ARGS=("-m" "scripts.train_neural_modules")
        if [ -f "$SCRIPT_DIR/artifacts/checkpoints/neural.pt" ]; then
            TRAIN_ARGS+=("--resume")
        fi
        exec "$PYTHON_EXE" "${TRAIN_ARGS[@]}" "$@"
        ;;
    verify)
        echo "Verificando instalación de ARES..."
        exec "$PYTHON_EXE" -m scripts.verify_installation "$@"
        ;;
    demo)
        echo "Ejecutando verificación autónoma de demostración..."
        exec "$PYTHON_EXE" -m scripts.run_autonomous_verification_demo "$@"
        ;;
    *)
        exec "$PYTHON_EXE" -m scripts.run_campaign "$ACTION" "$@"
        ;;
esac
