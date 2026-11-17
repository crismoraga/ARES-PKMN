#!/usr/bin/env bash
# ============================================================================
# ARES-PKMN Linux / macOS Automated Setup
# ============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================================"
echo "  ARES-PKMN: Instalación y Configuración Automatizada (Linux / macOS)"
echo "============================================================================"
echo ""

# Step 1: Detect Python 3.12
echo "[1/5] Detectando Python 3.12..."
SYS_PYTHON=""
if command -v python3.12 >/dev/null 2>&1; then
    SYS_PYTHON="$(command -v python3.12)"
elif command -v python3 >/dev/null 2>&1; then
    VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null || true)"
    if [ "$VER" = "3.12" ]; then
        SYS_PYTHON="$(command -v python3)"
    else
        echo "[AVISO] python3 es versión $VER (se recomienda 3.12). Continuando con python3..."
        SYS_PYTHON="$(command -v python3)"
    fi
elif command -v python >/dev/null 2>&1; then
    SYS_PYTHON="$(command -v python)"
fi

if [ -z "$SYS_PYTHON" ]; then
    echo "[ERROR] No se encontró Python 3 en el sistema."
    echo "Instala Python 3.12 usando tu gestor de paquetes (apt, dnf, pacman, brew):"
    echo "  Ubuntu/Debian: sudo apt update && sudo apt install python3.12 python3.12-venv"
    echo "  macOS (Homebrew): brew install python@3.12"
    exit 1
fi
echo "  Encontrado: $SYS_PYTHON ($($SYS_PYTHON --version))"
echo ""

# Step 2: Check uv
echo "[2/5] Comprobando herramienta uv..."
UV_CMD=""
if command -v uv >/dev/null 2>&1; then
    UV_CMD="uv"
    echo "  uv encontrado en el sistema."
else
    echo "  uv no encontrado; intentando instalar con pip..."
    "$SYS_PYTHON" -m pip install --user uv >/dev/null 2>&1 || true
    if command -v uv >/dev/null 2>&1; then
        UV_CMD="uv"
    elif [ -f "$HOME/.cargo/bin/uv" ]; then
        UV_CMD="$HOME/.cargo/bin/uv"
    elif [ -f "$HOME/.local/bin/uv" ]; then
        UV_CMD="$HOME/.local/bin/uv"
    fi
fi
echo ""

# Step 3: Create virtual environment
echo "[3/5] Configurando entorno virtual (.venv)...
VENV_DIR="$SCRIPT_DIR/.venv"
VENV_PYTHON="$VENV_DIR/bin/python"

if [ ! -f "$VENV_PYTHON" ]; then
    if [ -n "$UV_CMD" ]; then
        echo "  Creando .venv con uv..."
        "$UV_CMD" venv --python "$SYS_PYTHON" "$VENV_DIR"
    else
        echo "  Creando .venv con python -m venv..."
        "$SYS_PYTHON" -m venv "$VENV_DIR"
    fi
else
    echo "  Entorno virtual ya existente en .venv."
fi
echo ""

# Step 4: Install dependencies
echo "[4/5] Instalando dependencias del proyecto..."
CPU_INDEX="https://download.pytorch.org/whl/cpu"
if [ -n "$UV_CMD" ]; then
    echo "  Sincronizando dependencias con uv..."
    "$UV_CMD" pip install --python "$VENV_PYTHON" --extra-index-url "$CPU_INDEX" \
        torch numpy scipy pyyaml z3-solver pygame
    "$UV_CMD" pip install --python "$VENV_PYTHON" --no-deps -e "$SCRIPT_DIR"
else
    echo "  Instalando dependencias con pip..."
    "$VENV_PYTHON" -m pip install --upgrade pip
    "$VENV_PYTHON" -m pip install --extra-index-url "$CPU_INDEX" \
        torch numpy scipy pyyaml z3-solver pygame
    "$VENV_PYTHON" -m pip install --no-deps -e "$SCRIPT_DIR"
fi
echo ""

# Step 5: Verification
echo "[5/5] Verificando instalación..."
chmod +x "$SCRIPT_DIR/ares.sh" "$SCRIPT_DIR/demo.sh" 2>/dev/null || true
"$VENV_PYTHON" -c "
import torch, numpy, scipy, z3, pygame
from core.stadium.dex import load_dex
from core.stadium.teams import load_bank
dex = load_dex(prefer_snapshot=True)
bank = load_bank(dex)
print('  [OK] Todas las librerias y tablas de datos Gen-3 estan cargadas y verificadas.')
print(f'  [OK] Stadium disponible sin ROMs: {len(dex.species)} especies, {len(dex.moves)} movimientos.')
"

echo ""
echo "============================================================================"
echo "  Instalación completada con éxito."
echo "  Para iniciar la demostración en vivo de convención, ejecuta: ./demo.sh"
echo "  O para iniciar la estación de monitoreo web: ./ares.sh dashboard"
echo "============================================================================"
