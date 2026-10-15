# Instalación reproducible en Windows

Esta es la plataforma de instalación mantenida por el proyecto: Windows x64 y Python 3.12. El instalador rechaza otras versiones/plataformas para evitar una instalación parcialmente compatible. Los comandos se ejecutan en PowerShell.

## 1. Herramientas

Instala Python 3.12 x64, Git, `uv`, CMake, Ninja y GCC x64. `winget` puede instalar varios paquetes:

```powershell
winget install --id Python.Python.3.12 --exact
winget install --id Git.Git --exact
winget install --id astral-sh.uv --exact
winget install --id Kitware.CMake --exact
winget install --id Ninja-build.Ninja --exact
```

Instala GCC de MSYS2/MinGW-w64 por el instalador oficial de tu distribución, y agrega el directorio `mingw64\bin` (o equivalente) al `PATH`. Abre una consola nueva y verifica que `gcc --version` funcione. El setup no instala compiladores ni altera variables globales.

```powershell
python --version
git --version
uv --version
cmake --version
ninja --version
gcc --version
```

`python --version` debe ser `3.12.x`. Si tienes varias versiones, deja Python 3.12 en `PATH`; `uv venv --python 3.12` elige esa versión para `.venv`.

## 2. Clon y ROMs

```powershell
git clone https://github.com/crismoraga/ARES-PKMN.git G:\ARES\ares-pkmn
Set-Location G:\ARES\ares-pkmn
```

El clon debe contener el commit publicado que se quiere reproducir. Si el propietario publica otra URL/rama, usa esa URL y anótala junto con el SHA (`git rev-parse HEAD`) para que dos máquinas usen el mismo código.

El motor busca ROMs en el directorio padre del repositorio. En el ejemplo, ponlas en `G:\ARES` con estos nombres. Cada hash debe coincidir exactamente:

| Juego | Nombre | SHA-1 |
|---|---|---|
| FireRed | `Pokemon - FireRed Version (USA).gba` | `41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc` |
| Emerald | `Pokemon - Emerald Version (USA, Europe).gba` | `f3ae088181bf583e55daf962a92bb46f4f1d07b7` |

El programa valida el hash antes de adjuntar el backend. Los archivos ROM no forman parte del clon, instalador, releases ni instrucciones de descarga: aporta una copia que estés autorizado a usar. No subas ROMs ni saves personales a Git.

## 3. Entorno aislado y compilación

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\ARES.ps1 setup
```

El script `scripts.setup_environment` realiza, en orden:

1. Crea `.venv` con Python 3.12 si aún no existe.
2. Ejecuta `uv pip sync` usando `requirements.lock` y el índice CPU de PyTorch.
3. Instala el paquete local en modo editable sin resolver nuevas dependencias.
4. Descarga los commits fijados de mGBA 0.10.5, `pokefirered` y `pokeemerald`, y los mapas de símbolos correspondientes.
5. Compila `native\build\ares_mgba.dll` mediante CMake/Ninja/GCC.

Para revisar prerrequisitos sin instalar nada:

```powershell
python -m scripts.setup_environment --check-only
```

Si el setup falló a mitad, corrige primero el error (normalmente `PATH`, Internet o espacio). Luego vuelve a correr `.\ARES.ps1 setup`; la adquisición comprueba commits y no reemplaza deliberadamente una fuente local con un commit distinto. El instalador y dependencias quedan dentro del repo, en `.venv` y `native\vendor`; no se modifica Python global.

## 4. Verificación de instalación

La primera pasada rápida no requiere ejecutar el entrenamiento completo:

```powershell
.\ARES.ps1 verify -Quick
```

Pruebas unitarias y tipos directamente:

```powershell
.\.venv\Scripts\python.exe -m pytest -v --tb=short
.\.venv\Scripts\python.exe -m mypy --strict core scripts tests
```

La suite `native` puede requerir las ROMs correctas y la DLL compilada. La aceptación completa agrega entrenamiento y demo:

```powershell
.\ARES.ps1 verify
```

Busca reporte/setup en `artifacts\setup.json`, logs en `artifacts\verification` y demo en `artifacts\synthetic`. Para repetir exactamente, registra el commit de ARES, hashes ROM y commits fuente que aparecen en `data/source_manifest.json`; utiliza el mismo Windows/arquitectura y `requirements.lock`.

## 5. GPU y CPU

La CPU es el backend de referencia. La detección DirectML ejecuta pruebas completas de forward, backward y actualización antes de seleccionarlo; tener una GPU AMD/NVIDIA no implica automáticamente que DirectML sea compatible o más rápido. Si falla, se informa y se continúa en CPU. No es requisito instalar drivers especiales para ejecutar la ruta CPU. Revisa el log de entrenamiento para ver `Backend: cpu` o `Backend: directml`.
