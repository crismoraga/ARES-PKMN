# ARES-PKMN

ARES-PKMN es un agente de investigación para Pokémon FireRed y Emerald de tercera generación. Integra un emulador mGBA nativo, lectura de RAM, navegación, planificación, análisis de combate y modelos PyTorch. Las campañas nativas avanzan enviando botones al emulador; el laboratorio sintético sirve para pruebas repetibles y **no certifica** una partida real.

Esta guía permite preparar Windows desde un clon limpio, ejecutar una campaña y observarla junto con el entrenamiento. El instalador usa Python 3.12 aislado en `.venv`; no instala paquetes en el Python global. La instalación descarga dependencias y fuentes fijadas, por lo que necesita Internet.

## Estado actual y límites de la evidencia

En el checkout de desarrollo se registraron 143 tests aprobados, mypy estricto aprobado y una demo sintética de ambos juegos. El entrenamiento registrado produjo VICReg `63.62 → 39.26`, Trajectory Balance `3662.38 → 180.39` y win rate PPO evaluado `39.1 % → 65.6 %`. DirectML falló en la comprobación completa de backward en este equipo; el entrenamiento usó CPU.

Las campañas FireRed/Emerald nativas **siguen sin terminar ni certificarse**. Los status actuales disponibles son exploratorios: FireRed tiene 5 hitos y Emerald 8; ninguno registra medallas o entradas al Salón de la Fama. No presentes las capturas ni el demo como prueba de haber vencido al Campeón. Consulta [estado y evidencia](docs/ESTADO_Y_EVIDENCIA.md).

## Requisitos del PC

- Windows x64 (Windows 10/11), Python 3.12 x64, Git, `uv`, CMake, Ninja y GCC en `PATH`.
- Conexión a Internet durante la instalación para descargar ruedas Python y fuentes nativas fijadas.
- Al menos 16 GB de RAM recomendados para ejecutar dos juegos y entrenar. La GPU es opcional; CPU es la ruta de referencia.
- Dos ROMs compatibles que el usuario aporte y tenga autorización para usar. No se incluyen ni se descargan desde este proyecto.

Para instalar las herramientas faltantes con `winget` en PowerShell (si los IDs están disponibles en el catálogo de tu PC):

```powershell
winget install --id Python.Python.3.12 --exact
winget install --id Git.Git --exact
winget install --id astral-sh.uv --exact
winget install --id Kitware.CMake --exact
winget install --id Ninja-build.Ninja --exact
```

GCC debe ser una distribución Windows x64, por ejemplo la toolchain de MSYS2/MinGW-w64; añade su carpeta `bin` al `PATH` y verifica `gcc --version`. Cierra y vuelve a abrir PowerShell después de cambiar `PATH`. Comprueba todo con:

```powershell
python --version   # 3.12.x
git --version
uv --version
cmake --version
ninja --version
gcc --version
```

## Clonar e instalar

El repositorio configurado en este checkout es `https://github.com/crismoraga/ARES-PKMN.git`. Para una máquina nueva, clona la rama publicada y entra al directorio:

```powershell
git clone https://github.com/crismoraga/ARES-PKMN.git G:\ARES\ares-pkmn
Set-Location G:\ARES\ares-pkmn
```

Si usas otra carpeta, conserva esta estructura: las ROMs van en la carpeta padre del checkout. Antes de clonar, confirma que la rama/commit con esta documentación ya esté publicado; una carpeta local no publicada no puede replicarse con `git clone`.

Coloca exactamente estos nombres de archivo en `G:\ARES` si clonaste en `G:\ARES\ares-pkmn`:

| Juego | Archivo esperado | SHA-1 admitido |
|---|---|---|
| FireRed | `Pokemon - FireRed Version (USA).gba` | `41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc` |
| Emerald | `Pokemon - Emerald Version (USA, Europe).gba` | `f3ae088181bf583e55daf962a92bb46f4f1d07b7` |

Instala y verifica:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\ARES.ps1 setup
.\ARES.ps1 verify -Quick
```

`setup` crea `.venv`, sincroniza `requirements.lock`, instala el proyecto editable, baja mGBA 0.10.5 y las fuentes/símbolos pret fijados en `data/source_manifest.json`, y compila `native\build\ares_mgba.dll`. El build necesita Git, CMake, Ninja y GCC. No borres ni edites `requirements.lock` para una réplica comparable. Más detalles y solución de errores: [instalación Windows](docs/INSTALACION_WINDOWS.md) y [problemas comunes](docs/PROBLEMAS.md).

## Ejecutar campaña y verla en vivo

Abre **dos ventanas de PowerShell** en `G:\ARES\ares-pkmn`. En la primera inicia las dos campañas nativas en segundo plano:

```powershell
.\ARES.ps1 run -Game both -Background
```

El comando imprime el PID y crea `artifacts\processes\latest.json`, más logs stdout/stderr. `run` inicia una partida nueva con nombre ARES, Bulbasaur en FireRed y Mudkip en Emerald. Si ya existe progreso que quieras conservar, usa `resume` en vez de `run`.

En la segunda abre la ventana visual del emulador:

```powershell
.\ARES.ps1 watch
```

El visor muestra los framebuffers reales de 240×160, objetivo, mapa/coordenadas, party, contadores de combate/decisiones y eventos recientes para ambos juegos. `Esc` o cerrar esa ventana cierra únicamente el visor; no detiene las campañas. Para guardar una captura periódica de la ventana completa:

```powershell
.\ARES.ps1 watch -SnapshotPath artifacts\live_viewer.png
```

El visor requiere el mismo directorio de checkpoints que el proceso. El comando anterior usa `artifacts\campaigns`; si ejecutaste campañas con `--checkpoint-root` distinto directamente, pásalo al visor con el comando Python indicado en [monitoreo en vivo](docs/MONITOREO_EN_VIVO.md).

Consulta el avance y reanuda con:

```powershell
.\ARES.ps1 status -Game both
.\ARES.ps1 resume -Game both -Background
```

Para una prueba acotada, ejecuta en primer plano `\.\ARES.ps1 run -Game firered -MaxFrames 20000`; guarda estado/checkpoint al salir, pero código 2 significa que alcanzó el límite antes de terminar. Las campañas normales no tienen límite total. Para pedir guardado o parada ordenada de workers lanzados con la ruta por defecto:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_campaign --game both --checkpoint
.\.venv\Scripts\python.exe -m scripts.run_campaign --game both --stop
```

`--stop` solicita checkpoint y cierre ordenado; reanuda luego con `\.\ARES.ps1 resume -Game both`. No mates el proceso desde el administrador de tareas si puedes usar esa parada.

## Entrenamiento visible

Para que una campaña cargue el checkpoint neuronal recién entrenado, entrena **antes** de iniciar `run`/`resume`. En una tercera ventana PowerShell, o antes de abrir las dos ventanas anteriores:

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
.\.venv\Scripts\python.exe -u -m scripts.train_neural_modules 2>&1 |
  Tee-Object -FilePath artifacts\training-live.log
```

La consola imprime métricas JSON durante VICReg/Trajectory Balance y evaluación PPO; `Tee-Object` conserva una copia en `artifacts\training-live.log`. Para que la campaña use esos pesos, deja terminar el entrenamiento y después lánzala. Si prefieres monitorizar un entrenamiento de fondo ya iniciado mediante `\.\ARES.ps1 train -Background`, consulta su ruta en `artifacts\processes\latest.json` y sigue el `stdout` con `Get-Content <ruta-stdout> -Wait`.

La ventana del emulador muestra métricas de campaña; el entrenamiento imprime sus métricas en la consola de entrenamiento. Son flujos separados. Consulta [monitoreo en vivo](docs/MONITOREO_EN_VIVO.md) para comandos, rutas y recuperación.

## Comprobaciones y demo sintética

```powershell
.\ARES.ps1 verify
.\ARES.ps1 demo
```

La verificación completa equivale a:

```powershell
.\.venv\Scripts\python.exe -m pytest -v --tb=short
.\.venv\Scripts\python.exe -m mypy --strict core scripts tests
.\.venv\Scripts\python.exe -m scripts.train_neural_modules
.\.venv\Scripts\python.exe -m scripts.run_autonomous_verification_demo
```

`verify -Quick` omite el entrenamiento y la demo larga. Los logs y códigos de salida se guardan en `artifacts\verification`; la demo en `artifacts\synthetic`; el checkpoint neuronal en `artifacts\checkpoints\neural.pt`.

## Estructura de ejecución

- `core/`: emulación/bridge, memoria, combate, causalidad, navegación, modelos y orquestador.
- `native/`: shim C/CMake y build local; fuentes de terceros adquiridas por setup en `native/vendor/`.
- `config/`, `data/`: configuración y manifiesto de versiones, ROM hashes y símbolos.
- `scripts/`: instalación, campaña, visor, entrenamiento y verificación.
- `tests/`: pruebas sintéticas y unitarias.
- `artifacts/`: logs, capturas, métricas y checkpoints locales. No se necesita versionar este directorio.

Las campañas registran decisiones demostradas/no demostradas, riesgo, desmayos y derrotas. No se garantiza supervivencia absoluta: cuando el solver no prueba la seguridad, la política selecciona la alternativa de menor riesgo estimado. La definición de éxito nativo requiere ocho medallas, flag de juego completado y entrada real al Salón de la Fama; verificación detallada en [estado y evidencia](docs/ESTADO_Y_EVIDENCIA.md).

## Guías

- [Instalación reproducible en Windows](docs/INSTALACION_WINDOWS.md)
- [Campaña, visor y entrenamiento en vivo](docs/MONITOREO_EN_VIVO.md)
- [Arquitectura, pruebas y evidencia](docs/ESTADO_Y_EVIDENCIA.md)
- [Solución de problemas](docs/PROBLEMAS.md)
