# ARES-PKMN

ARES-PKMN es un agente de investigación para Pokémon FireRed y Emerald de tercera generación. Integra un emulador mGBA nativo, lectura de RAM, navegación, planificación, análisis de combate y modelos PyTorch. Las campañas nativas avanzan enviando botones al emulador; el laboratorio sintético sirve para pruebas repetibles y **no certifica** una partida real.

Esta guía permite preparar Windows desde un clon limpio, ejecutar una campaña y observarla junto con el entrenamiento. El instalador usa Python 3.12 aislado en `.venv`; no instala paquetes en el Python global. La instalación descarga dependencias y fuentes fijadas, por lo que necesita Internet.

## Estado actual y límites de la evidencia

La verificación incluye regresiones de evolución/aprendizaje, captura/Pokédex, paredes direccionales de cuevas, persecución de NPCs, ataques sin efecto y ciclos sin avance de historia. Los resultados del checkout se detallan en [estado y evidencia](docs/ESTADO_Y_EVIDENCIA.md). La ruta funcional de referencia usa CPU; DirectML debe superar sus propias comprobaciones antes de seleccionarse.

Las campañas FireRed/Emerald nativas **siguen sin terminar ni certificarse**. La recuperación nativa de FireRed completó el guardián del fósil, el fósil de Mt. Moon y el rival de Ciudad Celeste; la de Emerald consiguió hablar con Briney y viajar a Dewford. Los contadores actuales se consultan en el dashboard y no son un benchmark desde una partida nueva. La certificación exige ocho medallas y entrada al Salón de la Fama.

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

En la segunda abre la **estación de monitoreo** en el navegador:

```powershell
.\ARES.ps1 dashboard
```

Abre `http://127.0.0.1:8765/` con ambos juegos en vivo: pantalla real, estado del proceso, objetivo e hitos, medallas, Pokédex, dinero y tiempo de juego, equipo completo (nivel, PS, estado, tipos, EXP, movimientos y PP), combate en curso, tabla de decisiones con su riesgo, línea de tiempo filtrable, gráficos temporales, checkpoints y supervisor, más botones para guardar o detener. La pestaña **Entrenamiento** muestra la etapa, el proceso y las curvas reales de VICReg, Trajectory Balance, victorias PPO y pérdida de valor, separadas por sesión. Detalles en [monitoreo en vivo](docs/MONITOREO_EN_VIVO.md).

También existe el visor de escritorio:

```powershell
.\ARES.ps1 watch
```

El visor muestra los framebuffers reales de 240×160, objetivo, mapa/coordenadas, el equipo con nombres y barras de PS, la salud del agente, contadores de combate/decisiones y eventos recientes para ambos juegos. `Esc` o cerrar esa ventana cierra únicamente el visor; no detiene las campañas. Para guardar una captura periódica de la ventana completa:

```powershell
.\ARES.ps1 watch -SnapshotPath artifacts\live_viewer.png
```

El visor requiere el mismo directorio de checkpoints que el proceso. El comando anterior usa `artifacts\campaigns`; si ejecutaste campañas con `--checkpoint-root` distinto directamente, pásalo al visor con el comando Python indicado en [monitoreo en vivo](docs/MONITOREO_EN_VIVO.md).

Consulta el avance y reanuda con:

```powershell
.\ARES.ps1 status -Game both
.\ARES.ps1 resume -Game both -Background
```

Para una prueba acotada, ejecuta en primer plano `.\ARES.ps1 run -Game firered -MaxFrames 20000`; guarda estado/checkpoint al salir, pero código 2 significa que alcanzó el límite antes de terminar. Las campañas normales no tienen límite total. Para pedir guardado o parada ordenada de workers lanzados con la ruta por defecto:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_campaign --game both --checkpoint
.\.venv\Scripts\python.exe -m scripts.run_campaign --game both --stop
```

`--stop` solicita checkpoint y cierre ordenado; reanuda luego con `.\ARES.ps1 resume -Game both`. No mates el proceso desde el administrador de tareas si puedes usar esa parada.

### Autonomía: qué pasa cuando algo falla

Un error de política se registra (`agent_fault`) y provoca una nueva planificación. La recuperación tiene pasos y presupuestos explícitos: replanificar, resolver menús, desviar, vetar salidas y restaurar un checkpoint verificado. Un reloj estratégico independiente detecta falta de historia aunque siga aumentando la experiencia; el entrenamiento real le descuenta tiempo con un presupuesto acotado y, al vencer, el agente **cambia de estrategia** en lugar de repetir la misma. Cuando el supervisor agota los reinicios sin hitos nuevos, espera (`--stagnation-cooldown`, 300 s) y vuelve a intentarlo con las lecciones acumuladas. Los checkpoints se conservan por retención (24 recientes + hitos + fijados). Estas defensas reducen los ciclos conocidos; no garantizan resolver cualquier estado de la ROM. Detalles en [recuperación autónoma](docs/RECUPERACION_AUTONOMA.md).

### Inteligencia: requisitos, lecciones y preparación

- **Guardias y puertas con requisito**: el agente conoce, por análisis de los scripts originales, cada casilla que un script deshace (los guardias sedientos de Azafrán piden el Té; la tormenta de la Ruta 111 exige Gafas Aisladoras; los guardias de la Ruta 23, las medallas) y planifica rutas que no las crucen mientras el requisito falte. Si un script lo devuelve en un sitio desconocido, **aprende** ese paso denegado y no lo repite hasta que cambie el mundo (hito u objeto clave nuevo).
- **Hacer algo antes de avanzar**: si todas las rutas están cerradas, busca qué las abre (la flag, el objeto o el evento que retira a unos NPCs) y va primero allí; por ejemplo, habla con Dock en el astillero de Portual para que la fila del Equipo Aqua deje libre el museo.
- **Preparación y derrotas**: antes de un líder o rival se cura si el miembro principal no está casi lleno o sus mejores ataques tienen poco PP; cada derrota se analiza (entrar sin curar no sube el nivel objetivo, entrar preparado sí).
- **Tiendas y pociones**: compra Superpociones y Poké Balls cuando hay dinero y una tienda cerca, y usa pociones en combate cuando atacar probablemente perdería.
- **Lecciones duraderas** (`lessons.json`): pasos denegados, inmunidades observadas y derrotas sobreviven a retrocesos y reinicios.

## Entrenamiento visible

Para que una partida nueva cargue el checkpoint neuronal recién entrenado, entrena **antes** de iniciar `run`. `resume` restaura los pesos y el estado cognitivo guardados en su propio checkpoint de campaña. En una tercera ventana PowerShell:

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
.\.venv\Scripts\python.exe -u -m scripts.train_neural_modules 2>&1 |
  Tee-Object -FilePath artifacts\training-live.log
```

La consola imprime métricas JSON durante VICReg/Trajectory Balance y evaluación PPO; `Tee-Object` conserva una copia en `artifacts\training-live.log`. Para que la campaña use esos pesos, deja terminar el entrenamiento y después lánzala. Si prefieres monitorizar un entrenamiento de fondo ya iniciado mediante `.\ARES.ps1 train -Background`, consulta su ruta en `artifacts\processes\latest.json` y sigue el `stdout` con `Get-Content <ruta-stdout> -Wait`.

El entrenamiento publica también `artifacts/training/live.json` y `metrics.jsonl`. Puedes seguirlo en la pestaña **Entrenamiento** del mismo dashboard mientras los juegos avanzan; las curvas de laboratorio y la historia de las ROMs están identificadas por separado. Los pesos de una campaña en marcha no se sustituyen automáticamente al acabar ese entrenamiento.

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
- `scripts/`: instalación, campaña y supervisor, estación de monitoreo web (`dashboard.py` + `dashboard_static/`), visor, entrenamiento y verificación.
- `tests/`: pruebas sintéticas y unitarias.
- `artifacts/`: logs, capturas, métricas y checkpoints locales. No se necesita versionar este directorio.

Las campañas registran decisiones demostradas/no demostradas, riesgo, desmayos y derrotas. No se garantiza supervivencia absoluta: cuando el solver no prueba la seguridad, la política selecciona la alternativa de menor riesgo estimado. La definición de éxito nativo requiere ocho medallas, flag de juego completado y entrada real al Salón de la Fama; verificación detallada en [estado y evidencia](docs/ESTADO_Y_EVIDENCIA.md).

## Guías

- [Instalación reproducible en Windows](docs/INSTALACION_WINDOWS.md)
- [Campaña, visor y entrenamiento en vivo](docs/MONITOREO_EN_VIVO.md)
- [Arquitectura, pruebas y evidencia](docs/ESTADO_Y_EVIDENCIA.md)
- [Solución de problemas](docs/PROBLEMAS.md)
- [Recuperación autónoma y reproducción de bloqueos](docs/RECUPERACION_AUTONOMA.md)
