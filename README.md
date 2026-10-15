# ARES-PKMN

Agente para FireRed USA y Emerald USA en Windows, con mGBA real, memoria GBA tipada, navegación por RAM, planificación HTN, inferencia causal, Z3 y modelos PyTorch. Las ROMs originales permanecen intactas; la campaña avanza mediante botones del emulador.

**Estado verificado:** la última ejecución registrada pasó 109 tests y las demos sintéticas de ambos juegos. El entrenamiento redujo VICReg de 63,62 a 39,26, Trajectory Balance de 3662,38 a 180,39 y mejoró la evaluación PPO de 39,1 % a 62,5 %. **Todavía no hay certificación de Campeón y Salón de la Fama en las dos ROMs reales.** La demo sintética no constituye esa certificación.

## Instalación

Trabaja desde `G:\ARES\ares-pkmn`. Se necesitan Windows x64, Python 3.12, `uv`, Git y CMake/Ninja/GCC en `PATH`. En este equipo ya están disponibles.

```powershell
.\ARES.ps1 setup
.\ARES.ps1 verify -Quick
```

El instalador crea `.venv`, sincroniza `requirements.lock` mediante `uv`, instala el proyecto en modo editable, adquiere las tres fuentes fijadas en `data/source_manifest.json` y compila `native/build/ares_mgba.dll`. No instala paquetes en Python global. El lock incluye PyTorch CPU, torchvision y DirectML; la selección automática usa CPU si la prueba completa de DirectML falla o resulta más lenta. En la verificación actual, DirectML falló durante backward y se eligió CPU.

Las ROMs deben estar un nivel por encima del proyecto:

| Archivo | SHA-1 admitido |
|---|---|
| `Pokemon - FireRed Version (USA).gba` | `41cb23d8dccc8ebd7c649cd8fbb58eeace6e2fdc` |
| `Pokemon - Emerald Version (USA, Europe).gba` | `f3ae088181bf583e55daf962a92bb46f4f1d07b7` |

## Ejecutar y reanudar

```powershell
.\ARES.ps1 run -Game both
.\ARES.ps1 status -Game both
.\ARES.ps1 resume -Game both
```

`run` comienza partidas nuevas con ARES, Bulbasaur y Mudkip. `both` ejecuta los juegos en procesos separados. No hay límite total de frames; `-MaxFrames 20000` permite una prueba acotada y conserva checkpoint. Una ejecución acotada que no haya terminado la campaña devuelve código 2. `resume` carga el último checkpoint completo de cada juego.

Para dejar una campaña en segundo plano:

```powershell
.\ARES.ps1 resume -Game both -Background
```

El launcher devuelve el PID comprobado y las rutas de stdout/stderr en `artifacts/processes`. Los checkpoints, registros de decisiones, estado JSON y capturas del emulador se guardan por backend y juego en `artifacts/campaigns`. Se guardan también cada diez minutos y después de hitos observados. El estado de la campaña puede consultarse mientras sigue corriendo.

## Verificar y entrenar

```powershell
.\ARES.ps1 train
.\ARES.ps1 demo
.\ARES.ps1 verify
```

`verify` ejecuta los cuatro comandos de aceptación y guarda sus logs y códigos de salida en `artifacts/verification`:

```powershell
.\.venv\Scripts\python.exe -m pytest -v --tb=short
.\.venv\Scripts\python.exe -m mypy --strict core scripts tests
.\.venv\Scripts\python.exe -m scripts.train_neural_modules
.\.venv\Scripts\python.exe -m scripts.run_autonomous_verification_demo
```

El checkpoint neuronal queda en `artifacts/checkpoints/neural.pt`. Las demos usan escenarios físicos configurados, movimientos, menús, capturas, victorias, elevaciones y más de 50 relocalizaciones de saves por juego; sus resultados se guardan bajo `artifacts/synthetic`.

## Criterios de evidencia

La certificación nativa exige ocho medallas, flag de juego completado y contador real de entradas al Salón de la Fama, acompañados por savestate, batería y registro de ejecución. Llegar a un destino o completar la demo no activa una certificación nativa.

Z3 registra qué decisiones tienen una prueba válida dentro del contexto modelado. Cuando el estado oculto del rival o los efectos del motor impiden demostrar supervivencia, el agente usa el riesgo estimado mínimo y registra la decisión como no demostrada; registra también desmayos y derrotas. Los datos ocultos del rival se excluyen de las entradas de la política y se estiman mediante creencias.
