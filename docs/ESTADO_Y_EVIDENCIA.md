# Arquitectura, verificación y evidencia

## Qué se ejecuta

`scripts.run_campaign` crea una orquestación independiente por juego. En el backend nativo abre la ROM con `native\build\ares_mgba.dll`, conduce frames/botones y obtiene observaciones de RAM. El visor (`scripts.watch_campaign`) solo lee publicaciones de frame/estado y no controla el emulador. El backend sintético se reserva para escenarios de pruebas con estados reproducibles.

Los componentes de código están en `core/`: bridge y estructuras de memoria, guardado/cifrado, combate/seguridad, inferencia causal, mapas y planificación, módulos neuronales y orquestador. `native/` contiene el adaptador C/CMake; el setup descarga las fuentes externas en commits fijos de `data/source_manifest.json`. `config/` y `data/` guardan configuración y metadatos de fuentes/símbolos. `tests/` contiene pruebas unitarias/integración.

## Repetibilidad

Guarda estos datos con cada ejecución para poder comparar equipos:

1. Commit de ARES (`git rev-parse HEAD`).
2. Versión de Python (`.venv\Scripts\python.exe --version`) y Windows x64.
3. Versión de `requirements.lock`; no resolver dependencias flotantes.
4. Hashes ROM observados en `native_completion_evidence`/estado.
5. `data/source_manifest.json` y backend Torch elegido.
6. Semilla (`42` por defecto), argumentos completos y raíz de checkpoints.
7. `status.json`, `events.jsonl`, stdout/stderr y checkpoints de cada juego.

No subas ROMs, archivos de batería ni datos privados a un repositorio público. Para compartir evidencia, publica logs sanitizados y capturas autorizadas.

## Verificación software

Comandos completos:

```powershell
.\.venv\Scripts\python.exe -m pytest -v --tb=short
.\.venv\Scripts\python.exe -m mypy --strict core scripts tests
.\.venv\Scripts\python.exe -m scripts.train_neural_modules
.\.venv\Scripts\python.exe -m scripts.run_autonomous_verification_demo
```

El launcher los ejecuta con `.\ARES.ps1 verify`; `-Quick` ejecuta solamente las verificaciones rápidas. Los archivos de resultados se guardan en `artifacts\verification`. La verificación durante la recuperación del 15 de octubre registró 377 tests aprobados y mypy estricto sobre 75 archivos; los cambios finales del supervisor y sus regresiones tienen además su verificación específica. Logs locales: `artifacts/pytest-autonomy-current.log` y `artifacts/mypy-autonomy-current.log`. Repite los comandos en el clon que se pretenda certificar: esos resultados no garantizan otro checkout o hardware.

El entrenamiento reanudado con `--resume --device cpu` terminó con código cero: VICReg de validación `38.36165 → 38.36165` (se conservó el mejor checkpoint, sin mejora medida), Trajectory Balance `30.36211 → 14.83554` y victorias PPO `92.1875 % → 100 %` en su evaluación de laboratorio. La telemetría guarda el identificador de sesión y sus curvas; `artifacts/training-recovery-live.log` conserva stdout. Esas victorias no corresponden a las campañas nativas.

## Qué prueba la demo y qué no

La demo sintética recorre escenarios deterministas diseñados para probar subsistemas y sus interacciones. No levanta una campaña nueva en los dos juegos originales, no obtiene las ocho medallas del archivo ROM, y no demuestra el Salón de la Fama. Un test aprobado significa que pasó una expectativa software, no que la estrategia venza toda partida.

## Certificación de campaña nativa

Una campaña se marca completa solo al observar en la ROM correcta:

- Las ocho medallas de esa partida.
- La flag real de juego completado.
- Un contador positivo de entradas al Salón de la Fama.
- Evidencia acompañante: hash ROM esperado, checkpoint/savestate, status final y log de eventos.

Las recuperaciones nativas registradas todavía no cumplen esos criterios. FireRed pasó de 6 a 9 hitos: guardián del fósil, recogida del fósil de Mt. Moon y rival de Ciudad Celeste; Emerald pasó de 15 a 16 hitos después de interceptar a Briney, viajar a Dewford y entregar la carta a Steven. Ambas partidas conservaban una medalla y cero entradas al Salón de la Fama. Son continuaciones de partidas exploratorias, con derrotas y retrocesos previos, sin reiniciar ni editar los Pokémon para la prueba.

Evidencia local: `artifacts/verified_story/native/firered/recovery_result.json`, `artifacts/briney_diagnosis/patrol-report.json`, `artifacts/briney_diagnosis/steven-report.json` y `artifacts/verified_recovery_import.json`. Los checkpoints originales se conservaron y las copias verificadas se importaron con procedencia registrada. Los contadores posteriores cambian mientras los workers juegan: consulta el dashboard para el estado actual.

El solver marca como demostradas solo decisiones con prueba válida en el contexto modelado. Con incertidumbre o `unknown`, no afirma seguridad; toma la acción de menor riesgo estimado y registra “sin prueba”. Desmayos y derrotas también son parte de la telemetría.

## GPU

La instalación bloquea Torch CPU como ruta estándar y añade la rueda DirectML compatible con Windows. DirectML solo se selecciona si pasa la prueba de forward, backward, actualización y comparación con CPU y ofrece una ruta útil. En el equipo Radeon RX 6700 XT de referencia esa prueba falló con una aserción de PyTorch y se usó CPU. No prometas aceleración GPU en otra PC sin leer primero la línea `Backend:` de su propio entrenamiento.
