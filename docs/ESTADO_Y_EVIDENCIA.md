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

El launcher los ejecuta con `.ARES.ps1 verify`; `-Quick` ejecuta solamente las verificaciones rápidas. Los archivos de resultados se guardan en `artifacts\verification`. Resultado registrado en este checkout: 143 tests aprobados; mypy estricto pasó; entrenamiento y demo finalizaron correctamente. En la ejecución documentada de entrenamiento, se seleccionó CPU después de que la prueba DirectML backward falló. Repite estos comandos en el clon que se pretenda certificar: esos resultados históricos no garantizan el mismo hardware ni un checkout distinto.

## Qué prueba la demo y qué no

La demo sintética recorre escenarios deterministas diseñados para probar subsistemas y sus interacciones. No levanta una campaña nueva en los dos juegos originales, no obtiene las ocho medallas del archivo ROM, y no demuestra el Salón de la Fama. Un test aprobado significa que pasó una expectativa software, no que la estrategia venza toda partida.

## Certificación de campaña nativa

Una campaña se marca completa solo al observar en la ROM correcta:

- Las ocho medallas de esa partida.
- La flag real de juego completado.
- Un contador positivo de entradas al Salón de la Fama.
- Evidencia acompañante: hash ROM esperado, checkpoint/savestate, status final y log de eventos.

El estado de desarrollo consultado para esta documentación todavía no cumple esos criterios. FireRed registraba 5 hitos (242422 frames, 7 combates, 4 victorias, 2 derrotas y 2 debilitados); Emerald registraba 8 hitos (285000 frames, 13 combates, 11 victorias, una derrota y un debilitado). Ambos tenían cero medallas y cero entradas al Hall of Fame. Son estados exploratorios locales, no una certificación ni un benchmark limpio reproducible. Los directorios de campaña pueden contener progreso parcial.

El solver marca como demostradas solo decisiones con prueba válida en el contexto modelado. Con incertidumbre o `unknown`, no afirma seguridad; toma la acción de menor riesgo estimado y registra “sin prueba”. Desmayos y derrotas también son parte de la telemetría.

## GPU

La instalación bloquea Torch CPU como ruta estándar y añade la rueda DirectML compatible con Windows. DirectML solo se selecciona si pasa la prueba de forward, backward, actualización y comparación con CPU y ofrece una ruta útil. En el equipo Radeon RX 6700 XT de referencia esa prueba falló con una aserción de PyTorch y se usó CPU. No prometas aceleración GPU en otra PC sin leer primero la línea `Backend:` de su propio entrenamiento.
