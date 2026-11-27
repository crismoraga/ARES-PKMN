# docs/investigacion — Índice

Esta carpeta contiene la investigación del proyecto **ARES-U** del curso IPD434 Soft Computing (2026-2). Las notas de investigación tienen fecha de corte 2026-11-24. El objetivo de largo plazo es un agente neuronal que juegue *cualquier* Pokémon. El objetivo del curso es demostrar con rigor, mediante E1 y E2, la pieza que habilita ese objetivo: **representaciones independientes del juego que transfieren a contenido no visto**.

## Documentos principales

| Archivo | Qué contiene | Cuándo leerlo |
|---|---|---|
| [`PLAN_IPD434.md`](PLAN_IPD434.md) | Plan del proyecto: visión, alcance, especificación de **E1** (ARES-Battle: Transformer de entidades sin IDs; entrena con Gen 1 y se evalúa zero-shot en Gen 2 y Gen 3) y **E2** (ARES-Nav: píxeles con AE y RND, de FireRed a Emerald), líneas base, protocolo, nota sobre DirectML y cronograma | Primero |
| [`ESTADO_REPOSITORIO.md`](ESTADO_REPOSITORIO.md) | Cómo está configurado y cómo funciona ARES-PKMN hoy: arquitectura, traza del bucle de decisión, qué son reglas y qué es aprendido, rol real de las redes, Stadium, tests y mypy, progreso de las campañas, atascos, throughput, hallazgo de DirectML, acoplamiento a Gen 3 y diagnóstico | Para entender el punto de partida |
| [`ESTADO_DEL_ARTE.md`](ESTADO_DEL_ARTE.md) | Revisión estructurada en cinco partes: (a) RL, (b) LLM/VLM, (c) IA de combate y benchmarks (PokéAgent, Metamon…), (d) agentes generalistas y métodos NN modernos, (e) infraestructura multi-generación. Incluye tablas con año, enfoque, juegos, resultados, limitaciones y URL | Sección de trabajo relacionado del informe |
| [`VICIOS_Y_SOLUCIONES.md`](VICIOS_Y_SOLUCIONES.md) | **Entregable central.** Catálogo de 17 vicios del estado del arte y 6 propios de ARES-PKMN, con evidencia, severidad, respuesta de ARES-U y su verificación (E1/E2), y lo que queda sin resolver. Termina con la **arquitectura objetivo de ARES-U** y su correspondencia con las unidades del curso | Motivación y diseño |
| [`referencias.bib`](referencias.bib) | BibTeX para `IEEEtran`. Los papers llevan su ID de arXiv; los repos y streams van como `@misc` con `urldate = 2026-11-24`. Las entradas con `note = {VERIFICAR ...}` tienen autores o títulos que no se pudieron confirmar | Al escribir el informe |

## Notas de investigación (`notas/`)

Son la fuente primaria de los documentos anteriores. Usan las marcas [V] (verificado), [S] (solo fuente secundaria) y [NV] (no verificado).

| Nota | Tema |
|---|---|
| [`notas/repo_decision_loop_bridge.md`](notas/repo_decision_loop_bridge.md) | Auditoría del bridge mGBA y del bucle de decisión: observación por RAM, traza de un tick, aprendido frente a hardcodeado, atascos, throughput, techo de generalización |
| [`notas/repo_neural_stadium.md`](notas/repo_neural_stadium.md) | Auditoría del stack neuronal (H-JEPA, GFlowNet, Mamba/Transformer, PPO/PSRO) y del Stadium (motor, políticas, GAME/QD, velocidad, líneas base) |
| [`notas/repo_health.md`](notas/repo_health.md) | Salud del repositorio: git, dependencias, hardware, tests (451 passed), mypy, estado real de las campañas |
| [`notas/course_and_local_docs.md`](notas/course_and_local_docs.md) | Qué exige el curso IPD434 y qué conceptos enseña (con líneas); Blueprint v3; paper 2601.19562; prompt del agente de código |
| [`notas/rl_agents.md`](notas/rl_agents.md) | Agentes RL para Pokémon (Pleines, PokeRL/puffer, Whidden, PokeGym, PokeRL 2026, Hamburg, JEPA) y síntesis de sus vicios |
| [`notas/llm_agents.md`](notas/llm_agents.md) | Agentes LLM/VLM (Claude, Gemini, GPT, Fable 5 y su réplica, Continual Harness, PokéAgent) y lo que aporta el harness |
| [`notas/benchmarks_battle.md`](notas/benchmarks_battle.md) | PokéAgent, Metamon, PokéChamp, PokéLLMon, VGC-Bench, Foul Play, Jaxcalibur, benchmarks; desajuste entre Showdown y el juego |
| [`notas/generalist_and_methods.md`](notas/generalist_and_methods.md) | VPT, SIMA, Genie/LAPO, NitroGen, Game-TARS/Lumine, world models, DT/xLSTM, RND/E3B/Go-Explore, VLM+RL; stack recomendado |
| [`notas/infra_and_critiques.md`](notas/infra_and_critiques.md) | PyBoy, mGBA, NDS, stable-retro, pret, structs por generación, propuesta de Game Abstraction Layer; críticas de la comunidad |

## Hechos clave (para tenerlos a mano)

- **Repositorio**:
  - Tests: 451 passed y 1 skipped; `mypy --strict` da 1 error en 99 archivos.
  - Medallas: unas 3 como máximo histórico por juego; la sesión actual va en 1/8 en FireRed y 2/8 en Emerald.
  - Campañas detenidas desde el 17-nov.
  - Throughput mediano: 33 fps en Emerald y 241 fps en FireRed.
  - La red de combate pesa 0.05 en la decisión (`core/orchestrator.py:1001`).
- **DirectML** funciona, con un speed-up de unas 6 veces sobre la CPU. La conclusión de que "DirectML falló" venía de un assert de autograd (`device_ready_queues_`) que aparece al ejecutar un `backward()` en CPU antes del primero en DML. Ver `PLAN_IPD434.md`.
- **Estado del arte**:
  - Ningún agente juega más de un Pokémon.
  - Ningún RL vence Red con todos los scripts apagados.
  - Los LLM terminan juegos con harness pesado. Sin harness completan ~0 % (PokéAgent).
  - El ganador de PokéAgent T2 (Heatz) usa el LLM como profesor de scripts y destila a una red neuronal con RL.

## Convenciones

- Idioma: español. Las citas al repositorio usan el formato `ruta:línea`, relativo a la raíz `ares-pkmn/`.
- No se inventan cifras: todo número sale de `notas/`, y lo que no se pudo verificar se marca como [NV].
- [LP] en `VICIOS_Y_SOLUCIONES.md` significa *largo plazo*: es diseño, no evidencia.
