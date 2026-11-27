# Estado del repositorio ARES-PKMN (auditoría 2026-11-24)

> Documento de diagnóstico para el proyecto IPD434 (ARES-U). Todo lo que aparece aquí sale de auditorías de solo lectura del repositorio (HEAD `1031f3a "WIP"`, 17-nov-2026), registradas en `notas/repo_decision_loop_bridge.md`, `notas/repo_neural_stadium.md`, `notas/repo_health.md` y `notas/course_and_local_docs.md`. Las referencias `ruta:línea` apuntan al código de `G:\ARES\ares-pkmn`. Si una cifra no se pudo comprobar, se indica con **[NV]**.

## 0. Resumen en cinco líneas

1. ARES-PKMN es un **sistema neuro-simbólico específico de Gen 3**: solo juega FireRed y Emerald, observa **RAM simbólica** (no píxeles) y su conducta la deciden casi por completo **reglas, búsqueda y una campaña escrita a mano**.
2. Los cuatro módulos neuronales (H-JEPA, GFlowNet, "Mamba-2 + Entity Transformer", PPO/PSRO) están bien implementados en su forma, pero en la campaña **pesan muy poco o nada**. Por ejemplo, la red de combate entra en el puntaje con un peso de 0.05 (`core/orchestrator.py:1001`).
3. Los tests están en verde (**451 passed, 1 skipped**; `mypy --strict` con 1 error en 99 archivos), pero eso mide la **corrección del software**, no la capacidad de jugar.
4. Progreso real: el **máximo histórico es de unas 3 medallas por juego**. La sesión actual, que empezó desde cero el 17-nov, va en **1/8 en FireRed y 2/8 en Emerald**, con atascos repetidos que contienen el watchdog y el rewind, no el aprendizaje. Las campañas están **detenidas desde el 17-nov**, aunque `status.json` todavía dice `running: true`.
5. Frente a la meta de "jugar cualquier Pokémon", el **techo de generalización es bajo**: un juego sin decompilación pret ni símbolos es inobservable para el agente. Lo que más sirve para el curso es el **Stadium** (un motor de combate Gen 3 determinista) y el **bridge mGBA** como entorno rápido.

---

## 1. Configuración del entorno

| Aspecto | Valor | Evidencia |
|---|---|---|
| Versión del paquete | 3.0.0 | `pyproject.toml:7` |
| Python | `>=3.12,<3.13` | `pyproject.toml:9` |
| Dependencias | `numpy==2.0.2`, `torch==2.4.1`, scipy, PyYAML, z3-solver, pygame | `pyproject.toml:10` |
| Extra GPU | `torch-directml==0.2.5.dev240914`, torchvision 0.19.1 | `pyproject.toml:14`, `requirements.lock:30-31` |
| CPU / RAM | Ryzen 7 5700X3D (8C/16T), 23.9 GB | `notas/repo_health.md` §2 |
| GPU | AMD Radeon RX 6700 XT, 12 GB. **Sin CUDA**; solo se puede usar con DirectML | `notas/repo_health.md` §2 |
| Último entrenamiento | `"backend": "cpu"` | `artifacts/checkpoints/training_metrics.json` |
| Historial git | 6 commits; HEAD `1031f3a WIP` (18 archivos, +1573/−409) | `notas/repo_neural_stadium.md` §2.7 |

### 1.1 Hallazgo DirectML (corrige una conclusión anterior)

La documentación operativa trata DirectML como algo que "cae a CPU" (`docs/PROBLEMAS.md:62-64`), y la conclusión que se arrastraba era que **"DirectML no funciona"**. El 2026-11-24 se verificó que esa conclusión es **falsa** (`PLAN_IPD434.md`, "Nota DirectML"):

- torch-directml **sí funciona** en la RX 6700 XT, es unas **6× más rápido que la CPU** y da los mismos resultados que la CPU con una tolerancia de **1e-6**.
- El error que se veía antes era un *assert* interno de autograd de PyTorch (`device_ready_queues_`). Aparece **cuando se ejecuta un `backward()` en CPU dentro del mismo proceso antes del primer `backward()` en DML**. No es un problema de cobertura de operadores de la GPU.
- Hay tres condiciones para que funcione: (1) no ejecutar ningún `backward()` en CPU antes del primero en DML dentro del mismo proceso; (2) no usar `nn.TransformerEncoderLayer(norm_first=True)`, porque un bloque pre-norm escrito a mano sí funciona; (3) usar `Adam(..., foreach=False)`.
- Consecuencia de diseño para E1: el *learner* corre en DML en su propio proceso, y los rollouts corren en CPU con varios procesos.

---

## 2. Arquitectura

```
ROM GBA (FireRed / Emerald, SHA1 fijos)
   │
native/ares_mgba.c (shim C, 159 líneas) ── libmgba
   │  ares_create/step/set_keys/read/write/save_state/load_state/pixels/battery
   ▼
core/bridge/ffi_mgba.py  (NativeLibMGBABridge; tablas de direcciones por juego hardcodeadas)
   │  + config/{firered_v1_usa,emerald_usa}.yaml (símbolos pret + offsets de structs)
   │  + data/pokefirered.sym / pokeemerald.sym (~40 símbolos internos de menús/tareas)
   ▼
core/observations.py  (vector de mundo de 96 dims, encode_battle)
   ▼
core/orchestrator.py  (bucle por frame; prioridades fijas)
   ├── combate: menu FSM + choose_battle_decision (EFE causal + daño analítico + red 0.05)
   ├── overworld: HTN (milestones a mano) → Dijkstra de warps → A* de tiles → movement FSM
   ├── field_controls / menus / shopping / puzzles (FSMs por símbolo de decomp)
   └── progress_watchdog + escalera _recover + rewind a checkpoint
   ▼
scripts/run_campaign.py + supervisor (reinicios ante CampaignStalled)

core/stadium/*  (aparte: motor de combate Gen-3 determinista + coevolución GAME/QD)
```

### 2.1 Fuente de observación: RAM simbólica, no píxeles

- Las direcciones de cada juego están **hardcodeadas** en `core/bridge/ffi_mgba.py:54-64`: 16 símbolos por juego (gSaveBlock1Ptr, gPlayerParty, gBattleMons, gObjectEvents, gMain, gTasks…). El lock de script también está fijo (`ffi_mgba.py:350`). Hay 40 literales `0x02/0x03xxxxxx` en `ffi_mgba.py` y 30 en `synthetic_gba_silicon.py`.
- `config/*.yaml` (58 líneas cada uno) contiene símbolos del decomp pret en un commit fijo y offsets de structs (`flags_offset 3808`, `vars_offset 4096`, party de 100 B, battle mon de 88 B, object event de 36 B).
- `framebuffer()` existe (`ffi_mgba.py:472-476`), pero **ninguna política consume píxeles**: solo se usa para screenshots y el visor (`scripts/run_campaign.py:127,152`), para el hash de save_state (`ffi_mgba.py:486`) y en tests.
- El código de menús y de campo lee alrededor de **40 símbolos internos** del decomp (sMenu, sStartMenuOrder, sNamingScreen, gTasks por nombre de función…), por ejemplo en `core/field_controls.py:135-842`.
- `core/game_data.py:20-28` define `Game = Literal["firered","emerald"]` con 18 tipos Gen 3. `core/observations.py:52,59-88,95` normaliza las especies dividiendo por **411**, una constante de Gen 3.

### 2.2 Traza del bucle de decisión (1 tick = 1 frame)

1. `step_vblank_cycle` (`core/orchestrator.py:2973-3017`) llama a `_select_frame_input()`. Cada 30 frames hace `_observe_safely()` (watchdog y milestones), luego `bridge.step_frame()` y `_track_map_transition`, y guarda un checkpoint cada 600 s.
2. `_select_frame_input` (`orchestrator.py:2380-2460`) aplica una **prioridad fija**: teclas de prueba de recuperación → bootstrap de UI (`NativeUIController`) → evolución (`field.scene_keys`) → *disclosures* → `bridge.is_in_battle()` (leído de RAM) → combate u overworld.
3. **Combate** (`orchestrator.py:2324-2378`): `menu.observe_menu()` (una FSM de cursor que lee RAM) → `_decide_battle` → `choose_battle_decision` (`:950`) → `_request_battle_command` → `menu.tick()`, que devuelve una máscara de 10 bits → `set_hardware_keys`.
4. **Overworld** (`orchestrator.py:1822-2010`), en este orden: pushback, *objective talk* y tienda → `recovery_key` de menús → interceptar NPC que patrulla → `movement.tick()` (FSM de pasos, `core/navigation/movement_fsm.py`) → operaciones de menú de campo → si el hardware no está listo, pulsa A cada 12 frames → configurar Options → `planner.get_active_directive()` (HTN, `core/planner/htn_planner.py:732`) → ruta con Dijkstra sobre el grafo de warps (`warp_topology.py`) y A* de tiles (`constrained_astar.py`) → el world model elige caminar o correr → la movement FSM pulsa la dirección.
5. `_handle_fault` (`orchestrator.py:913-931`): si la política lanza cualquier excepción, se contiene en ese frame y se pulsa A.

### 2.3 Qué son reglas y qué es aprendido

| Decisión | Mecanismo real | ¿Aprendido? | Evidencia |
|---|---|---|---|
| Qué objetivo perseguir | HTN con **45 milestones (FR) y 56 (EM)** escritos a mano, cada uno con mapa, regex de script, flag/var de completitud, nivel, tipos y HMs | No | `htn_planner.py:133-183`, `:184-246` |
| Cómo llegar | Dijkstra de warps + A* con física de tiles Gen 3 | No | `warp_topology.py`, `constrained_astar.py` |
| Caminar o correr | H-JEPA elige entre 2 candidatos sobre una ruta ya calculada; con Running Shoes se fuerza `selection = 1` | Casi nulo | `orchestrator.py:1425-1447`, `:1440-1444` |
| Qué movimiento usar en combate | `score = efe - expected_damage/enemy_max_hp*4 - logits*0.05`; cambio `1.0 - logits*0.05`; objetivo `*0.02` | La red solo desempata | `orchestrator.py:1001`, `:1002`, `:1031` |
| Qué especies capturar | GFlowNet (TB) con recompensa heurística de cobertura; `desired_moves` solo se registra | Sesgo débil | `orchestrator.py:727-765`, `:1043`, `:1120`, `gflownet_team.py:100-108` |
| Menús, diálogos, puzzles | FSMs por símbolo; solvers de Strength y del hielo de Sootopolis | No | `core/navigation/puzzles.py:30,110,199` |
| Requisitos para avanzar | Ejecución simbólica de los scripts `.inc` del decomp | No (análisis estático) | `script_gates.py:1-16` |
| Salir de atascos | Watchdog, escalera de recuperación, tabú y rewind | No | `core/progress_watchdog.py:44-62`, `orchestrator.py:2631-2802` |

### 2.4 El rol real de los módulos neuronales

| Módulo | Implementación | Datos de entrenamiento | Rol en la campaña | Veredicto |
|---|---|---|---|---|
| "H-JEPA" (`core/world_model/hjepa.py`) | Encoder MLP 96→128→128 con target EMA (0.996), predictor de 256, pérdida estilo VICReg y multi-horizonte (1, 2, 4); unos 60k parámetros (`hjepa.py:19-33,63-114`) | Transiciones de un **`SyntheticGBASiliconBridge`** con mapas sintéticos de 12×14 y acciones aleatorias (`core/training_data.py:16-59`). **No usa datos de la ROM** | Elige caminar o correr, y queda sobrescrito con Running Shoes | **Decorativo**. Además, no es jerárquico: el "H" no corresponde a nada en el código |
| GFlowNet de equipo (`core/planner/gflownet_team.py`) | MLP sobre 411 especies y 355 movimientos, Trajectory Balance con PB uniforme (`:33-45,94-98`) | Recompensa heurística `exp(#tipos/6 + cobertura − riesgo)`, no resultados de combate | Sesga qué capturar | Influencia débil y sin medir |
| "Mamba-2 + Entity Transformer" (`core/battle/mamba_entity_net.py`) | Encoder de entidades 48→128, TransformerEncoder de 3 capas, 4 heads y d=128, más un `SelectiveSSMBlock` SSD portable (`:34-167`); 532,837 parámetros | PPO en una arena de juguete (ver abajo) | Desempate con peso 0.05 | **Prácticamente decorativo**. El SSM se aplica a un token por turno (`:197`), así que en la práctica es una RNN |
| PPO/PSRO (`core/battle/psro_league_trainer.py`) | PPO+GAE correcto en su forma, PSRO con *fictitious play* (`:207-387`) | `BatchedIntegerBattleArena` (`:68-172`): 1v1, HP=100, 4 movimientos con potencias {20,40,80,120}, **sin tipos, sin cambios y sin estados**. La política óptima es simplemente "elige la mayor potencia" | Ninguno directo | Problema trivial: el win rate de 1.0 no prueba nada |

Entrenamiento (`scripts/train_neural_modules.py`): unos **80 s** en total, **1 semilla (42)**, sin líneas base. `training.log` muestra un win rate de 0.39 → 0.625, VICReg de 63.6 → 39.3 y TB de 3662 → 180. **Estos números no deben usarse como resultado del informe** (`notas/repo_neural_stadium.md` §1.4-1.5).

En la telemetría de campaña aparecen `neural_decisions` (194 en FR y 495 en EM) y `proved_decisions = 0` en ambos juegos. Las "decisiones neuronales" son decisiones de combate en las que la red apenas desempata.

---

## 3. El Stadium (`core/stadium/*`), el activo más útil

- **Motor** (`engine.py`, 1474 líneas): singles 6v6 deterministas por semilla (`engine.py:261-299`), con `clone()` para contrafactuales (`:295`). El daño siempre pasa por `GbaIntegerSafetyVerifier.compute_context_damage_reference`, con roll 85-100 (`engine.py:856,1351`). La tabla de tipos es la de la ROM. Hay unas 124 ramas de efectos, más habilidades, estados, clima, pantallas, Púas, Sustituto, Protección (`:1214`) y críticos (`:929`).
- **Límites declarados**: no implementa Metrónomo, Transformación ni Relevo; en esos casos emite un evento `unsupported` que se cuenta (`engine.py:687-688`). Tampoco hay objetos de mochila. **No se ha validado combate a combate contra el emulador** y no es Showdown. En el informe hay que describirlo como un "simulador Gen-3 aproximado con daño entero exacto".
- **Dex**: `data/stadium_dex.json`, exportado de la ROM, con 412 especies y sus learnsets (`dex.py:66`; `PLAN_IPD434.md` E1).
- **Políticas**: `MLPGenome` (25 features → 16 → 8 → 1, 561 parámetros, `policies.py:35,53-95`) con el prior experto embebido (`policies.py:36-39,110-128`); `RandomPolicy` (`:172`), `GreedyDamagePolicy` (`:183-199`), `NeuralPolicy` (`:202-229`, 532,837 parámetros) y `HybridPolicy` (`:232-252`). Las features por opción (`features.py:27-30`) incorporan mucho conocimiento experto.
- **Coevolución**: GAME/MTMB-ME (Anne et al., arXiv 2601.19562) con archivo creciente y selección por Ranking, Pareto, Behavior o Random (`game_qd.py:50-103,208-212,230,294-295`); lecciones contrafactuales con CRN (`learning.py:60-188`); un canal profundo BC → PPO → destilación (`deep.py:38-211`).
- **Velocidad medida**: con MLP, greedy o random, entre unos 130 y 250 combates/s por proceso; con `NeuralPolicy` de 533k parámetros en CPU, unos 5.7 combates/s (entre 25 y 40 veces más lento). La liga histórica jugó 520,187 combates en 5531 s.
- **Líneas base medidas** (n=150): greedy vs random 0.713; prior-MLP vs greedy 0.42; random vs random 0.467. Random es un baseline muy débil, y **greedy es la referencia relevante**.
- **Resultados previos**: en `compare-20261027-234444`, Ranking y Pareto superan a Behavior y Random (Mann-Whitney con Holm). En `ablation-20261028-002210`, que comparó la versión completa contra GAME en 5 réplicas, **ninguna diferencia fue significativa**: las extensiones propias no mejoran de forma demostrable. En la liga larga, la ventaja contra el ancla greedy no crece (ROJO 0.91 → 0.86).
- **Tests**: `tests/test_stadium.py` (24) y `tests/test_stadium_rules.py` (8) pasan los 32 en 17 s.

---

## 4. Salud del software

| Indicador | Valor | Evidencia |
|---|---|---|
| pytest `-q -x` | **451 passed, 1 skipped** en 256.9 s | `notas/repo_health.md` §3 |
| Skip | `tests/test_navigation_recovery.py:84`: "Campaign has moved beyond the recorded east MtMoon B2F section". Ese test depende del estado vivo de `artifacts/`, así que **no es reproducible en un clon limpio** | idem |
| mypy `--strict core scripts tests` | **1 error en 99 archivos**: `tests/test_stadium.py:69` (`RomCatalog` recibe `str` y espera `Literal['firered','emerald']`) | idem |
| Archivos de test | 27 entradas, unas 285 funciones `def test_` | idem |

**Lectura honesta.** El prompt fundacional del agente de código (`gemini-code-1791938796398.xml`) daba el trabajo por terminado cuando *pytest + mypy + demo de verificación* devolvían exit 0 (`notas/course_and_local_docs.md` D3). Ese criterio es **verificación de software**, no evidencia experimental de aprendizaje, y el curso advierte explícitamente que no basta (`00_Introduccion.md:211-213`).

---

## 5. Progreso de las campañas

### 5.1 Estado actual (detenido)

- **Las campañas no están corriendo.** El último heartbeat es de epoch 1794890152, es decir, **2026-11-17 01:35**. `status.json` todavía dice `"running": true` (pids 22980 y 16912), lo que indica un bug de apagado o una muerte abrupta del proceso.
- La sesión del 17-nov **empezó una partida nueva desde cero** (`artifacts/processes/latest.json`: acción `run`, `checkpoint: null`).

| | FireRed | Emerald |
|---|---|---|
| Frames de la sesión | 200,000 | 580,000 |
| **Medallas en la sesión** | **1/8** (Brock) | **2/8** (Roxanne, Brawly) |
| Directiva actual | MT_MOON_FOSSIL_GUARD | GYM_3_WATTSON (puzzle `mauville_switches`) |
| Combates / victorias / derrotas | 102 / 77 / 0 | 313 / 284 / 1 |
| Capturas / desmayos | 5 / 1 | 7 / 15 |
| Equipo | Ivysaur Lv19 más 4 Pokémon de nivel 3-8 (depende de un solo Pokémon) | — |
| **Máximo histórico** | **unas 3 medallas** (hasta GYM_3_SURGE / SURGE_SWITCHES; luego bloqueado en CELADON_TEA) | **unas 3 medallas** (GYM_3_WATTSON, HM_ROCK_SMASH y METEOR_FALLS_TEAMS, observados una vez cada uno) |
| Milestones únicos en el log histórico | 14 de 45 | 23 de 56 (el `status.json` actual lista 20) |

Ninguna campaña se ha completado.

### 5.2 Atascos (stalls)

- **Límites del watchdog** (`core/progress_watchdog.py:44-62`): campo 600 frames, menú 1200, combate 2400, bootstrap 3600; `cycle_limit` de 14,400 y `strategic_limit` de 72,000 frames sin nuevos hechos de historia.
- **Escalera `_recover`** (`orchestrator.py:2631-2700+`): strategy_shift → release_and_replan → cancelar la operación de campo → cerrar menú → secuencias de teclas de prueba → desvío de exploración de 900 frames (`:1768-1787`) más tabú de transiciones (`:1491-1528`) → rewind a checkpoint, hasta `max_rewinds=6` (`:183`, `:2802`) → `CampaignStalled` → el supervisor reinicia.
- **Eventos registrados** (`events.jsonl` histórico):
  - FR: 338 `stall_detected`, 310 `stall_recovery`, 2,367 `campaign_blocked` (3,902 menciones de "Strongest campaign Pokemon has low HP" y 357 de "live collision path unavailable"), 40 `oscillation_detected`, 226 `transition_tabu`, 180 `exploration_detour`.
  - EM: 720 `stall_detected`, 649 `stall_recovery` (375 estacionarios y **301 `repeated_state_cycle`**), 2,911 `campaign_blocked` (1,964 en PETALBURG_WOODS_AQUA), 63 `oscillation_detected`.
- **Supervisor**:
  - FR: 21 arranques, 13 reinicios. `CampaignStalled` se repite en MT_MOON_FOSSIL_GUARD (11 veces o más), CERULEAN_RIVAL (10 o más) y CELADON_TEA (7).
  - EM: 15 arranques, 9 reinicios. La causa dominante es STERN_GOODS (17 o más).
- **Lecciones** (`lessons.json`): 17 derrotas contra el rival de Celeste (`retry_level 30`), un muro que el agente resuelve **grindeando**.
- **Cómo se ha avanzado**: `docs/RECUPERACION_AUTONOMA.md` (145 líneas) cataloga bloqueos que se **reprodujeron con savestates y se corrigieron a mano**: evolución, olvidar movimientos, árboles de Corte fuera de cámara, secciones de Mt. Moon, la patrulla de Briney, los guardias del té, la preparación contra Brawly… El progreso salió de **parches específicos de cada mapa**, no de generalización aprendida.

### 5.3 Throughput

Medido en `scripts/run_campaign.py:184-190`, con el agente completo más el emulador en 1 proceso:

| | Media | p10 | **p50** | p90 | Máx |
|---|---|---|---|---|---|
| FireRed (n=4219) | 228 | 96 | **241** | 330 | 1013 |
| Emerald (n=6032) | 115 | 0.5 | **33** | 369 | 1335 |

La mediana va de **33 a 241 fps**, es decir, entre 0.5× y 4× tiempo real (la GBA corre a 59.7 fps). El p10 de unos 0.5 fps en Emerald viene de la planificación de rutas por frame ("can consume seconds per VBlank", `orchestrator.py:1930-1932`). **No hay un benchmark del emulador crudo sin el orquestador [NV]**. Medirlo es requisito para E2, porque el RL necesita millones de pasos.

---

## 6. Acoplamiento a Gen 3

| Tipo de acoplamiento | Magnitud | Ejemplos |
|---|---|---|
| Ramas condicionadas al juego (`is_emerald`, `== "firered"`) | **unas 90 en `core/`** | field_controls 18, ffi_mgba 14, htn_planner 10, synthetic 8, orchestrator 7, constrained_astar 7 (spin tiles de FR, `:30-31,193,218`), menu_cursor 6, warp_topology 5, source_data 5, shopping 3 |
| Literales de memoria | 40 en `ffi_mgba.py` y 30 en `synthetic_gba_silicon.py` | `ffi_mgba.py:54-64`, `:350` |
| Valores mágicos por juego | — | Índices de HM (`field_controls.py:442-445`), Start menu (`:219`, `:619-620`), summary `0x3289 if firered else 0x40C7` (`:644`), item 364 de FR (`:835`) |
| Identificadores de historia | 30 en `htn_planner` y 19 en `stadium/league.py` | líderes, rival, Rocket |
| Campaña | 101 milestones escritos a mano; 27 con `puzzle=` (14 tipos) | `htn_planner.py:133-246` |
| Constantes de generación | 411 especies, 18 tipos, structs de 100/88/36 B cifradas | `observations.py:52,95`, `game_data.py:20-28` |
| Dependencias externas | Decomp pret con `.sym`, scripts `.inc` y JSON de mapas | `script_gates.py`, `source_data.py` |

**Esfuerzo de extensión** (`notas/repo_decision_loop_bridge.md` §8-9):

- **Ruby/Sapphire y LeafGreen** son la extensión más barata (mismo motor y decomp parecido).
- **Gen 1/2 con PyBoy**: habría que escribir un backend nuevo que cumpla `GBABridgeProtocol` (`ffi_mgba.py:130-187`, unos 30 métodos). Las structs son otras (sin cifrado y sin ASLR), los mapas usan metatiles de 2×2, hay 15/17 tipos y los scripts están en ensamblador SM83, así que `script_gates.py` no aplica. Las FSMs y la campaña habría que reescribirlas. Son **semanas o meses**.
- **Gen 4 (NDS)**: no hay emulador embebible en el shim, los decomps están incompletos y el hardware tiene doble pantalla táctil. Son **meses**.

---

## 7. Visión original frente a lo implementado

El *Blueprint Maestro v3* (`ARES_PKMN_Master_Architecture_Blueprint_v3.pdf`, p.1-4) define un sistema **para Gen 3**: Inferencia Activa (EFE), SCM con do-calculus, H-JEPA, GFlowNet, Entity-Transformer más Mamba-2, Z3/SMT, HTN y A* 3D sobre libmgba. En §1.1 descarta "PPO/DQN directo" por la escasez de recompensas. Lo implementado cumple esa visión **en su forma** (todos los módulos existen), pero:

- la capa 1 depende por completo de la ROM, lo que contradice la meta actual de "cualquier Pokémon";
- terminó incluyendo PPO y PSRO, que el Blueprint había descartado, mientras que **DQN, el algoritmo canónico del curso, aparece en 0 archivos**;
- muchos conceptos usados no están en el curso (PPO, PSRO, JEPA/VICReg, GFlowNet, Mamba, EFE, SCM, Z3, HTN) y habría que explicarlos todos en un informe de 5 páginas (`notas/course_and_local_docs.md` §C).

---

## 8. Diagnóstico honesto frente a la meta "jugar cualquier Pokémon"

| Requisito de la meta | Estado en ARES-PKMN | Brecha |
|---|---|---|
| Percibir cualquier juego | Solo RAM de Gen 3 vía símbolos pret; los píxeles no se usan | **Crítica** |
| Actuar con una interfaz común | Teclas de hardware GBA, pero producidas por FSMs ligadas a símbolos | Alta |
| Decidir qué hacer | Campaña escrita a mano (101 milestones) | **Crítica** |
| Aprender de la experiencia | Las redes están entrenadas con datos sintéticos o de juguete y apenas influyen en la campaña | **Crítica** |
| Salir de atascos | Watchdog, rewind y parches a mano por mapa | Alta |
| Evidencia experimental | Tests verdes; 1 semilla; sin líneas base en la campaña; la ablación del Stadium no es significativa | Alta |
| Progreso | Máximo de unas 3/8 medallas por juego; ninguna campaña completa | Alta |

**Conclusión.** ARES-PKMN es un buen **harness de ingeniería para Gen 3**, con piezas valiosas: un bridge mGBA determinista con savestates, el Stadium con daño exacto, un estimador analítico de daño que sirve como baseline fuerte y una estadística ya escrita (Mann-Whitney con Holm). Pero **no es un agente que aprenda a jugar**, y menos uno que generalice entre juegos. Por eso ARES-U no intenta "extender la campaña". Reusa el Stadium y el bridge como entornos y demuestra la pieza que la visión necesita: **representaciones independientes del juego que transfieren a contenido no visto** (E1: combate Gen1 → Gen2/Gen3 sin IDs; E2: píxeles FireRed → Emerald). Ver `PLAN_IPD434.md` y `VICIOS_Y_SOLUCIONES.md`.

### 8.1 Activos reutilizables para ARES-U

| Activo | Uso en ARES-U |
|---|---|
| `core/stadium/engine.py` + `data/stadium_dex.json` (412 especies con learnsets) | Entorno de E1 |
| `GreedyDamagePolicy`, `MLPGenome.prior`, `RandomPolicy` | Líneas base de E1 |
| Harness estadístico del Stadium (workers, Mann-Whitney, Holm) | Protocolo de E1 |
| `NativeLibMGBABridge` (step_frame, set_hardware_keys, read_memory, save/load_state, framebuffer) | Entorno crudo de E2 (píxeles, con RAM solo para medir) |
| Métricas de logs (stalls, milestones por frame) | Contraste con el agente aprendido |
