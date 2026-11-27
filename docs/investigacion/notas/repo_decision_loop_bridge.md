# Auditoria: bridge + decision loop (ARES-PKMN)

Estado: COMPLETO (auditoria basada en evidencia, path:line). No se modifico codigo ni se lanzaron campañas.

## 1. Fuente de observacion: RAM simbolica, NO pixeles
- Shim nativo `native/ares_mgba.c` (159 lineas): API ares_create/step/set_keys/read/write/save_state/load_state/pixels/battery (`native/ares_mgba.c:41-155`). `ares_read`/`ares_write` validan spans EWRAM/IWRAM (`:85`).
- `core/bridge/ffi_mgba.py`: tablas de direcciones por juego hardcodeadas `ffi_mgba.py:54-64` (FireRed SHA1 41cb23..., Emerald f3ae08...; 16 simbolos c/u: gSaveBlock1Ptr, gPlayerParty, gBattleMons, gObjectEvents, gMain, gTasks...). Lock de script hardcodeado `ffi_mgba.py:350` (0x03000F2C / 0x03000F9C). 40 literales 0x02/0x03xxxxxx en ffi_mgba.py, 30 en synthetic_gba_silicon.py.
- `config/firered_v1_usa.yaml`, `config/emerald_usa.yaml` (58 lineas c/u): direcciones de simbolos del decomp pret (commit fijado) + offsets de struct (flags_offset 3808, vars_offset 4096, party 100 B, battle mon 88 B, object event 36 B).
- `framebuffer()` existe (`ffi_mgba.py:472-476`) pero solo se usa para screenshots/visor (`scripts/run_campaign.py:127,152`), hash de save_state (`ffi_mgba.py:486`) y tests. NINGUNA politica consume pixeles.
- Ademas, el codigo de menus/campo lee ~40 simbolos internos del decomp (sMenu, sStartMenuOrder, sNamingScreen, gTasks por nombre de funcion, gBattleCommunication...) via `data/pokefirered.sym` / `pokeemerald.sym` (p.ej. `core/field_controls.py:135-842`).
- `core/game_data.py:20-28`: `Game = Literal["firered","emerald"]`, SHA1 y nombres de ROM fijos; 18 tipos Gen3; catalogo de especies/moves leido de la ROM con offsets de simbolos.
- `core/observations.py:59-88`: vector de mundo de 96 dims (coords, 6x10 party, colision vecindario, inventario 12 items fijos, flags de batalla); especies normalizadas por 411 (constante Gen3, `:52`, `:95`).

## 2. Traza de un tick (1 tick = 1 frame)
1. `step_vblank_cycle` (`core/orchestrator.py:2973-3017`): `_select_frame_input()`; cada 30 frames `_observe_safely()` (watchdog/milestones); `bridge.step_frame()`; `_track_map_transition`; checkpoint cada 600 s.
2. `_select_frame_input` (`orchestrator.py:2380-2460`), prioridad fija:
   probe keys de recuperacion -> bootstrap UI (`NativeUIController`) -> evolucion (`field.scene_keys`) -> disclosures -> `bridge.is_in_battle()` (RAM) -> combate u overworld.
3. Combate (`orchestrator.py:2324-2378`): `menu.observe_menu()` (FSM de cursor leyendo RAM) -> `_decide_battle` -> `choose_battle_decision` (`:950`) -> `_request_battle_command` -> `menu.tick()` devuelve mascara de 10 bits -> `set_hardware_keys`.
4. Overworld (`orchestrator.py:1822-2010`): pushback/objective talk/mart -> recovery_key de menus -> interceptar NPC patrullando -> `movement.tick()` (FSM de pasos, `core/navigation/movement_fsm.py`) -> field menu ops (`field.poll_key_mask`) -> si `!is_hardware_ready` pulsa A cada 12 frames -> Options setup -> `planner.get_active_directive()` (HTN, `core/planner/htn_planner.py:732`) -> ruta por warp graph (Dijkstra, `warp_topology.py`) + A* de tiles (`constrained_astar.py`) -> world model elige walk/run -> movement FSM pulsa direccion.

## 3. Aprendido vs hand-coded
- Batalla: `AresBattlePolicyNetwork.forward_turn` (Mamba/SSM, `orchestrator.py:972`) se llama, pero el score final es
  `score = efe - expected_damage/enemy_max_hp*4 - float(logits[0, slot])*0.05` (`orchestrator.py:1001`), switch `1.0 - logits*0.05` (`:1031`), target `*0.02` (`:1002`). Dominan: calculo de dano analitico, EFE causal (`self.causal.evaluate_free_energy`), `estimate_attack_plan` (horizonte de riesgo), filtro de inmunidades observadas, items de cura. La red es un tie-breaker.
- World model H-JEPA (`orchestrator.py:1425-1447`): solo elige walk vs run sobre una ruta ya calculada por A*, y se sobreescribe a run si hay Running Shoes (`:1440-1444`). Se entrena online (`:1414-1420`) pero su efecto es casi nulo.
- GFlowNet de equipo (`orchestrator.py:218`, `:738`): muestrea composicion de equipo (411 especies fijas).
- Pesos se cargan de `artifacts/checkpoints/neural.pt` (evento `neural_weights_loaded` en logs).
- Todo lo demas es reglas/busqueda: HTN con milestones escritos a mano, Dijkstra de warps, A* con fisica de tiles Gen3, FSMs de menu/movimiento, analisis simbolico de scripts pret (`script_gates.py`), solvers de puzzles (strength, hielo Sootopolis: `core/navigation/puzzles.py:30,110,199`), watchdog.

## 4. Hardcoding por juego / casos especiales (conteos aproximados)
- Campaña: `htn_planner.py:133-183` FireRed = 45 milestones; `:184-246` Emerald = 56 milestones, cada uno con nombre de mapa, regex de script, flag/var de completitud, nivel requerido, tipos, HMs. 27 milestones con `puzzle=` (14 tipos de puzzle etiquetados, p.ej. rocket_spinners, cinnabar_quizzes, mansion_statues). `data/campaign_*.json` es el export de esa lista (`export_campaign`, `htn_planner.py:808`).
- Ramas por juego (`is_emerald`, `== "firered"`, etc.): field_controls 18, ffi_mgba 14, htn_planner 10, synthetic 8, orchestrator 7, constrained_astar 7 (spin tiles FireRed `constrained_astar.py:30-31,193,218`), menu_cursor_controller 6, warp_topology 5, source_data 5, shopping 3. Total ~90 en core.
- Ejemplos: indices de HM por juego (`field_controls.py:442-445`), indices de Start menu (`:219`, `:619-620`), span de summary screen `0x3289 if firered else 0x40C7` (`:644`), item 364 FireRed (`:835`).
- Identificadores de historia (gym leaders, rival, Rocket, etc.): 30 en htn_planner, 19 en stadium/league.py, pocos en el resto.
- Lo general (dentro de Gen3 pret): `script_gates.py` hace ejecucion simbolica de los scripts .inc del decomp (`script_gates.py:1-16`) para deducir requisitos de paso; `source_data.py` compila mapas/warps desde JSON del decomp. Es elegante pero requiere el decomp fuente del juego.

## 5. Manejo de atascos
- `core/progress_watchdog.py:44-62`: limites estacionarios field 600, menu 1200, battle 2400, bootstrap 3600 frames; cycle_limit 14400; strategic_limit 72000 frames sin hechos de historia nuevos.
- Escalera `_recover` (`orchestrator.py:2631-2700+`): strategy_shift (si objective_no_progress) -> release_and_replan -> cancelar operacion de campo -> dismiss menu -> pools de teclas probe (secuencias A/B/arriba/abajo) -> detour de exploracion 900 frames (`:1768-1787`) + tabu de transiciones (`:1491-1528`) -> tras attempt>4 rewind a checkpoint (save_state) hasta max_rewinds=6 (`:183`, `:2802`) -> `CampaignStalled` y el supervisor reinicia desde checkpoint verificado mas antiguo.
- `_handle_fault` (`orchestrator.py:913-931`): cualquier excepcion de politica se contiene al frame y pulsa A.
- Evidencia en logs: firered/events.jsonl: 338 stall_detected, 310 stall_recovery, 226 transition_tabu, 180 exploration_detour, 2367 campaign_blocked; emerald: 720 stall_detected.

## 6. Resultados observados en logs (artifacts/campaigns/native/*)
- FireRed: 14 milestones unicos (hasta GYM_3_SURGE; Brock en frame 88130), ultimo frame 202601 (de 45 milestones).
- Emerald: 23 milestones unicos (hasta GYM_3_WATTSON / METEOR_FALLS_TEAMS), ultimo frame 580258 (de 56).
- Ninguna campaña completa.

## 7. Throughput (frames/s)
- Medido en `scripts/run_campaign.py:184-190` (campo `fps` en telemetry.jsonl, agente completo + emulador, 1 proceso).
- FireRed telemetry (n=4219): media 228, p10 96, p50 241, p90 330, max 1013 fps.
- Emerald telemetry (n=6032): media 115, p10 0.5, p50 33, p90 369, max 1335 fps (p10 ~0.5 = planificacion de rutas costosa por frame; comentario en `orchestrator.py:1930-1932` "can consume seconds per VBlank").
- No hay benchmark del emulador "crudo" sin agente. Tiempo real GBA = 59.7 fps, asi que el agente corre ~0.5x-5x tiempo real (mediana ~4x FR, ~0.5x EM). Para RL con millones de pasos esto es bajo; un env limpio (sin orquestador) con mGBA probablemente da miles de fps por core (no medido aqui).

## 8. Techo de generalizacion a juegos no vistos
- Bajo. Todo el pipeline asume: (a) decomp pret con simbolos + scripts .inc + JSON de mapas; (b) structs Gen3 (party 100 B cifrada, battle mon 88 B, save blocks con ASLR); (c) campaña escrita a mano; (d) FSMs de menu por nombre de task/simbolo. Ningun componente percibe pantalla, por lo que un juego sin decomp/simbolos es inobservable.
- Lo transferible conceptualmente: A* con fisica de tiles, Dijkstra de warps, watchdog/rewind, estimador de dano (con tabla de tipos), la idea de analisis simbolico de scripts.
- Ruby/Sapphire (pokeruby existe) seria la extension mas barata; LeafGreen tambien (mismo decomp pokefirered, otros offsets).

## 9. Esfuerzo para Gen1/2 (PyBoy) o Gen4 (NDS)
- Gen1/2 via PyBoy: nuevo backend implementando `GBABridgeProtocol` (`ffi_mgba.py:130-187`, ~30 metodos: party descifrada, battle mon, object events, map matrices, menus). Estructuras Gen1/2 totalmente distintas (sin cifrado, sin ASLR, mapas de 2x2 metatiles, 15/17 tipos, 151/251 especies vs 411 hardcodeado). pokered/pokecrystal tienen .sym pero sus scripts son ensamblador SM83, `script_gates.py` no aplica. FSMs de menu/batalla y campaña: reescritura. Estimacion: semanas a meses; inviable en 1 semana para el agente completo. Viable en 1 semana: un env PyBoy minimo (RAM o pixeles) para un experimento acotado.
- Gen4 NDS: no hay emulador embebible en el shim (mGBA no emula NDS; requeriria melonDS/DeSmuME con bindings), decomps pokeplatinum/pokeheartgold incompletos, structs y pantallas dobles/tactil. Esfuerzo: meses.

## 10. Reutilizable para el proyecto del curso (1 semana, NN + baseline)
- `NativeLibMGBABridge` (step_frame, set_hardware_keys, read_memory, save/load_state, framebuffer) como env rapido y determinista: sirve para un wrapper Gym con observacion de pixeles (240x160) y/o RAM.
- save_state/load_state para resets de episodio desde estados fijos (p.ej. inicio de una batalla o un mapa) y semillas.
- `encode_battle` / `encode_world_state` como features simbolicas; `choose_battle_decision` (estimador analitico) como BASELINE heuristico fuerte contra el cual comparar una NN (p.ej. politica de batalla aprendida, o una politica de pixeles en una sub-tarea de navegacion vs A*).
- Metricas de logs (milestones/frames, stall counts) como protocolo de evaluacion.
