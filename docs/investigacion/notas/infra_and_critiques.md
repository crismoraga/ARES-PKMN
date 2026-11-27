# Infraestructura multi-generación y críticas de la comunidad (ARES-PKMN)

Fecha de investigación: 2026-11-24. Proyecto: IPD434 Soft Computing (UTFSM) — agente neuronal que juegue CUALQUIER Pokémon mainline.
Convención: "(verificado)" = confirmado contra fuente primaria (README / arXiv / código) en esta sesión. "(no verificado)" = de fuente secundaria o memoria; no usar como cifra en el informe sin re-chequear.

---

## Parte A — Infraestructura multi-generación

### A1. PyBoy (GB/GBC) — Gen 1 y Gen 2
- Repo: https://github.com/Baekalfen/PyBoy (~5.2k stars; último push 2026-10-30) (verificado vía `gh api`).
- README (verificado): benchmark "Full rendering" **x625 realtime**, "render every 25th frame" **x970**, "No rendering" **x1014** realtime. El README aclara que 100x realtime = 6.000 fps, así que x1014 ≈ **~60.800 frames/s** por proceso (cálculo derivado). "Your performance will depend on the game."
- API: `set_emulation_speed(0)`, `tick(n, render)`, hooks en direcciones de memoria, acceso a RAM y a la tilemap. Ideal para Red/Blue/Yellow/Gold/Silver/Crystal.
- Medición en Pokémon Red (Pleines et al., 2502.19920, verificado en PDF): **9.403 SPS** con acciones aleatorias en Ryzen 7 2700X, con **1 decisión cada 24 frames** (8 frames presionado + 16 liberado). Comparación del mismo paper: Atari Breakout 7.117 SPS, Procgen 18.530 SPS.
- pokemonred_puffer (docs de drubinstein, verificado en la página): 24 ticks/acción, headless, pico ~**10.000 SPS** tras optimizaciones (flags de compilador, liberar GIL en Cython, hooks, leer RAM con menos frecuencia); obs de pantalla 92 kB → ~2 kB (gris, 72x80, 2-bit). Hardware: 4 máquinas RTX 4090 + i9-14900K + 128 GB.

### A2. mGBA / libmgba (GBA) — Gen 3 (Ruby/Sapphire/Emerald/FireRed/LeafGreen)
- mGBA trae bindings Python oficiales pero **desactivados por defecto**: requieren compilar con `-DBUILD_PYTHON=ON` (verificado en README de pygba). Wheels precompilados solo para Python ≥3.10 en Linux/macOS; en **Windows hay que compilar desde fuente** (relevante: la máquina de desarrollo es Windows).
- **pygba** (https://github.com/dvruette/pygba, 33 stars, push 2025-07): wrapper gymnasium sobre mGBA con wrapper de reward para Pokémon Emerald (verificado).
- **hanzi/libmgba-py** (fork de bindings, 16 stars, push 2024-12): permite instancia "headless" desde Python (verificado README vía búsqueda). Base de **pokebot-gen3** (bot de shiny hunting Gen 3; no verificado en detalle por timeout de API).
- **gbax** (PyPI, v0.16.0): alternativa vía core libretro `mgba_libretro.so` + shim cffi de ~300 LOC; se autodescribe para "speedrun Pokémon Emerald with Claude" (verificado en PyPI; afirmaciones de reproducibilidad son del autor, no verificadas).
- FPS headless de mGBA en Pokémon: **no verificado** (no encontré benchmark primario). Pendiente: benchmark propio.

### A3. Nintendo DS — Gen 4 (Diamond/Pearl/Platinum/HeartGold/SoulSilver) y Gen 5
- **py-desmume** (https://github.com/SkyTemple/py-desmume, 35 stars, push 2025-12): librería Python sobre el frontend "interface" de DeSmuME; wheels para Linux/Windows/macOS (verificado README). Usado por JVP15/BattleTowerAgent (bot scripted para Battle Tower de Platinum; no verificado en repo original).
- **melonDS** (https://github.com/melonDS-emu/melonDS, ~5.2k stars, activo 2026-11): sin bindings Python oficiales encontrados (búsqueda negativa). Core libretro: https://github.com/libretro/melonDS (verificado existe).
- **stable-retro (Farama)** (https://github.com/Farama-Foundation/stable-retro, ~1k stars, push 2026-11-22): README lista soporte **Nintendo DS (melonDS)**, Game Boy/Color (Gambatte) y GBA (mGBA) en Linux/Windows/macOS (verificado). => **Una sola API Gym para GB/GBC/GBA/NDS**, candidato fuerte para la capa de emulación de ARES (perdiendo hooks específicos de PyBoy).
- FPS headless NDS: **no verificado**; NDS es mucho más pesado (2 CPUs ARM, 3D, 2 pantallas 256x192). Hipótesis: 1–2 órdenes de magnitud menos que GB. Hay que medir.

### A4. pret: decompilaciones como fuente de RAM/símbolos
| Repo | Gen/Plataforma | Estado (verificado gh api) |
|---|---|---|
| pret/pokered | Gen1 GB | ~5.0k stars, activo 2026-11-24 (disassembly) |
| pret/pokecrystal | Gen2 GBC | ~2.6k stars, activo 2026-11-16 |
| pret/pokeemerald | Gen3 GBA | ~3.7k stars, activo 2026-11-23 |
| pret/pokefirered | Gen3 GBA | ~1.7k stars, activo 2026-10-20 |
| pret/pokeplatinum | Gen4 NDS | ~0.6k stars; README: "WIP decompilation" |
| pret/pokeheartgold | Gen4 NDS | ~0.8k stars, activo 2026-11-23 |
- Los builds producen archivos `.sym`/`.map` con direcciones de símbolos (p.ej. `wPartyCount`, `wObtainedBadges`, `wCurMap`, `wBattleMon`, `wEnemyMon` en `pokered/ram/wram.asm`, verificado). Esto permite adapters de RAM sin ingeniería inversa manual.
- Riesgo: decomps Gen 4 son WIP; en NDS muchas estructuras viven en heap → direcciones no fijas (punteros base dinámicos) (no verificado formalmente, conocimiento de comunidad).

### A5. Diferencias de estructuras de datos por generación (evidencia en pret)
- **Gen 1/2**: estructuras planas sin cifrar en WRAM (party/battle structs en `wram.asm`). Lectura directa trivial.
- **Gen 3** (pokeemerald `include/pokemon.h`, verificado): `struct BoxPokemon` = personality + otId + nickname + ... + `checksum` + unión `secure` con **4 substructs** (`substructs[4]`), y `struct Pokemon` = box + status/level/hp/stats. Los substructs se **cifran (XOR con personality^otId)** y su **orden depende de personality mod 24** (conocimiento estándar de comunidad; el cifrado no está en el header, no verificado aquí). Tamaños típicos 80 B (box) / 100 B (party) (no verificado en código).
- **Gen 4** (pokeplatinum `include/struct_defs/pokemon.h`, verificado): `BoxPokemon` con `personality`, `checksum`, `checksumFailed` y `dataBlocks[4]` (bloques A–D) + `PartyPokemon`. Los bloques se cifran con PRNG sembrado por el checksum y se barajan por personality (no verificado en código).
- Consecuencia para ARES: el adapter simbólico Gen3/Gen4 debe **descifrar y re-ordenar**, y validar checksum. Gen1/2 no.

### A6. Datos externos
- **Pokémon Showdown** (https://github.com/smogon/pokemon-showdown, ~6.0k stars): `data/` contiene `pokedex.ts`, `moves.ts`, `learnsets.ts`, `typechart.ts`, `items.ts`, `abilities.ts`, `mods/` (mods por generación, p.ej. gen1–gen8) (verificado listado). Fuente de mecánicas por gen para embeddings de especies/movimientos.
- **poke-env** (https://github.com/hsahovic/poke-env, ~0.5k stars): interfaz Python para bots de Showdown.
- **PokeAPI** (https://github.com/PokeAPI/pokeapi, ~5.5k stars) + PokeAPI/sprites (~2.0k): datos y sprites multi-gen (útil para pre-entrenar un encoder visual de sprites cross-gen).

### A7. Propuesta: Game Abstraction Layer (GAL) para ARES
```
Emulator backend (PyBoy | stable-retro[Gambatte/mGBA/melonDS] | py-desmume)
   -> PlatformAdapter: step(buttons, frames), screen(), read_mem(addr), save/load_state
   -> GameAdapter (por juego, generado desde pret .sym):
        symbolic_state(): party[], badges, map_id, xy, in_battle, menu_state, flags  (OPCIONAL)
        events(): {badge_gained, new_map, battle_won, fainted, ...}  (para reward/eval, NO necesariamente para obs)
   -> UniversalEnv (gymnasium):
        obs = { "pixels": frame normalizado (p.ej. 2 pantallas NDS apiladas o recortadas a 160x144/240x160 -> resize común),
                "symbolic": vector opcional con máscara de disponibilidad (feature dropout) }
        action = Discrete({A,B,Start,Select,Up,Down,Left,Right,NOOP} [+ Touch(x,y) macro solo NDS]) x action_repeat
```
- Principios: (1) **pixels-first**: lo simbólico es opcional y se puede apagar para medir "cuánto hace la RAM" (ablación obligatoria); (2) **reward desde eventos genéricos** (badges, mapas nuevos, nivel, captura) definidos uniformemente para todas las gens — evita reward específico por juego; (3) el harness NO ejecuta acciones de juego (nada de auto-cut/auto-surf); (4) definición única de "progreso" = lista de hitos por juego y métrica en **pasos de agente + frames emulados + wall-clock + costo**.
- FPS headless por plataforma: GB ≈ 60k fps sin render (verificado README PyBoy); ~9.4–10k SPS a 24 frames/acción (verificado); GBA y NDS: **medir** con script de benchmark propio (no hay cifras primarias).

---

## Parte B — Críticas de la comunidad a los agentes de Pokémon

### B1. "El harness hace el trabajo"
- **Gemini Plays Pokémon** (reporte técnico Gemini 2.5, arXiv 2507.06261, verificado HTML): el agente recibe "a subset of RAM information", mapa fog-of-war cuyo tipo de tile "is recorded from RAM", y herramientas `pathfinder` y `boulder_puzzle_strategist` (instancias de Gemini 2.5 Pro). Primer run: **813 horas**; segundo run con harness fijo: **406,5 horas**.
- El autor (Joel Z) pidió no tratarlo como benchmark (vía LessWrong "Is Gemini now better than Claude at Pokémon?", Julian Bradshaw).
- PokéAgent paper (2603.15563, verificado): "By conflating harness with model capability, it became impossible to attribute success..." (paráfrasis), y métricas "steps/actions/turns" sin definición común.
- Research notes LessWrong (abr 2025): incluso con overlay de tiles transitables, navegación y "Oversight" model, visión de Claude 3.7 "cripplingly bad"; o3 se queda ~800 acciones en un Pokécenter; costo ~$30 por 2.000 pasos.

### B2. Información privilegiada de RAM
- HN "Show HN: LLM plays Pokémon" (hn.svelte.dev/item/43187231): debate sobre la frase de la stream "parsed directly from the RAM" y ambigüedad de qué recibe el modelo (no verificado comentario por comentario).
- Gemini: tipos de tile desde RAM (verificado en tech report). Ganador del Speedrunning track de PokéAgent destaca contraste: otro equipo declara "absence of privileged game state" (verificado en paper).
- RL clásico (PWhiddy, Pleines, puffer) usa RAM para **reward** y a menudo para obs (mapa/coords/eventos) — Pleines lo reconoce como limitación.

### B3. Reward hacking
- Pleines et al. (verificado): "Heal Reward Exploitation" — Bulbasaur + Leech Seed vs Zubats en Mt. Moon para batallas casi infinitas y farmear reward de curación; se repite en otras ablaciones.
- PokeRL 2604.10812 (verificado abstract): agentes degeneran en "action loops, menu spam, or unproductive wandering".

### B4. Segmentos scripted / ayudas del entorno
- pokemonred_puffer `config.yaml` (verificado): `auto_flash: True`, `auto_use_strength: True`, `auto_use_surf: True`, `auto_teach_surf/strength: True`, `auto_solve_strength_puzzles: True`, `auto_pokeflute: True`, `auto_remove_all_nonuseful_items: True`, **`infinite_money: True`**; además el loop de acción presiona "a" hasta 1000 veces mientras `wJoyIgnore` bloquea el input (docs). Gran parte del juego difícil lo resuelve el entorno.
- PokéAgent ganador Heatz (verificado): LLM descompone en subgoals y **genera políticas scripted** que luego se destilan a red neuronal con IL + RL.

### B5. Sin aprendizaje / sin generalización
- LLM agents no aprenden entre runs (memoria en notas); dev.to (Qingze Hu): control en **Crystal** atascado en Route 29 tras 763 turnos vs FireRed avanza → éxito explicado por **memorización** del juego (N=1, no verificado independientemente).
- LessWrong (may 2026): conocimiento de walkthroughs "contamina" los runs; "Pokémon Red was never an ideal challenge" para razonamiento.
- RL: un agente por juego, reward y obs específicos de Pokémon Red; nadie ha mostrado transferencia cross-gen (no se encontró evidencia).

### B6. Costo
- dev.to: ~**$65,40** hasta la 1ª medalla (turno 1.785), $73,50 a 2.000 turnos; estimación de completar en "low four-figure dollars" (estimación del autor).
- Research notes LW: ~$30/2.000 pasos; ~$1.500 para 50k pasos.
- RL: Pleines ~36 h por run en HPC; puffer usó 4×RTX 4090; antes pagaba $200/mes en vast.ai.

### B7. Cherry-picking / no reproducible
- Anthropic (Fable 5 launch): afirmación "minimal, vision-only harness" para FireRed **sin publicar harness, pasos, costo ni número de intentos** (según dev.to; anuncio primario no verificado por mí).
- Harness de ClaudePlaysPokemon cambió muchas veces (se quitó "Critic Claude", hints, límites de memoria) → comparaciones entre versiones inválidas (LessWrong may 2026).
- Runs N=1 casi universales en LLM; RL reporta "mejor de 5 seeds" o figuras seleccionadas (Pleines tabla II selecciona "best data point").

### B8. Definiciones inconsistentes de "beat the game"
- PokéAgent (verificado): "Was Gemini 3 Pro's 173-hour completion better than Claude Opus 4.6 reaching Victory Road? Did GPT-5's step count account for the same mechanics?" — GPT-5 "6,470 steps", Gemini 2.5 Pro "406 hours", Claude "35,000 actions" para una sección.
- Métricas mezclan horas, pasos de harness (con múltiples botones por paso), acciones, turnos. Comentario LW: Opus 4.7 podía enviar secuencias largas de botones por turno.
- PokéAgent propone 15 hitos hasta Roxanne (1er gimnasio Emerald); humano top 18 min, promedio 1:22:05; mejor agente Heatz **40:13** (2,2× más lento que speedrunner); 2º Hamburg PokéRunners (PPO recurrente puro) 1:14:43.

---

## Implicaciones para ARES (vicios a corregir explícitamente)
1. Obs pixels-first; RAM sólo para reward/eval, con ablación "con/sin simbólico".
2. Cero ayudas de juego en el entorno (sin auto-HM, sin dinero infinito); si se usan, declararlas en tabla.
3. Reward genérico por eventos cross-gen + penalización/regularización anti-loop (aprendido de PokeRL/Pleines).
4. Protocolo de evaluación fijo: hitos por juego, ≥3–5 seeds, reportar pasos de agente, frames emulados, wall-clock, GPU-horas.
5. Test de generalización: entrenar en Gen1/Gen3, evaluar zero/few-shot en juego no visto (p.ej. Crystal) — ataca directamente la crítica de memorización.

## Referencias
- PyBoy: https://github.com/Baekalfen/PyBoy
- Pleines et al. 2025, Pokemon Red via RL: https://arxiv.org/abs/2502.19920
- PokeRL 2026: https://arxiv.org/abs/2604.10812
- pokemonred_puffer: https://github.com/drubinstein/pokemonred_puffer ; docs https://drubinstein.github.io/pokerl/docs/chapter-3/running ; config https://github.com/drubinstein/pokemonred_puffer/blob/main/config.yaml
- PWhiddy PokemonRedExperiments: https://github.com/PWhiddy/PokemonRedExperiments
- pygba: https://github.com/dvruette/pygba
- libmgba-py: https://github.com/hanzi/libmgba-py
- pokebot-gen3: https://github.com/hanzi/pokebot-gen3
- gbax (PyPI): https://pypi.org/project/gbax/
- py-desmume: https://github.com/SkyTemple/py-desmume
- melonDS: https://github.com/melonDS-emu/melonDS ; libretro core https://github.com/libretro/melonDS
- stable-retro: https://github.com/Farama-Foundation/stable-retro
- pret: https://github.com/pret/pokered , https://github.com/pret/pokecrystal , https://github.com/pret/pokeemerald , https://github.com/pret/pokefirered , https://github.com/pret/pokeplatinum , https://github.com/pret/pokeheartgold
- Pokémon Showdown: https://github.com/smogon/pokemon-showdown ; poke-env: https://github.com/hsahovic/poke-env ; PokeAPI: https://github.com/PokeAPI/pokeapi
- PokéAgent Challenge paper: https://arxiv.org/abs/2603.15563 ; https://pokeagentchallenge.com
- Gemini 2.5 tech report: https://arxiv.org/abs/2507.06261
- LessWrong, "a year late claude finally beats pokemon": https://www.lesswrong.com/posts/sehJYg5Yny9fvpbpt/a-year-late-claude-finally-beats-pokemon
- LessWrong, "Is Gemini now better than Claude at Pokémon?": https://www.lesswrong.com/posts/7mqp8uRnnPdbBzJZE/is-gemini-now-better-than-claude-at-pokemon
- LessWrong, research notes Claude 3.7/Gemini 2.5/o3: https://www.lesswrong.com/posts/8aPyKyRrMAQatFSnG/research-notes-running-claude-3-7-gemini-2-5-pro-and-o3-on
- dev.to (Qingze Hu), replicación vision-only FireRed: https://dev.to/qingze_hu_c4c251c1b353ede/i-replicated-the-vision-only-pokemon-run-anthropic-showcased-on-the-fable-5-launch-page-then-i-35ah
- HN Show HN LLM plays Pokémon: https://hn.svelte.dev/item/43187231
- TechCrunch, Gemini beats Pokémon Blue "with a little help": https://techcrunch.com/2025/05/03/googles-gemini-has-beaten-pokemon-blue-with-a-little-help
- dbreunig, case study: https://www.dbreunig.com/2025/06/17/an-agentic-case-study-playing-pokémon-with-gemini.html
- Ars Technica (no accesible desde aquí): "Why Anthropic's Claude still hasn't beaten Pokémon" / "Why Google Gemini's Pokémon success isn't all it's cracked up to be" (no verificado contenido)
- Tom's Hardware, "JEV decision model beat Pokemon Red" (solo titular, no verificado): https://www.tomshardware.com/tech-industry/artificial-intelligence/developer-says-jev-decision-model-beat-pokemon-red-in-under-a-week-non-llm-engine-succeeds-where-traditional-chatbots-stalled-for-months-but-claude-opus-5-coached-the-model-through-its-dead-ends
