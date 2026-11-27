# Estado del arte: agentes que juegan Pokémon y agentes generalistas (corte 2026-11-24)

> Este documento resume las notas `notas/rl_agents.md`, `notas/llm_agents.md`, `notas/benchmarks_battle.md`, `notas/generalist_and_methods.md` y `notas/infra_and_critiques.md`. Las marcas indican cuánto se pudo comprobar cada dato: **[V]** se verificó en fuente primaria (paper, README o post del autor); **[S]** proviene solo de fuente secundaria (prensa); **[NV]** no se verificó, así que no debe citarse como dato duro. Las claves BibTeX están en `referencias.bib`.

## 0. Síntesis

1. **No existe ningún agente que juegue "cualquier Pokémon".** Cada trabajo publicado entrena o ajusta para **un juego** (Red, Emerald, Crystal o FireRed) y ninguno evalúa transferencia a otra generación. El único resultado entre juegos con un control explícito es negativo: la réplica de Fable 5 se queda atascada en Crystal (§2).
2. **El RL puro avanza poco y depende de muchos scripts.** Pleines et al. llegan a Cerulean y nunca completan esa sección [V]. PokeRL/puffer dice que "vence Red", pero su propia página reconoce que **ninguna corrida estable lo logra con todos los scripts apagados** [V].
3. **Los LLM/VLM ya terminan juegos completos** (Red, Blue, Crystal, Yellow Legacy, FireRed), pero lo hacen con harnesses que dan estado leído de RAM, mapas y pathfinders, a veces con **intervención humana**, y con costos enormes: 813 h en Gemini Blue run 1 y 1,880 millones de tokens en Gemini 3 Crystal [V]. Sin harness, los VLM de frontera **completan prácticamente el 0 %** [V, PokéAgent].
4. **El combate es el único subproblema donde hay un benchmark serio** (PokéAgent, Showdown, Metamon). Allí ganan el RL offline y la búsqueda, no los LLM [V]. Pero Showdown no es el combate del juego (§3.4).
5. **Los métodos generalistas modernos** (VPT/IDM, NitroGen, acciones latentes, JEPA, RND/E3B, Go-Explore, xLSTM/Mamba) ofrecen piezas transferibles que **nadie ha combinado todavía para Pokémon** sin RAM ni scripts. Ese hueco es el que aborda ARES-U.

---

## 1. (a) Agentes de RL

| Nombre | Año | Enfoque | Juego(s) | Resultado | Limitaciones | URL |
|---|---|---|---|---|---|---|
| Pleines et al., "Pokemon Red via RL" [V] | 2025 | PPO con Nature CNN sobre píxeles 72×80 ×3, crop de visitados de 48×48, vector de estado (HP, nivel, flags) y GRU opcional; 7 botones, 1 decisión cada 24 frames | Red (alrededor del 20 % del juego, hasta Cerulean; arranca tras el parcel con Squirtle) | 4.19/7 hitos; Brock 0.99; llega a Cerulean 0.85; Misty 0.27; **Cerulean completo 0.00**. Agente 25,299 pasos frente a 11,188 de un humano. 5 semillas | Reward hacking (399 curaciones frente a 23 de un humano). Nadie aprende Cut. ~36 h por run; la variante GRU en CPU tarda ~24 días. RAM usada para flags | https://arxiv.org/abs/2502.19920 |
| PokeRL / pokemonred_puffer (Rubinstein, Whidden, PufferAI) [V] | 2025 | PPO + LSTM (~5M parámetros según la página de policy; el titular dice <10M); CNN sobre 72×80 más embeddings de RAM (party, 2560 flags, items, map ID); swarming | Red | Dice que "vence Red" desde feb-2025 | **Ninguna corrida estable gana con todos los scripts apagados.** Scripts activos: dinero infinito, Strength y Surf automáticos, puzzles resueltos, Flash, Pokéflauta. Más de 25 términos de recompensa; recompensa por map ID ("on rails"). 288 envs y 10^10 pasos configurados; 4× RTX 4090 | https://drubinstein.github.io/pokerl/ · https://github.com/drubinstein/pokemonred_puffer |
| PokemonRedExperiments (Whidden) [V] | 2023-24 | PPO de SB3 con PyBoy. V1 explora por novedad de frames (KNN); V2 por coordenadas | Red | V2 "llega a Cerulean"; más de 50,000 h simuladas [S TechCrunch] | *Noisy TV*: se queda mirando agua y pasto animados. Evita el PC para siempre tras un evento negativo. Empieza tras el parcel desde un savestate | https://github.com/PWhiddy/PokemonRedExperiments |
| PokeGym (xinpw8 / PufferAI) [V] | 2023-24 | Fork de Whidden sobre PufferLib | Red | Medalla 1 a 9.6M pasos; medalla 3 a **404M pasos** | La métrica es "primera vez en algún env", no una tasa por semilla | https://github.com/PufferAI/pokegym |
| PokeRL (Mudireddy & Patibandla) [V] | 2026 | PPO de SB3 (~1.07M parámetros), píxeles más máscara de visitados; RAM solo para la recompensa; recompensa jerárquica; currículum por savestates; wrapper anti-loop | Red (inicio del juego) | Episodios en loop: 41.2 % → 4.7 %; ~50 % de victorias contra el rival a 500k pasos | Solo cubre el inicio. Start/Select eliminados porque el agente aprendió a guardar la partida. Documenta botón spam (A 32.1 %) | https://arxiv.org/abs/2604.10812 |
| Hamburg PokéRunners (PokéAgent T2, 2º) [V] | 2025 | PPO recurrente + LSTM; CNN sobre 128×128 con un vector de 38 hitos; solo recompensas positivas | Emerald hasta Roxanne | 1:14:43 | Modelos por segmento cosidos en inferencia (no es un generalista). Con penalizaciones aparecía "only safe behavior" | https://arxiv.org/abs/2603.15563 |
| JEPA multimodal (Campese & Moschitti) [V página; detalles NV] | 2026 | Encoder JEPA de píxeles más RAM, preentrenado offline y congelado para PPO | Red | Entre 2 y 8 veces la recompensa de DINOv3 o de una inicialización aleatoria; el fine-tuning degrada la representación; la diversidad de datos pesa más que la pericia | Un solo juego; la unidad de recompensa no está clara en la página | https://openreview.net/forum?id=8li5NOHh9S |
| Gold/Crystal (tesis Leiden; Medium; PokemonCrystalRL) [NV] | 2024-25 | RL con currículum o recompensas tomadas de RAM | Gold/Crystal | Sin cifras verificadas | Casi no hay trabajo serio de RL en Gen 2 | https://theses.liacs.nl/pdf/2024-2025-EdelaarPPhilip.pdf |

**Lectura.** Todos los RL serios usan **recompensa diseñada a mano para el juego**, **RAM de Gen 1** como observación o recompensa, y **savestates o swarming**. Ninguno se evalúa en otro juego.

---

## 2. (b) Agentes LLM/VLM

| Nombre | Año | Harness | Juego | Resultado | Limitaciones | URL |
|---|---|---|---|---|---|---|
| Claude 3.7 Sonnet Plays Pokémon [V] | 2025 | 2 screenshots (uno con overlay de coordenadas), estado breve leído de RAM, KB de 8k tokens, `navigator` a tiles visibles; luego un crítico | Red | 3 medallas; "35,000 acciones" hasta Surge [S] | 52 h en Mt. Moon; 12 h pulsando A sobre una alfombra; transiciones inventadas | https://latent.space/reading/claude-plays-pokemon/ |
| Claude Opus 4.5 / 4.6 [V] | 2026 | Harness "mínimo" | Red | 8 medallas; atascado en los boulders de Victory Road a 230k pasos | Visión "still pretty bad"; no usó Gold Teeth | https://www.lesswrong.com/posts/gogZyeistdaDFuhbG/claude-plays-pokemon-opus-4-5-follow-up |
| Claude Opus 4.7 [V] | 2026 | Se agregaron Zoom y memoria de screenshots; crítico debilitado [NV] | Red | **Gana Red** | Pasos y horas no publicados. Sigue deambulando y olvidando; los boulders le tomaron ~1 día | https://www.lesswrong.com/posts/sehJYg5Yny9fvpbpt/a-year-late-claude-finally-beats-pokemon |
| Fable 5 (Anthropic) [S] | 2026 | Se dice "solo visión", sin mapas ni ayudas | FireRed | Dicen que gana | No hay harness, pasos, costo ni número de intentos publicados (página oficial no leída) | https://phandroid.com/2026/06/11/claude-fable-5-beat-pokemon-firered-with-no-maps-or-tools-just-an-overleveled-charizard/ |
| Réplica de Fable 5 (Qingze Hu) [V post] | 2026 | ~800 líneas con mGBA, 1 screenshot, notas y 1 herramienta | FireRed / Crystal | Boulder Badge en el turno 1,785 (US$65.40) | **Memorización**: anota "Oak's Parcel" 141 turnos antes de recibirlo. **Control en Crystal: 763 turnos atascado en Route 29.** N=1 | https://dev.to/qingze_hu_c4c251c1b353ede/i-replicated-the-vision-only-pokemon-run-anthropic-showcased-on-the-fable-5-launch-page-then-i-35ah |
| Gemini 2.5 Pro Plays Pokémon [V] | 2025 | RAM convertida a texto, mapa *fog-of-war* con tipos de tile leídos de RAM, `pathfinder` y `boulder_puzzle_strategist` (instancias de Gemini) | Blue | Campeón: **813 h** (run 1, con intervenciones) y 406.5 h (run 2) | *Agent panic*, *context poisoning*, alucinaciones de otras versiones. Quitarle la visión casi no cambia el desempeño | https://arxiv.org/abs/2507.06261 |
| Gemini 3 Pro en Crystal [V] | 2025 | Mental Map (solo tiles vistos), notepad, ejecución de código, agentes propios | Crystal | Vence a Red en el turno **24,178**, con **1,880 millones de tokens** en ~17 días | Alucina causas; ~2 días en Goldenrod Underground | https://blog.jcz.dev/gemini-3-pro-vs-25-pro-in-pokemon-crystal |
| Gemini en Yellow Legacy (hard) [V] | 2025 | Meta-herramientas: el modelo construye su propio harness | Yellow Legacy | Elite Four derrotado (con 18-20 intentos contra cada miembro) | Bucle en la Power Plant: el mismo payload 842 veces en 1,003 turnos | https://arxiv.org/abs/2605.09998 |
| Continual Harness (Karten et al.) [V] | 2026 | Refinamiento automático del harness en línea, sin resets; co-aprendizaje con Gemma-4 (SFT + GRPO + DAgger) | Emerald, Red | Emerald (Pro): 100 % de hitos a una mediana de US$130, frente a 98 % a US$215 con un harness mínimo | Inconsistencias internas en el paper. La ganancia depende de la capacidad del modelo | https://arxiv.org/abs/2605.09998 |
| GPT o3 / GPT-5 Plays Pokémon [S] | 2025 | Frame, ubicación, party, dinero, colores de tile, minimapa, notepad. Emulador pausado entre acciones | Red, Crystal | GPT-5: Red en 6,470 pasos (~7 días) [S] | No hay fuente primaria. La definición de "paso" no es comparable | https://www.techradar.com/ai-platforms-assistants/chatgpt/gpt-5-just-completed-pokemon-red-in-a-new-world-record-time-claude-gemini-and-chatgpt-o3-arent-even-close |
| llm_pokemon_scaffold (cicero225, autor anónimo; post de LessWrong publicado por J. Bradshaw) [V] | 2025 | Mapa ASCII de colisión, etiquetas "CHECK HERE", navegación DFS | Red | "helps somewhat" | ~US$30 por 2,000 pasos; o3 pasa ~800 acciones dentro de un Pokécenter | https://github.com/cicero225/llm_pokemon_scaffold |
| PokéAI [V] | 2025 | Agentes de texto Planning/Execution/Critique | Red | Combate: 80.8 % de victorias en 50 salvajes | Solo texto; muestra pequeña | https://arxiv.org/abs/2506.23689 |
| ASH [V abstract] | 2026 | Aprende de video de internet mediante un IDM | Emerald | 11.2/12 hitos en 8 h, frente a 9.3/12 del mejor baseline | Detalles no verificados | https://arxiv.org/abs/2605.14211 |
| Jev (TypeSafe AI) [NV] | 2026 | Motor de decisión no-LLM "entrenado" vía log por Claude Opus 5 | Red | Ganaría Red en menos de 1 semana | Solo se leyó el titular | https://www.tomshardware.com/tech-industry/artificial-intelligence/developer-says-jev-decision-model-beat-pokemon-red-in-under-a-week-non-llm-engine-succeeds-where-traditional-chatbots-stalled-for-months-but-claude-opus-5-coached-the-model-through-its-dead-ends |

**Qué aporta el harness** (resumen de `notas/llm_agents.md` §2): según el caso, el harness entrega **estado leído de RAM, mapas, pathfinders o A\*, memoria y sub-agentes**. Por eso PokéAgent concluye que los resultados miden *modelo + harness* y no pueden atribuirse solo al modelo [V].

---

## 3. (c) IA de combate y benchmarks

### 3.1 Competencias y benchmarks de juego completo

| Nombre | Año | Qué mide | Resultados clave | Limitaciones | URL |
|---|---|---|---|---|---|
| **PokéAgent Challenge (NeurIPS 2025)** [V] | 2025-26 | T1: combate en Showdown (Gen1 OU, Gen9 OU), rating FH-BT con bootstrap y al menos 250 combates. T2: speedrun de Emerald hasta Roxanne en 15 hitos; la acción final debe salir de una red neuronal | Más de 100 equipos. T1: PA-Agent (offline RL tipo Metamon) gana Gen1 OU 50-28; FoulPlay (MCTS) gana Gen9 OU 50-14. T2: **Heatz 40:13** (LLM → scripts → destilación a red + DQN), Hamburg 1:14:43 (PPO recurrente), anthonys 1:29:17 (mejor LLM puro), Deepest con 649 pasos. Humanos: 18 min el mejor, 1:22:05 el promedio. **VLM sin harness: ~0 %**. 20M trayectorias liberadas | Modelo y harness quedan confundidos; "steps/actions/turns" no tienen una definición común; la contaminación solo se discute con argumentos | https://arxiv.org/abs/2603.15563 · https://pokeagentchallenge.com |
| lmgame-Bench [V abstract] | 2025-26 | API Gym unificada; Pokémon Red sobre PyBoy; 13 modelos | Reconoce la **contaminación** y el RL en un juego transfiere a juegos no vistos | El combate en Red es "too easy" y navegar "too hard" [NV] | https://arxiv.org/abs/2505.15146 |
| Orak (Krafton) [V] | 2025 | 12 juegos; en Red, vencer a Brock (12 flags), con estado en texto | GPT-5 88.9±4.8; humanos novatos 86.1±12.7; Claude-3.7 63.9 | Estado textual privilegiado | https://arxiv.org/abs/2506.03610 |
| VideoGameBench [V abstract] | 2025 | 10 juegos de los 90, incluido Crystal, solo con visión | El mejor (Gemini 2.5 Pro) completa 0.48 % (1.6 % en la versión Lite) | — | https://arxiv.org/abs/2505.18134 |
| PokeGym (Legends Z-A) [V abstract] | 2026 | Benchmark visual 3D | G-EvoMAC 60.18 % | Otro género (3D) | https://arxiv.org/abs/2604.08340 |

### 3.2 IA de combate

| Nombre | Año | Enfoque | Formato | Resultado | Limitaciones | URL |
|---|---|---|---|---|---|---|
| **Metamon** [V] | 2025 | IL → offline RL → self-play offline; RNN y Transformers de hasta 200M; reconstruye la vista en primera persona desde replays | Gen 1-4 OU, Gen 9 OU | GXE: Kakuna (142M) 82/70/63/64/71 % en G1/G2/G3/G4/G9; TaurosEnsemble **#1 del ladder humano de Gen1OU** | Brecha sim2sim entre replays y servidor; la observación está ligada al formato (falló en Gen9 sin Team Preview); el GXE local no es comparable con el del ladder | https://arxiv.org/abs/2504.04395 · https://github.com/UT-Austin-RPL/metamon |
| PokéChamp [V] | 2025 | Minimax donde un LLM propone acciones, modela al oponente y estima el valor | Gen9 OU | GPT-4o: 76 % contra PokéLLMon y 84 % contra Abyssal; Elo **proyectado** de 1300-1500 | El Elo no proviene de un ladder sostenido; costo por turno | https://arxiv.org/abs/2503.04094 |
| PokéLLMon [V] | 2024 | GPT-4 con ICRL, KAG y auto-consistencia | Gen8 Random | 64 % contra el heurístico; 48.57 % en el ladder (105 combates) | **Panic switching**; contra estrategias de desgaste cae a 18.75 % (n=16); acierta solo el 84 % de las ventajas de tipo | https://arxiv.org/abs/2402.01118 |
| VGC-Bench [V] | 2025 | BC y PSRO (SP, FP, DO) | VGC dobles | Vence a un profesional en mirror con 1 equipo | Con más equipos rinde peor y es más explotable; el LLM se evaluó solo con 20 combates | https://arxiv.org/abs/2506.10326 |
| PokaiTrainer [V abstract] | 2026 | Búsqueda de equilibrio (tipo CFR) más motor en Rust (~99 % de paridad con Showdown) | VGC | 59 % en 150 sets Bo3; pico de 1492 | "La red sola pierde contra un heurístico de búsqueda superficial" | https://arxiv.org/abs/2608.29197 |
| Foul Play [V parcial] | 2024-26 | MCTS/expectiminimax en Rust, compilado por generación | Singles de todas las gens | Campeón Gen9 OU en PokéAgent; pico de ~2341 Elo | Conocimiento hardcodeado; sin modelo multi-gen | https://github.com/pmariglia/foul-play |
| Jaxcalibur [V blog] | 2026 | Transformer de 8.5M con heads auxiliares; PPO por self-play (~100M partidas en 1 H100 durante ~1 semana) más pUCT | Gen9 Random | Pico de 2557 Elo y 95.3 GXE | Sin código ni pesos; una sola ventana de 3 semanas | https://jaxcalibur.github.io/ |
| Porimon [V abstract] | 2026 | LLM con KAG y API de daño | — | Supera a PokéLLMon | La contribución del KAG es "inconclusa" | https://arxiv.org/abs/2609.32544 |

### 3.3 Líneas base estándar en combate

Las más comunes son Random, MaxBasePower, max-damage con fórmula y tipos (`Grunt` de Metamon), SimpleHeuristicsPlayer (poke-env), Abyssal, *one-step lookahead* e IA de NPC del juego (`Gen1BossAI`, `EmeraldKaizo` en Metamon). En ARES ya existen equivalentes: `RandomPolicy`, `GreedyDamagePolicy` y `MLPGenome.prior`.

### 3.4 Showdown frente al combate dentro del juego

Seis desajustes (`notas/benchmarks_battle.md`, análisis transversal):

1. **Oponente**: en Showdown es un humano o un bot competitivo; en el juego es una IA de NPC simple con equipos fijos.
2. **Información**: Team Preview frente a rivales ocultos pero memorizables.
3. **Recursos persistentes**: en el juego, HP, PP e ítems persisten entre combates y hay EXP y captura; el objetivo es progresar con poco costo, no ganar una partida.
4. **Reglas**: cláusulas de Smogon frente a mecánicas y glitches del cartucho.
5. **Interfaz**: estado estructurado frente a menús por botones, que es donde aparecen el *menu spam* y los loops.
6. **Formato**: singles 6v6 o VGC frente a encuentros 1vN.

Por esta razón PokéAgent separa los tracks.

---

## 4. (d) Agentes generalistas entre juegos y métodos NN modernos

| Nombre | Año | Enfoque | Qué transfiere | Resultado | Limitaciones | URL |
|---|---|---|---|---|---|---|
| VPT (OpenAI) [V] | 2022 | Un IDM con pocas etiquetas pseudo-etiqueta 70,000 h de video; BC y luego RL | Prior de comportamiento aprendido de video | Herramientas de diamante en Minecraft | Un solo juego; cómputo masivo | https://arxiv.org/abs/2206.11795 |
| SIMA / SIMA 2 (DeepMind) [V] | 2024-25 | Imagen y lenguaje → teclado y mouse | Relación lenguaje-acción | En juegos no vistos, ~90 % del especialista (lectura de gráfico) | Cerrado; no reproducible | https://arxiv.org/abs/2404.10179 · https://arxiv.org/abs/2512.04797 |
| Genie 1/2/3; LAPO; LAPA [V] | 2024-25 | Acciones latentes aprendidas desde video (VQ) | Unifica acciones sin etiquetas | LAPO recupera la estructura de acciones en Procgen | Genie es cerrado; horizontes de minutos | https://arxiv.org/abs/2402.15391 · https://arxiv.org/abs/2312.10812 · https://arxiv.org/abs/2410.11758 |
| **NitroGen (NVIDIA)** [V] | 2026 | DiT de 500M; BC sobre 40,000 h de más de 1,000 juegos, con acciones tomadas de overlays de gamepad | **Espacio de acción de gamepad unificado** | Hasta 52 % de mejora relativa en juegos no vistos | Sin memoria (ve solo el último frame); no planifica | https://arxiv.org/abs/2601.02427 |
| Game-TARS / Lumine (ByteDance) [V] | 2025 | VLM con acciones nativas y razonamiento esparso o adaptativo | Mismo género o publisher | Lumine completa 5 h de Genshin; zero-shot en HSR | Escala industrial; transferencia entre juegos "fáciles" | https://arxiv.org/abs/2510.23691 · https://arxiv.org/abs/2511.08892 |
| DreamerV3 / Drama / IRIS / DIAMOND [V] | 2023-24 | World models (RSSM, Mamba, Transformer, difusión) | Un algoritmo robusto | Drama: competitivo en Atari100k con 7M parámetros | Sin evaluación en Pokémon | https://arxiv.org/abs/2301.04104 · https://arxiv.org/abs/2410.08893 |
| V-JEPA 2 [V] | 2025 | Auto-supervisado en el espacio latente | Features de video | SSv2 77.3 | Probablemente sobredimensionado para pixel-art | https://arxiv.org/abs/2506.09985 |
| Decision Transformer / LRAM (xLSTM) [V] | 2021-24 | RL como secuencia; recurrencia lineal | Política offline multi-tarea | LRAM: 432 tareas, comparable a Transformers | Depende de la calidad de los datos | https://arxiv.org/abs/2106.01345 · https://arxiv.org/abs/2410.22391 |
| RND / NovelD / E3B [V] | 2018-22 | Bonus intrínseco (red aleatoria, frontera, elíptico sobre un embedding IDM) | Exploración genérica | RND supera el promedio humano en Montezuma; E3B es SOTA en MiniHack | E3B requiere un IDM | https://arxiv.org/abs/1810.12894 · https://arxiv.org/abs/2210.05805 |
| **Go-Explore** [V] | 2019-21 | Archivo de celdas, volver restaurando el estado y explorar (QD) | Archivo de estados | Resuelve *detachment* y *derailment* | Necesita definir celdas | https://arxiv.org/abs/2004.12919 |
| G1 / GiGPO / Odysseus [V] | 2025-26 | VLM pequeño con RL (GRPO de dos niveles, PPO con crítico por turno) | Conocimiento del mundo del VLM | Odysseus: ≥3× progreso en Mario Land; PPO más estable que GRPO | GRPO es inestable en horizontes largos; inferencia cara | https://arxiv.org/abs/2505.13426 · https://arxiv.org/abs/2505.10978 · https://arxiv.org/abs/2605.00347 |
| GAME / Tournament-Informed QD (Anne et al.) [V] | 2026 | Coevolución QD adversarial con selección de oponentes por Ranking o Pareto | Protocolo de liga | Ranking mejor en win rate, ELO y AQD (20 réplicas, Holm) | No trata Pokémon | https://arxiv.org/abs/2601.19562 |

---

## 5. (e) Infraestructura multi-generación

| Componente | Plataforma / Gen | Estado | Rendimiento | URL |
|---|---|---|---|---|
| PyBoy [V] | GB/GBC (Gen 1-2) | Activo; wrapper `pokemon_gen1` | ×1014 tiempo real sin render (≈60,800 fps derivados); 9,403 SPS a 24 frames por acción en Red (Pleines); ~10k SPS en puffer | https://github.com/Baekalfen/PyBoy |
| mGBA / libmgba [V] | GBA (Gen 3) | Bindings de Python desactivados por defecto; en Windows hay que compilar. ARES ya tiene su propio shim C | FPS headless **no medido [NV]** | https://github.com/mgba-emu/mgba |
| pygba, libmgba-py, gbax [V] | GBA | Wrappers gymnasium o cffi | — | https://github.com/dvruette/pygba · https://pypi.org/project/gbax/ |
| stable-retro (Farama) [V] | GB/GBC (Gambatte), GBA (mGBA), **NDS (melonDS)** | Activo; Linux, Windows y macOS | **Una sola API Gym para GB/GBC/GBA/NDS** | https://github.com/Farama-Foundation/stable-retro |
| py-desmume / melonDS [V] | NDS (Gen 4-5) | py-desmume tiene wheels; melonDS no tiene bindings oficiales | FPS **no medido [NV]**; se espera mucho más pesado | https://github.com/SkyTemple/py-desmume |
| pret (pokered, pokecrystal, pokeemerald, pokefirered, pokeplatinum, pokeheartgold) [V] | Gen 1-4 | Gen 1-3 maduros; Gen 4 en progreso (WIP) | `.sym` para adaptadores de RAM | https://github.com/pret |
| Pokémon Showdown / poke-env / PokeAPI [V] | Gen 1-9 (datos y combate) | Activos | Mecánicas por generación (`data/mods`) | https://github.com/smogon/pokemon-showdown · https://github.com/hsahovic/poke-env |
| PufferLib [V] | Infra de RL | Activo | Vectorización rápida | https://arxiv.org/abs/2406.12905 |
| PokeJAX / EmuRust [V abstract] | Showdown en JAX; PyBoy en Rust | — | El overhead del entorno baja de 4 % | https://arxiv.org/abs/2603.12145 |

**Diferencias de structs** [V headers pret]: en Gen 1/2 son planas y sin cifrar; en Gen 3 hay 4 subestructuras cifradas con XOR y un orden que depende de la personalidad del Pokémon; en Gen 4 los bloques A-D se cifran con un PRNG (el detalle del cifrado es [NV]). Por eso un adaptador simbólico necesita descifrar y validar por generación, y la observación por píxeles se vuelve la opción **universal**.

---

## 6. Huecos que motivan ARES-U

1. Nadie evalúa **transferencia entre juegos o generaciones** con un protocolo de juego no visto (*leave-one-game-out*).
2. Nadie publica una **tabla de scripts on/off** junto con el progreso por semilla.
3. Las representaciones de combate usan **IDs de especie y movimiento** (Metamon, Showdown), que no transfieren a especies nuevas.
4. Las métricas no se pueden comparar entre trabajos (pasos, turnos, horas, tokens).
5. El LLM que gana lo hace como **profesor offline destilado** (Heatz), no como jugador en línea.

El catálogo completo de vicios, con la solución de ARES-U para cada uno, está en `VICIOS_Y_SOLUCIONES.md`.
