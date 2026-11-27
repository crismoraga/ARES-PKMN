# Agentes LLM/VLM que juegan (y terminan) juegos Pokemon

Fecha de corte: 2026-11-24. Proyecto: IPD434 Soft Computing (ARES-PKMN).
Convencion: [V] = verificado por mi en fuente primaria (post del autor, paper arXiv, pagina oficial); [S] = solo fuente secundaria (prensa); [NV] = no verificado.

---

## 0. Resumen ejecutivo

- Entre feb-2025 y nov-2026 los LLM/VLM pasaron de no salir de la casa de Pallet Town (Claude 3.0 Sonnet) a terminar Red/Blue/Crystal/Yellow Legacy/FireRed. Pero **casi todos los "exitos" dependen de un harness hecho a mano** (estado RAM convertido a texto, minimapas, pathfinders, sub-agentes, criticos, resumen de contexto) e incluso de **intervencion humana en vivo** (Gemini Blue run 1, refinamiento iterativo de GPP).
- La eficiencia es pesima frente a humanos y a RL especializado: Gemini 2.5 Pro Blue = 813 h (run 1) / 406.5 h (run 2) [V]; Gemini 3 Pro Crystal = 24.178 turnos, 1.88 mil millones de tokens, ~17 dias [V]; GPT-5 Red = 6.470 pasos (~7 dias) [S]. En el PokeAgent Speedrun (Emerald hasta Roxanne), el mejor baseline LLM fue ~1.8x mas lento que un humano promedio y el ganador fue un enfoque **LLM->scripts->destilacion a red neuronal + RL** (40:13, 1.608 pasos) [V].
- Fallos recurrentes y bien documentados: (1) vision/percepcion espacial (puertas, arboles cortables, alfombras, NPCs mal identificados); (2) bucles y "stuck" de cientos o miles de turnos (Mt. Moon 52+ h; Olivine Lighthouse 16.403 turnos; Power Plant 842 repeticiones en 1.003 turnos); (3) alucinacion/contaminacion de conocimiento de entrenamiento (items de otra version, "Oak's Parcel" anotado antes de recibirlo); (4) "context poisoning": notas/resumenes con creencias falsas que se auto-refuerzan; (5) "agent panic" con HP bajo; (6) planificacion de largo plazo pobre (olvidar Gold Teeth -> sin Strength); (7) no testear hipotesis.
- **Debate harness-confound**: las corridas X-Plays-Pokemon no son comparables (distinto estado, herramientas, granularidad de accion, definicion de "paso", juego, intervenciones, N=1). PokeAgent (arXiv 2603.15563) lo formaliza y propone evaluar harness x modelo en 5 dimensiones. Anthropic (Fable 5, 2026) responde con harness "solo vision", pero la replica de Qingze Hu muestra que entrada limpia != conocimiento limpio (memorizacion del juego).

---

## 1. Fichas por proyecto

### 1.1 Claude Plays Pokemon (Anthropic / David Hershey, Twitch, Pokemon Red), feb-2025 -> may-2026
- **Harness original (3.7 Sonnet)** [V, Latent Space mar-2025 + post Anthropic]: cada accion devuelve 2 screenshots (normal + con overlay de coordenadas) + breve descripcion de estado leida de RAM. Herramientas: `press_buttons` (secuencia), `knowledge_base` (dict Python ~8.000 tokens anexado al prompt), `navigator` (solo a coordenadas visibles en pantalla, no destinos arbitrarios). System prompt ~1.000 tokens. Resumen periodico reemplaza historial. Detector de "sin movimiento" (~30 pasos como ejemplo) inyecta recordatorios. Requests de 5k-100k tokens. Luego se anadio "Critique Claude".
- **Resultados 3.7 Sonnet** [V Anthropic]: 3 medallas (Brock, Misty, Surge); Claude 3.0 Sonnet no salio de la casa; 3.5 Sonnet llego a Viridian Forest [S]. "35.000 acciones" para llegar a Surge [S; PokeAgent lo cita].
- **Fallos 3.7** [V Latent Space]: confundir una alfombra roja con dialogo y presionar A ~12 h; identificar mal a Oak; caminar contra paredes; 12 h persiguiendo una salida inexistente; inventar transiciones de zona; 52 h en Mt. Moon (al momento de la entrevista). "Blackout strategy" por creencia falsa (Rogero 2026).
- **Opus 4.5** [V, Josh Snider, LessWrong 2026-01-29]: 8 medallas, atascado en boulders de Victory Road a 230.000 pasos; Silph Co. "just over 50k" pasos de razonamiento (corrida previa 29k) [V Bradshaw]; Cinnabar Mansion 112k pasos. Vision "still pretty bad"; ignoro Gold Teeth (Strength) -> fallo de planificacion recuperado con backtracking. Autor: harness "minimo" -> mejor test del modelo.
- **Opus 4.6** [V Bradshaw]: Cinnabar 3k pasos; Victory Road en ~30k pasos pero meses atascado en boulders.
- **Opus 4.7 GANA Pokemon Red** [V, Bradshaw "A Year Late, Claude Finally Beats Pokemon", LessWrong 2026-05-16]. Nuevas herramientas: Zoom (region de pantalla) y almacenamiento de screenshots (memoria visual). Segun comentarios [NV]: se quito/debilito Critic Claude, hints visuales y limites de memoria/botones; multiples A por paso en dialogos. Elite 4 tomo un par de intentos; tras perder con Blue, compro curas. Fallos aun presentes: puerta leida como "Conveyor Belt", caminar contra objetos, puzzle de boulders ~1 dia, "unfocused wandering and sheer forgetfulness". Pasos/horas totales NO publicados. Autor: ganancia "incremental", impulsada por vision.
- **Fable 5 (Anthropic, ~jun-2026)**: el anuncio dice que gano FireRed con harness minimo solo-vision ("no maps, navigation aids, or extra game-state information") [S: Phandroid 2026-06-11 y dev.to; no lei la pagina oficial]. Sin pasos/tiempo/costo publicados. Estilo: sobre-nivelar Charizard; revivir Pokemon debiles que caen al turno siguiente [S].
- **Replica Fable 5 (Qingze Hu, dev.to, jul-2026)** [V del post]: ~800 lineas Python + mGBA; 1 screenshot + notas propias por turno + 1 herramienta de botones; compresion cada 20 turnos. Boulder Badge en turno 1.785 ($65.40); tope 2.000 turnos ($73.50, ~3.7 centavos/turno; 6h45m reales = ~19 min de juego). Evidencia de memorizacion: anoto "Oak's Parcel" 141 turnos antes de recibirlo; identifico entrenador no visible. Control en Crystal: 763 turnos atascado en Route 29 (3 reinicios). Un prompt que prohibe usar conocimiento memorizado NO lo evito. N=1.

### 1.2 Gemini Plays Pokemon (Joel Zhang, independiente; luego co-autor con Princeton/DeepMind)
- **Blue, Gemini 2.5 Pro** [V, Gemini 2.5 tech report arXiv 2507.06261 secc. 4.1 + apendice 8.2]: run 1 (2.5 Pro Exp 03-25) desde 2025-03-28, campeon 2025-05-02 tras **813 h**; run 2 "totalmente autonoma" (Preview 05-06) desde 2025-05-22, **406.5 h**. Harness: RAM -> texto (el modelo falla con pixeles crudos), mapa fog-of-war con contador de visitas y tipo de tile desde RAM, objetivos primario/secundario/terciario, resumen cada 100 turnos y compresion cada 1.000, Guidance Gemini cada 25 turnos, `pathfinder` (Gemini simulando BFS, contextos 100k+ tokens, rutas de hasta 50-150 acciones), `boulder_puzzle_strategist` (anadido tras resolver 2/4 puzzles solo). Ablacion: quitar la vision casi no cambia el rendimiento. Sobre ~100k tokens de contexto el agente tiende a repetir acciones del historial.
- Fallos (apendice) [V]: agent panic; context poisoning (misinformacion en metas/resumenes, "black-out strategy"); topological traps (spinners Rocket Hideout, laberinto Route 13); alucinaciones de otras versiones (TEA de FRLG); fijacion de metas. Misty rematch prep >24 h; softlock Cycling Road 4 h; Safari Zone 17 intentos (run 1) vs 5 (run 2).
- Intervenciones [V rentry 2025-06-12]: run 1 con intervenciones "generales"; excepcion: decirle que hable dos veces con el Rocket Grunt (bug de Lift Key). Disclaimer del autor: no es benchmark.
- **Yellow Legacy hard mode**: Elite Four derrotado ago-2025; intentos de por vida: Lorelei 18, Bruno 20, Agatha 18, Lance 19, Campeon 4 [V, arXiv 2605.09998]. Desde Yellow Legacy los especialistas hechos a mano se reemplazaron por meta-herramientas (`define_agent`, `run_code`, notepad) y el modelo construye su propio harness. Bucle Power Plant: mismo payload 842 veces en 1.003 turnos (~3.5 h); tras ~500 turnos de stall deja de crear herramientas.
- **Crystal, Gemini 3 Pro vs 2.5 Pro** [V, blog.jcz.dev 2025-12-12]: mismo harness con Mental Map (fog-of-war solo con tiles vistos, NO lee layout de RAM), Notepad, Map Markers, Code Execution, Custom Agents, Custom Tools; prompt de "cientifico" (hipotesis, construir herramientas, no confiar en conocimiento de entrenamiento). 3 Pro: Campeon de Johto sin perder batallas; vence a Red en turno **24.178** con **1.88 mil millones de tokens**, ~**17 dias** ("Operation Zombie Phoenix": Typhlosion lv75 + relleno lv8-19). Early game: ~mitad de turnos y ~60% menos tokens que 2.5 Pro. 2.5 Pro: Olivine Lighthouse turno 21.801 -> Fog Badge turno 38.204 (16.403 turnos); proyeccion 157k turnos / >15 mil millones tokens / ~69 dias. Fallos 3 Pro: alucina causas al sintonizar la radio (Snorlax) y repite horas; Goldenrod Underground ~2 dias por suponer que el orden de switches no importa, hasta hablar con un NPC; downtime de API ~250% mayor.
- **Continual Harness** (Karten, Zhang, ..., Chi Jin, Vodrahalli; arXiv 2605.09998, 2026-05-11) [V abstract + html]: automatiza el refinamiento humano del harness (prompt, sub-agentes, skills, memoria) online sin resets. Emerald (Pro): 100% milestones a $130 mediana vs minimalista 98% a $215 (~40% menos costo); Flash: 80% a $42 vs 77% a $30; Flash-Lite: 13-33% vs 20% a $11 (ganancia depende de capacidad). Ademas loop de co-aprendizaje: Gemma-4 (SFT LoRA + GRPO offline + DAgger 256 pasos, PRM Gemini-3-flash, teacher Gemini-3.1-pro) avanza hitos en Red sin reset. Inconsistencias internas en el paper (26B vs 31B; bootstrap vs from-scratch en Red).

### 1.3 GPT Plays Pokemon (Clad3815, comunidad; promovido por OpenAI)
- **o3** (may-2025): stream organizado por Clad3815 (agradecido por @OpenAIDevs) [S Gigazine/foro OpenAI]. Boulder Badge 710 pasos/~12 h [S]; ~150 h con 3 medallas en otro punto [S]; Red completo en 18.184 pasos (>15 dias) [S].
- **GPT-5** (ago-2025): Red en **6.470 pasos** (~7 dias) [S TechRadar/TweakTown; PokeAgent lo cita, no lo verifica]; 3.985 interacciones para 4 medallas vs >10.000 de o3 [S MeinMMO]. Crystal: 9.517 pasos vs o3 27.040; 16 medallas en 9.205 vs 22.334 [S 36Kr/QbitAI/Habr; NV primario]. Costo ~$3.500 [NV].
- **Harness** [V via PokeAgent Tabla 2]: frame, ubicacion, stats del equipo, dinero, colores de tiles; botones + minimapa auto-construido; goal tracker + notepad; zero-shot. Emulador congelado entre acciones (no penaliza inferencia lenta).
- Un comentarista en LessWrong afirma que GPT-6 vencio FireRed en ~18 h [NV].

### 1.4 Harnesses open-source y estudios de comparacion
- **ClaudePlaysPokemonStarter** (David Hershey, Anthropic) -> fork **cicero225/llm_pokemon_scaffold** (Bradshaw, LessWrong 2025-04-27) [V]: etiquetas "CHECK HERE" en tiles no explorados, mapa de colision ASCII auto-generado con distancia en movimientos (luego coordenadas puras), `mark_checkpoint`, `detailed_navigation` (segundo modelo explora tipo DFS), autopathing; jerarquia de confianza: RAM > conocimiento propio > vision; etiquetas propias = no confiables. "helps somewhat but doesn't make LLMs amazing".
- **"Research Notes: Running Claude 3.7, Gemini 2.5 Pro, and o3 on Pokemon Red"** (2025-04-21) [V]: mismo scaffold; ~$30 por 2.000 pasos; o3: ~800 acciones en un Pokemon Center que creia otro edificio, cientos en Oak's Lab por coordenadas erroneas; Claude no distingue arboles cortables; mapas ASCII incoherentes; notas refuerzan errores; sin sentido de cuanto tardo una tarea.
- **"Is Gemini now better than Claude at Pokemon?"** (Bradshaw, 2025-04-19) [V]: Gemini ~68k acciones vs Claude ~215k; Gemini tenia minimapa textual y pathfinder a destinos lejanos; harness cambiado en vivo; juegos distintos (Blue vs Red); haria falta ~10 corridas completas. Conclusion: no prueba que Gemini sea mejor modelo.
- **PokeAgent Challenge** (Karten et al., arXiv 2603.15563, NeurIPS 2025 competition, >100 equipos) [V]: Speedrun Emerald hasta Roxanne (15 hitos). Baseline: frames + estado limitado (party, HP, status; sin puzzles ni items), MCP tools `get_game_state`, `press_buttons`, `navigate_to` (A*), KB con recuperacion ponderada, compactacion tras 20+ turnos, sub-agentes batalla/reflexion/verificacion/gym puzzle, sin wiki (contradictoria entre versiones). Humano top 18 min, promedio 1:22:05; mejor baseline (Gemini 3 Flash) ~2:24 (~1.8x humano promedio); Sonnet 4.5 completa todo pero 6:25-20:45 y 3-4x costo; ganador Heatz 40:13 / 1.608 pasos con Scripted Policy Distillation (LLM -> scripts -> NN + RL); 2do Hamburg PokeRunners PPO recurrente; Deepest 649 pasos; mejor LLM puro "Anthonys" 1:29:17. VLM crudos sin harness ~0%; harnesses CLI (Claude Code, Codex, Gemini CLI) pierden coherencia en miles de decisiones. Propone "VLM-SLAM" como reto abierto.
- **PokeAI** (arXiv 2506.23689): framework texto con agentes Planning/Execution/Critique; modulo de batalla 80.8% de victorias en 50 encuentros salvajes (6% bajo humano experto).
- **ASH** (arXiv 2605.14211): aprende de video de internet con IDM; Emerald 11.2/12 hitos en 8 h vs 9.3/12 mejor baseline.
- **Jev** (TypeSafe AI, sep-2026): motor de decision no-LLM que habria ganado Red el 2026-09-23 en <1 semana, con Claude Opus 5 "entrenandolo" via log [S Tom's Hardware, solo titular leido; NV].

---

## 2. Que regala el harness (tabla resumen)

| Proyecto | Estado RAM | Mapa | Navegacion | Memoria | Critico/sub-agentes | Intervencion humana |
|---|---|---|---|---|---|---|
| Claude 3.7 (2025) | ubicacion + estado breve | no (overlay coords) | navigator a tile visible | KB 8k tokens + resumen | Critique Claude (despues) | prompt fijo |
| Claude Opus 4.7 (2026) | ubicacion/tiles caminables/party (PokeAgent T2) | no | pathfinder | archivos, screenshots guardados, Zoom | Critic debilitado [NV] | refinamientos entre modelos |
| Gemini 2.5 Blue | mucho: RAM->texto, tipos de tile | fog-of-war con tipos de tile de RAM | pathfinder Gemini (BFS mental) + boulder solver | resumen 100/1000 turnos, metas | Guidance Gemini cada 25 | run 1 si; run 2 "autonomo" |
| Gemini 3 Crystal | posiciones de objetos, props de tiles | Mental Map (solo tiles vistos) | code exec, custom tools/agents | notepad, markers | agentes auto-creados | harness iterado entre juegos |
| GPT-5 Red/Crystal | ubicacion, party, dinero, colores de tiles | minimapa auto-construido | botones | goal tracker + notepad | ? | ? |
| Fable 5 FireRed | ninguno | ninguno | ninguno | (no publicado) | ninguno | ? |
| PokeAgent baseline | party/HP/status | no | A* `navigate_to` | KB + compactacion | batalla/reflexion/verificacion/puzzle | no |

## 3. Vicios (para ARES)
1. Harness-confound / estado privilegiado (RAM, mapas de colision, A*) -> no transfiere entre juegos sin re-ingenieria por juego.
2. Ineficiencia extrema: decenas de miles de pasos, cientos de horas, miles de millones de tokens, miles de dolares.
3. Percepcion visual/espacial debil (VLM-SLAM).
4. Bucles/stuck sin mecanismo robusto de salida.
5. Context poisoning y notas falsas auto-reforzadas; no testear hipotesis.
6. Contaminacion por conocimiento memorizado (walkthroughs) y confusion entre versiones -> invalida "generalizacion".
7. Metricas no estandarizadas ("step/turn/action"), N=1, intervencion humana, emulador pausado.
8. Planificacion de largo plazo pobre; agent panic; estrategias de batalla burdas (sobre-nivelar starter).

## 4. Ideas para ARES
- Usar LLM solo como "profesor"/generador de subgoals offline y destilar a una politica neuronal pequena (lo que gano PokeAgent: Heatz; tambien Continual Harness + Gemma).
- Representacion de estado agnostica al juego basada en pixeles + memoria espacial aprendida (mapa topologico/SLAM), no RAM especifica.
- Detector de bucles/novedad intrinseca y "reflection" explicita; contador de repeticiones de accion.
- Evaluar con protocolo estandar: hitos, pasos de entorno, tiempo de pared, costo, N>=3 semillas, sin intervencion; reportar dimensiones harness x modelo (PokeAgent).
- Test de contaminacion: juegos/ROM hacks no vistos (Yellow Legacy, traducciones) y control cross-game (como Crystal en la replica de Fable 5).

---

## 5. Videos / articulos principales
- Anthropic, "Claude's extended thinking" (feb-2025): https://www.anthropic.com/news/visible-extended-thinking
- Latent Space con David Hershey (mar-2025): https://latent.space/reading/claude-plays-pokemon/
- YouTube "Lessons on AI agents from Claude Plays Pokemon" (Anthropic, Hershey + Alex Albert): https://m.youtube.com/watch?v=CXhYDOvgpuU
- YouTube "Claude Plays Pokemon Highlights" playlist: https://www.youtube.com/playlist?list=PLrx64D7mZoumALVMlfmPMlibh0djQrFuz
- YouTube "Claude Plays Pokemon Proves We Aren't Ready For AI Agents": https://www.youtube.com/watch?v=Ux34V0MecyY
- YouTube "AIs Play Pokemon Episode 1": https://www.youtube.com/watch?v=roUAuQ3tqPk
- TechCrunch Gemini Blue (may-2025): https://techcrunch.com/2025/05/03/googles-gemini-has-beaten-pokemon-blue-with-a-little-help
- TechCrunch "Gemini panicked" (jun-2025): https://techcrunch.com/2025/06/17/googles-gemini-panicked-when-playing-pokemon
- Drew Breunig sobre apendice Gemini 2.5: https://www.dbreunig.com/2025/06/17/an-agentic-case-study-playing-pokémon-with-gemini.html
- TechRadar GPT-5 Red: https://www.techradar.com/ai-platforms-assistants/chatgpt/gpt-5-just-completed-pokemon-red-in-a-new-world-record-time-claude-gemini-and-chatgpt-o3-arent-even-close

## 6. Referencias
- https://www.anthropic.com/news/visible-extended-thinking
- https://latent.space/reading/claude-plays-pokemon/
- https://www.lesswrong.com/posts/7mqp8uRnnPdbBzJZE/is-gemini-now-better-than-claude-at-pokemon
- https://www.lesswrong.com/posts/8aPyKyRrMAQatFSnG/research-notes-running-claude-3-7-gemini-2-5-pro-and-o3-on
- https://www.lesswrong.com/posts/Qk3kCb68NvKBayHZB/open-source-llm-pokemon-scaffold
- https://github.com/cicero225/llm_pokemon_scaffold
- https://www.lesswrong.com/posts/gogZyeistdaDFuhbG/claude-plays-pokemon-opus-4-5-follow-up
- https://www.lesswrong.com/posts/sehJYg5Yny9fvpbpt/a-year-late-claude-finally-beats-pokemon
- https://www.lesswrong.com/posts/q9HoBo6LCkCkxavLZ/on-getting-unstuck
- https://rentry.co/gpp-about-20250612
- https://arxiv.org/abs/2507.06261 (Gemini 2.5 tech report, secc. 4.1 y apendice 8.2)
- https://blog.jcz.dev/gemini-3-pro-vs-25-pro-in-pokemon-crystal
- https://arxiv.org/abs/2605.09998 (Continual Harness)
- https://arxiv.org/abs/2603.15563 (PokeAgent Challenge)
- https://arxiv.org/abs/2506.23689 (PokeAI)
- https://arxiv.org/abs/2605.14211 (ASH)
- https://dev.to/qingze_hu_c4c251c1b353ede/i-replicated-the-vision-only-pokemon-run-anthropic-showcased-on-the-fable-5-launch-page-then-i-35ah
- https://phandroid.com/2026/06/11/claude-fable-5-beat-pokemon-firered-with-no-maps-or-tools-just-an-overleveled-charizard/
- https://www.techradar.com/ai-platforms-assistants/chatgpt/gpt-5-just-completed-pokemon-red-in-a-new-world-record-time-claude-gemini-and-chatgpt-o3-arent-even-close
- https://www.tweaktown.com/news/107090/chatgpt-5-just-beat-pok-mon-red-in-record-time-for-an-ai-3x-faster-than-gpt-o3-managed/index.html
- https://www.36kr.com/p/3440506883577476
- https://mein-mmo.de/en/chatgpt-has-been-playing-pokemon-on-the-gameboy-for-a-week-and-it-could-soon-make-history,1520266
- https://community.openai.com/t/livestream-o3-plays-pokemon/1270979
- https://www.pcgamer.com/software/ai/google-ai-happens-to-be-really-bad-at-playing-pokemon-repeatedly-panicking-and-taking-more-than-800-hours-to-beat-the-elite-four
- https://www.tomshardware.com/tech-industry/artificial-intelligence/developer-says-jev-decision-model-beat-pokemon-red-in-under-a-week-non-llm-engine-succeeds-where-traditional-chatbots-stalled-for-months-but-claude-opus-5-coached-the-model-through-its-dead-ends
