# Videos, streams y análisis de la comunidad sobre agentes que juegan Pokémon (corte 2026-11-27)

> **Estado: versión 1 completa (2026-11-27).** 69 entradas en 10 grupos (§1) y síntesis en §3. El encargo pedía 25-40; se priorizó cobertura porque el dueño pidió revisar "todos" los videos y streams, y las entradas de menor valor (entretenimiento, prensa) se agrupan en tablas compactas.
>
> Complementa a `ESTADO_DEL_ARTE.md` (papers y repos) con lo que faltaba: **videos, transmisiones en vivo, charlas y análisis de la comunidad**. No repite las fichas de papers; cuando un video corresponde a un trabajo ya catalogado, solo se agrega lo que el video o la discusión aportan (cifras, fallas visibles, críticas).
>
> **Marcas**: **[V]** verificado en la fuente primaria (metadatos de YouTube leídos de la página del video u oEmbed; post, hilo, README o artículo original leído); **[S]** solo fuente secundaria (prensa o resumen de un tercero); **[NV]** no verificado. Las **vistas** son las del **2026-11-27** (fecha de consulta) y cambian con el tiempo. Las citas textuales tienen menos de 15 palabras.
>
> **Método.** (1) Búsqueda en YouTube y lectura de los metadatos de cada video (título, canal, fecha de publicación, duración, vistas, descripción y capítulos) desde la página del video; (2) API de Hacker News (Algolia) para hilos y comentarios; (3) lectura directa de artículos de prensa y READMEs de los repos enlazados en los videos (`gh api`); (4) búsqueda web para lo que no se pudo abrir. **Limitación**: no se pudieron descargar transcripciones de YouTube (la API de subtítulos exige un token), así que lo que "dice" un video se toma de su descripción y capítulos, de los repos enlazados o de fuentes que lo citan; cuando no hay nada de eso, el contenido se marca [NV].

---

## 0. Resumen

1. **"El harness hace el trabajo" es el consenso más sólido de la comunidad**: lo muestran o discuten 15 de las 69 entradas, de todas las familias de fuentes (Ars Technica, TechCrunch, TIME, Engadget, Hacker News, LessWrong, creadores y charlas) (§3.a). El propio autor de Gemini Plays Pokémon pide no usarlo como benchmark.
2. **Las afirmaciones de "sin scripts" o "solo visión" casi nunca resisten la lectura del README.** Jev ("No scripts, no cheats", 37 h 40 min para ganar Red) lee la RAM, usa A\*, conoce 33 metas escritas a mano, filtra y anota las opciones y recarga checkpoints [V]. PokéBench dice "no hints" pero entrega "the game state" en su categoría estándar [V].
3. **"Terminar el juego" no prueba percepción ni aprendizaje.** Según su autor, una lista fija de botones escrita a mano vencería **en teoría** más del 99 % de las partidas de FireRed sin mirar la pantalla; el propio README advierte que la lista completa está *"technically untested"* (H03, 2.2 M de vistas). En cambio, la entrada aleatoria no sale de Pallet Town ni consigue medallas en años (R02, H02). Esto respalda que ARES-U mida **generalización** y no completitud.
4. **La variabilidad entre corridas es grande y casi nunca se mide.** Hershey admite que la mayoría de las veces Claude no arma una estrategia coherente (C06). Un creador con muchas corridas por modelo muestra resultados agrupados por modelo, pero con mucha dispersión (C11). El único proyecto comunitario con 100 semillas y dos líneas base explícitas es un world model diminuto (R13). Casi todo lo demás es N=1.
5. **Las corridas cambian por dentro**: el harness de Claude se modificó entre intentos (X02, X03). El intento VIII que "ganó" con Opus 4.7 aparece etiquetado como Opus 4.6 en su primer VOD (C08). En una carrera de Crystal se cambiaron modelos, se dieron pistas escritas y se concedieron recargas (E05). El récord de GPT-6 Astra es el mejor de cuatro configuraciones (P06).
6. **Desde 2026 aparecen "benchmarks de creador"** (PokéBench, Andreas Astrup, Devon7925, Finn Mertens) con mismo harness, presupuesto fijo, hitos automáticos y logs públicos. Son la mejor práctica comunitaria encontrada, aunque con 100 a 1,000 vistas.
7. **La audiencia está en otro lado.** Los videos más vistos (Whidden 10.2 M; Purple Thunder 2.4 M; MartSnack 2.2 M; Poepeo 1.3 M) son el RL de 2023, entretenimiento sin método declarado o retos con un humano que percibe y ejecuta. En español casi no hay análisis técnico: el material es de reacción o de entretenimiento, más un TFG (R11).
8. **En combate el debate es ético, no técnico**: el bot que llegó al n.º 1 del ladder (Jaxcalibur) retiene código y pesos por miedo a que se use para hacer trampa (B01, B02). Esto explica una falta de reproducibilidad que en `ESTADO_DEL_ARTE.md` aparecía sin causa.

---

## 1. Índice del catálogo

| ID | Título (abreviado) | Canal / autor | Fecha | Tipo | Enfoque | Juego |
|---|---|---|---|---|---|---|
| R01 | Training AI to Play Pokemon with Reinforcement Learning | Peter Whidden | 2023-10-08 | video | RL | Red |
| R02 | Show HN: Beating Pokemon Red with RL and <10M Parameters | drubs (D. Rubinstein) en HN | 2025-03-05 | hilo | RL | Red |
| R03 | Mapa comunitario `pokerl-map-viz` | PWhiddy / comunidad PokeRL | 2024-25 | stream web | RL | Red |
| R04 | AI Plays Pokemon with Reinforcement Learning | Wes Roth | 2023-10-16 | video de comentario | RL | Red |
| R05 | Inteligencia Artificial aprende a jugar a Pokémon | Momento Guinxu | 2024-09-27 | video (ES) | RL | Red |
| R06 | Pokemon red AI ... exclusively intrinsic rewards | Álvaro Álvarez Iglesias | 2023-05-29 | video | RL (novedad) | Red |
| R07 | I Trained an AI to Speedrun Pokémon | A Goofy Intelligence | 2026-08-29 | video | RL | Red |
| R08 | My AI Taught Itself Pokémon Red. No Hints, No Walkthrough. | PAIOS Plays | 2026-11-25 | video | RL | Red |
| R09 | Can We Build an AI to Beat Pokémon Red? (serie de streams) | Joe the AI Guy | 2024-10-11 → | streams | RL | Red |
| R10 | How Speeding Up RL Led to Pufferlib, with Joseph Suarez | Deep Learning with Yacine | 2026-01-04 | entrevista | RL (infra) | — |
| R11 | PokemanIA - Una IA que juega Pokémon | Ramien (TFG, Univ. de les Illes Balears) | 2025-06-30 | video (ES) | RL + visión | Red |
| R12 | Inteligencia Artificial Aprende a Jugar Pokemon Stadium | El Maikini Dev | 2023-04-26 | video (ES) | RL en un minijuego recreado | Stadium (minijuego) |
| R13 | Teaching a World Model to Play Pokemon (+ hilo HN) | stmonty | 2026-09-20 / 09-25 | blog + hilo | world model (JEPA) + planificación | Red |
| C01 | Twitch `claudeplayspokemon` + hilo HN | Anthropic / HN | 2025-02-25 → | stream + hilo | LLM+harness | Red |
| C02 | How Claude Plays Pokémon was made | Latent Space | 2025-03-03 | podcast | LLM+harness | Red |
| C03 | I Let An AI Play Pokémon! (MLOps Podcast #297) | AAIF Live | 2025-03-05 | podcast | LLM+harness | Red |
| C04 | Claude Plays Pokémon Hackathon: Escape from Mt. Moon! | Latent Space | 2025-04-05 | evento/stream | LLM+harness | Red |
| C05 | Lessons on AI agents from Claude Plays Pokemon | Anthropic | 2025-04-24 | charla | LLM+harness | Red |
| C06 | Why Anthropic's Claude still hasn't beaten Pokémon | Kyle Orland, Ars Technica | 2025-03-21 | prensa | LLM+harness | Red |
| C07 | Claude Plays Pokémon Proves We Aren't Ready For AI Agents | Elodine | 2025-03-30 | video de análisis | LLM+harness | Red |
| C08 | Highlights #1-#3, clips y VODs | pookie, tendog, Kurukkoo | 2025-02 → 2026-06 | clips/VOD | LLM+harness | Red |
| C09 | La Vez Que la IA Intentó Jugar Pokémon (Parte 1) | PKMN Champion Emerald | 2025-05-03 | video (ES) | LLM+harness | Red |
| C10 | Claude Fable 5 beats Pokémon FireRed only using vision | Claude (canal oficial) | 2026-06-09 | video (timelapse) | LLM solo visión | FireRed |
| C11 | An Unreasonable Amount of Data on How Well Claude Plays Pokémon Red | Finn Mertens | 2026-11-15 | video de datos | LLM+harness | Red |
| G01 | Twitch `gemini_plays_pokemon` + Google (Pichai, Kilpatrick) | Joel Zhang / Google | 2025-03 → | stream + X | LLM+harness | Blue, Crystal, Yellow Legacy |
| G02 | Google's Gemini has beaten Pokémon Blue (with a little help) | Anthony Ha, TechCrunch | 2025-05-03 | prensa | LLM+harness | Blue |
| G03 | Why Google Gemini's Pokémon success isn't all it's cracked up to be | Kyle Orland, Ars Technica | 2025-05-05 | prensa | LLM+harness | Blue |
| G04 | Gemini Plays Pokémon -- Indigo Plateau & Credits | HitokageWorshipper | 2025-05-09 | VOD | LLM+harness | Blue |
| G05 | Gemini Plays Pokémon Situation: Did Agentic AI Get Better...? | Elodine | 2025-06-01 | video de análisis | LLM+harness | Blue |
| G06 | The Making of Gemini Plays Pokemon - Joel Zhang | PokeAgent | 2025-10-26 | charla | LLM+harness | Blue/Crystal |
| G07 | Gemini Plays Pokemon Crystal - Full Game Timelapse | MrCheeze | 2026-01-26 | timelapse | LLM+harness | Crystal |
| P01 | nunu.ai plays Pokémon Emerald! // Gym 1 | nunu ai | 2024-07-20 | stream | LLM+harness | Emerald |
| P02 | Can ChatGPT Beat Pokémon? | PaPaSea | 2024-05-25 | video | humano + LLM guía | Red/Blue |
| P03 | I Let ChatGPT Play Pokémon Fire Red! | Poepeo | 2025-12-23 | video | humano + LLM decide | FireRed |
| P04 | GPT 6 Astra Plays Pokémon Red without RAM Access | Devon7925 | 2026-09-08 | video | LLM solo visión | Red |
| P05 | Can ChatGPT play Pokémon Crystal? (with GPT-4V) | piman's dumping ground (pimanrules) | 2023-11-19 | video | LLM con visión | Crystal |
| P06 | GPT Plays Pokémon: posts de OpenAI Devs y Clad3815 | OpenAI Devs / Clad3815 (X + Twitch) | 2025-05-27 → 2026-09-03 | posts + stream | LLM+harness | Red, Crystal, FireRed |
| M01 | Which AI model is ACTUALLY the best at playing Pokémon? / GPT-5.6 Sol vs Claude Opus 5 | Andreas Astrup | 2026-06-21 / 2026-09-05 | videos | comparación LLM | FireRed |
| M02 | PokéBench (videos y streams) | PokéBench | 2026-09 → 2026-11 | benchmark + streams | comparación LLM | Red |
| M03 | Why the World's Best AI Systems Are Still So Bad at Pokémon | Tharin Pillay, TIME | 2026-01-13 | prensa | LLM+harness | varios |
| M04 | Jev Plays Pokémon Red (LIVE) | Frigade (C. Mathiesen) | 2026-09-25 | stream 38 h | modelo de decisión + harness | Red |
| T01 | PokéAgent Challenge: Research Talks & Hackathon Launch | PokeAgent | 2025-09-13 | charlas | RL + LLM + búsqueda | Emerald / Showdown |
| T02 | PokéChamp: An Expert-level Minimax Language Agent | PokeAgent (Seth Karten) | 2025-11-05 | charla | LLM + minimax | Showdown Gen9 |
| T03 | Streams oficiales del PokéAgent Challenge (Showdown y speedrun) | PokeAgent | 2025-10 → 2026-03 | streams | varios | Emerald / Showdown |
| B01 | I Fought The Most Terrifying AI in Pokemon Showdown History (Jaxcalibur) | blunder | 2026-08-30 | video | RL + búsqueda | Showdown Gen9 Random |
| B02 | Debate sobre bots en el ladder humano | Jimothy Cool, Mr Craw, DriftingTheTrainer, Smogon | 2026-08 → 2026-10 | videos + foro | — | Showdown / VGC |
| B03 | We Trained an AI to Play Pokémon Doubles (And it's scary) | Wisp Coach | 2026-11-15 | video | no declarado | VGC (Reg. M-C) |
| B04 | ChatGPT vs Google Gemini POKEMON BATTLE!! | ZXMany | 2026-02-07 | video | LLM | reglas de Stadium |
| B05 | I Created the Ultimate Pokémon AI (and used it to cheat) | Astroid Videos | 2026-11-22 | video | no declarado | torneo |
| B06 | Jev contra Opus 5 en un combate competitivo | Siddharth Arya (X) | 2026-09-17 | post | modelo de decisión vs LLM | Showdown |
| X01 | Show HN: LLM plays Pokémon (open sourced) | adenta en HN | 2025-02-26 | hilo | LLM+harness | FireRed |
| X02 | So how well is Claude playing Pokémon? | Julian Bradshaw (LessWrong) | 2025-03-07 | blog + comentarios | LLM+harness | Red |
| X03 | Insights into Claude Opus 4.5 from Pokémon | Julian Bradshaw (LessWrong) | 2025-12-09 | blog + comentarios | LLM+harness | Red |
| X04 | Gemini 3 Pro vs. 2.5 Pro in Pokemon Crystal (hilo) | Hacker News | 2025-12-16 | hilo | LLM+harness | Crystal |
| X05 | Show HN: Jev Plays Pokémon Red (hilo) | Hacker News | 2026-09-25 | hilo | modelo de decisión + harness | Red |
| X06 | Posts en X de Anthropic, Google y OpenAI | AnthropicAI, Pichai, Kilpatrick, OpenAIDevs | 2025-02 → 2025-08 | posts | — | Red, Blue |
| X07 | Prensa general (8 medios) | TechCrunch, Engadget, Futurism, Tom's Guide, GamesRadar, Xataka, mixed-news, Tom's Hardware | 2023-10 → 2026-09 | prensa | — | varios |
| X08 | Reddit y foros cerrados | r/ClaudePlaysPokemon, r/OpenAI, Smogon | 2025-26 | hilos | — | — |
| E01 | Videos de Purple Thunder (3) | Purple Thunder | 2026-07 → 2026-09 | videos | no declarado | — |
| E02 | Retos "la IA juega por mí" en español | SuperO4LIVE, aalecard, Paco Rodríguez | 2024-01 → 2026-10 | videos (ES) | humano + LLM | FireRed y otros |
| E03 | Can My AI BEAT Pokémon Fire Red? | PokeTech | 2026-11-11 | video | LLM (Grok) | FireRed |
| E04 | AIs Play Pokemon Episode 1 | AI World Championships | 2026-04-15 | video | comparación LLM | — |
| E05 | Pokemon Crystal Playoff (4 partes) | Chris Izatt | 2026-10-01 → 10-04 | streams | comparación LLM | Crystal |
| H01 | Twitch Plays Pokémon (2014) y sus historias | Twitch / TheSmithPlays / Darkraimola | 2014 → 2025 | stream + videos | multitud humana | Red |
| H02 | Pi Plays Pokémon Sapphire | WinningSequence | ~2022 → 2026 | stream 24/7 | entradas pseudoaleatorias (dígitos de π) | Sapphire |
| H03 | Can you beat Pokemon FireRed while blind and deaf? | MartSnack (K. Ghoreshi) | 2022-10-08 | video + repo | secuencia fija escrita a mano | FireRed |
| H04 | Pokemon Red / Blue Tool Assisted Speedrun History Explained | Abyssoft | 2021-08-27 | video | TAS | Red/Blue |
| H05 | Elo World: Pokémon Red/Blue AI Tournament | pimanrules | 2020-06-15 | video | IA de NPC del juego | Red |
| H06 | Te ENSEÑO como FUNCIONA la IA de Pokémon | Yellowpop | 2025-11-19 | video (ES) | IA de NPC del juego | GBA/DS |

---

## 2. Fichas

### 2.1 Agentes de RL (pregunta 1 y 2 del encargo)

#### R01. "Training AI to Play Pokemon with Reinforcement Learning" [V]
- **Canal**: Peter Whidden (@peterwhidden). **Fecha**: 2023-10-08. **Tipo**: video, 33:52. **Vistas**: 10,213,932. **URL**: https://www.youtube.com/watch?v=DcYLT37ImBY
- **Enfoque / juego**: RL (PPO de Stable-Baselines3 sobre PyBoy), Pokémon Red. El canal tiene **un solo video**; no hay secuelas en YouTube [V, listado del canal].
- **Qué muestra.** Los capítulos del propio video resumen el relato: *Exploration, distraction* (4:04), *Level reward* (5:46), *PC Trauma* (8:44), *Healing* (10:10), *Gym Battle* (10:45), *Mt Moon* (14:44), *Map Visualizations* (15:54), *RNG manipulation* (18:53), y una parte técnica: *Simplify*, *Efficient Iteration*, *Environment, Reward function*, *Metrics & Visualization*, *Future Improvements* [V, descripción]. La cifra de ">50,000 h" de juego simulado viene de TechCrunch [S] (ya en `notas/rl_agents.md` §3).
- **Vicios que revela.** *Noisy TV* (la novedad por frames lo deja mirando agua y pasto animados); aversión duradera tras un evento negativo ("PC Trauma"); recompensa hecha a mano por nivel, curación y exploración; arranque desde un savestate posterior al parcel de Oak porque el agente no lee diálogos [V en el repo; ver `notas/rl_agents.md`]. El contenido de "RNG manipulation" no se pudo comprobar sin transcripción [NV].
- **Secuela (no en video).** En TIME (2026-01-13) Whidden resume el contraste con los LLM: "It knows what it's supposed to do, but it bumbles the execution." [V, artículo M03].
- **Eco en HN**: al menos 8 envíos del video en octubre de 2023 y uno en enero de 2025, todos con poca discusión (máximo 11 puntos) [V, API de HN]. La discusión técnica fuerte llegó con PokeRL (R02).

#### R02. Hilo "Show HN: Beating Pokemon Red with RL and <10M Parameters" [V]
- **Autor**: `drubs` (David Rubinstein). **Fecha**: 2025-03-05. **Tipo**: hilo de Hacker News, 183 puntos, 68 comentarios. **URL**: https://news.ycombinator.com/item?id=43269330 (enlaza a https://drubinstein.github.io/pokerl/)
- **Afirmación**: el sistema "successfully completes the game" con una política de menos de 10 M parámetros y PPO; el post se publica explícitamente aprovechando el lanzamiento de Claude Plays Pokémon [V].
- **Críticas y datos que aparecen en el hilo** [V, comentarios leídos]:
  - *La recompensa contrabandea conocimiento*: un usuario (`levocardia`) observa que zonas críticas (Team Rocket, Safari Zone) dependen de codificar conocimiento del juego en la recompensa, que "smuggles in" inteligencia externa. El autor responde que parte de eso (enseñar y usar Cut) probablemente podría hacerlo un LLM.
  - *Reward hacking confirmado por el autor*: sin recompensa extra en zonas clave, los envs "wander the world reward hacking"; con combates salvajes activos, terminaban en Lavender Tower peleando contra Gastly.
  - *El combate no se aprende de verdad*: un colaborador (`xinpw8`) admite que en combate el agente "spamming options + picking damaging attacks".
  - *Línea base aleatoria*: con entradas aleatorias el agente no sale de Pallet Town tras miles de millones de pasos (`xinpw8`); con cómputo enorme no llega al Viridian Forest (`tehsauce`). Otro usuario cita "pi plays Pokemon Sapphire": más de 3 años sin medallas.
  - *Recompensa por leer carteles*: el autor reconoce que es "silly"; servía para empujar la exploración del Safari Zone.
  - *Falta de paper*: "Really missing the arxiv link" (`KeplerBoy`).
  - *Costo frente a LLM*: varios comentarios contrastan el costo de un PPO pequeño con el de Claude, todavía atascado.
- **Relevancia**: es la fuente comunitaria más clara de que "vencer Red" con RL dependió de recompensas y scripts específicos del juego (V3, V5), ya documentados en el sitio del proyecto.

#### R03. Mapa comunitario en vivo `pokerl-map-viz` [V]
- **Autor**: PWhiddy y la comunidad del Discord de PokeRL. **URL**: https://pwhiddy.github.io/pokerl-map-viz/ (página "Pokemon Red Map RL Visualizer"; repo `PWhiddy/pokerl-map-viz`, 57 estrellas, creado el 2024-01-14 y actualizado el 2026-10-19) [V]. **Tipo**: "stream" web que muestra cientos de agentes de RL de distintas corridas sobre un mapa compartido en tiempo real, según `tehsauce` en HN (2025-02-27 y 2025-03-06) [V comentario; el contenido en vivo no se inspeccionó].
- **Qué aporta**: hace visible el **swarming** y la diversidad de estados entre envs; es la forma en que la comunidad RL "transmite" sus corridas en lugar de Twitch.
- **Vicio asociado**: lo que se ve es la nube de envs, no una partida continua (V6).

#### R04. "AI Plays Pokemon with Reinforcement Learning" (Wes Roth) [V metadatos; contenido NV]
- **Fecha**: 2023-10-16. **Tipo**: video de comentario, 14:25. **Vistas**: 28,397. **URL**: https://www.youtube.com/watch?v=m4g_UT1HbpE
- **Enfoque**: divulgación sobre el proyecto de Whidden (enlaza a natural20.com/pokemon-ai). Es un ejemplo de la ola de videos de reacción que amplificó R01 en octubre de 2023. Contenido no revisado [NV].

#### R05. "Inteligencia Artificial aprende a jugar a Pokémon" (Momento Guinxu) [V metadatos]
- **Fecha**: 2024-09-27. **Tipo**: video en español (canal de clips del youtuber Guinxu), 5:35. **Vistas**: 18,385. **URL**: https://www.youtube.com/watch?v=85TRyIGwWRs
- **Enfoque**: reacción en español al video de Whidden (lo enlaza en la descripción); describe la IA como aprendiendo "mediante distintos refuerzos positivos" [V, descripción].
- **Valor para el informe**: muestra que el relato hispanohablante del tema se apoya casi por completo en R01; no se encontró un proyecto de RL hispanohablante con resultados propios comparables (búsqueda negativa; R11 es un TFG sin cifras publicadas; ver §4).

#### R06. "Pokemon red AI using reinforcement learning trained on exclusively intrinsic rewards" [V]
- **Canal**: Álvaro Álvarez Iglesias. **Fecha**: 2023-05-29 (cuatro meses **antes** que Whidden). **Tipo**: video corto, 1:01. **Vistas**: 816. **URL**: https://www.youtube.com/watch?v=XFjtxeT_iLU
- **Enfoque**: RL con **solo recompensa intrínseca**, entrada de imagen y novedad por "counting after hashing" [V, descripción].
- **Valor**: es un precedente directo del enfoque de E2 de ARES-U (novedad sin recompensa extrínseca del juego). No reporta cifras de progreso [NV]; al ser un clip de 1 minuto, sirve como existencia de la idea, no como evidencia de que funcione.

#### R07. "I Trained an AI to Speedrun Pokémon" (A Goofy Intelligence) [V descripción]
- **Fecha**: 2026-08-29. **Tipo**: video, 14:34. **Vistas**: 3,135. **URL**: https://www.youtube.com/watch?v=FUKhtyk8Rqg
- **Enfoque / juego**: RL en Pokémon Red, con la meta de vencer a Brock y luego acelerar la ruta. Capítulos: *The Route*, *The Engine*, *New Emulator Runs*, *Viridian Forest and Brock*, *Speeding It Up*, *The New Route* [V].
- **Cifra**: la primera corrida exitosa contra el primer líder tomó "roughly 14 months of gameplay" equivalentes [V, descripción].
- **Vicios**: costo de muestras enorme incluso para el primer gimnasio (V14); "The Route" sugiere una ruta definida por el autor (V3/V5) [NV sin ver el video].

#### R08. "My AI Taught Itself Pokémon Red. No Hints, No Walkthrough." (PAIOS Plays) [V descripción]
- **Fecha**: 2026-11-25. **Tipo**: video, 8:03, episodio 1 de una serie. **Vistas**: 157. **URL**: https://www.youtube.com/watch?v=a1DN4qOgFFI
- **Enfoque**: RL que "solo suma puntos" por lugar nuevo, subida de nivel y medalla.
- **Cifras**: vence a Brock tras **76 millones de movimientos** ("about a year of non-stop play"); todas las cifras salen de "40-game test runs" [V, descripción].
- **Vicios**: el propio autor titula un capítulo *The AI cheats* y dice que el agente "finds a loophole in my rules" (reward hacking, V4); costo de muestras (V14). **Lección positiva**: reportar sobre 40 partidas de prueba en lugar de N=1 (V12). La narración es una voz sintética (ElevenLabs).

#### R09. "Can We Build an AI to Beat Pokémon Red?" (Joe the AI Guy) [V metadatos]
- **Fechas**: desde 2024-10-11 (episodio 1, 1:41:56; 876 vistas); siguen streams como "Crushing Boulder City" (11:55:00) y "Deepseek's Code Changes Let Us Down" [V búsqueda]. **URL ep. 1**: https://www.youtube.com/watch?v=d3tyxtLME6Y
- **Enfoque**: live coding de RL con Python, PyBoy, Stable Baselines y PyTorch [V, descripción]. El título de un stream indica que usó un LLM (DeepSeek) para modificar el código del entorno [V título; detalle NV].
- **Valor**: muestra el ciclo real de un aficionado (iterar recompensas en vivo); no hay resultados agregados.

#### R10. "How Speeding Up RL Led to Pufferlib (4.8K Stars) | with Joseph Suarez" [V metadatos]
- **Canal**: Deep Learning with Yacine. **Fecha**: 2026-01-04. **Tipo**: entrevista, 1:05:33. **Vistas**: 3,365. **URL**: https://www.youtube.com/watch?v=Sirpfci74zU
- **Enfoque**: el origen de PufferLib (la infraestructura de PokeRL y PokeGym). Joseph Suarez también transmite su propio desarrollo ("Live RL Research on PufferLib", varios streams de 3 a 7 horas en 2025) [V búsqueda]. No se pudo confirmar en qué minuto se habla de Pokémon [NV].
- **Valor**: contexto de por qué el RL de Pokémon depende de entornos muy rápidos (~10k SPS) y de cómputo donado (V14).

#### R11. "PokemanIA- Una IA que juega Pokémon" (Ramien) [V metadatos]
- **Fecha**: 2025-06-30. **Tipo**: video en español, 2:13. **Vistas**: 123. **URL**: https://www.youtube.com/watch?v=btNxnf9FsZg
- **Qué es** [V, descripción]: presentación de un **Trabajo de Fin de Grado** de Ingeniería Informática de la Universitat de les Illes Balears; combina RL y visión por computador para entrenar un agente en Pokémon Rojo, y enlaza el video de Whidden como referencia. No hay cifras en la descripción [NV].
- **Valor**: es el único proyecto académico hispanohablante encontrado en video; confirma que el formato "RL a lo Whidden" es el punto de partida típico de un proyecto de curso, igual que en ARES-U.

#### R12. "Inteligencia Artificial Aprende a Jugar Pokemon Stadium" (El Maikini Dev) [V metadatos]
- **Fecha**: 2023-04-26. **Tipo**: video en español, 5:05. **Vistas**: 16,938. **URL**: https://www.youtube.com/watch?v=nIVqH0m-qyk
- **Qué es** [V, descripción]: el autor **recrea un minijuego** de Pokémon Stadium y entrena una IA en esa recreación (capítulos: escenario, animaciones, rocas, vida, entrenamiento, batalla).
- **Vicio**: el entorno es una reimplementación propia, no el juego; el resultado no dice nada sobre el juego original (análogo a V11 y al vicio propio A3 de ARES-PKMN).

#### R13. "Teaching a World Model to Play Pokémon" (stmonty) y su hilo en HN [V]
- **Autor**: stmonty. **Fechas**: post del 2026-09-20 (según Tom's Hardware, 2026-09-30); hilo HN del 2026-09-25 (24 puntos, 18 comentarios). **URL**: https://nostalgia.dev/posts/teaching-a-world-model-to-play-pokemon/ · https://news.ycombinator.com/item?id=49849907 . Código: `lePokeRed` (citado por Tom's Hardware) [S].
- **Enfoque**: world model estilo JEPA (LeWorldModel), ~12.5 M de parámetros, entrenado desde cero en una RTX 3080 Ti con **42,382 frames** en **1,009 trayectorias** cortas (rutas scripted, rutas con ruido y paseo aleatorio); planifica secuencias de 14 botones en el modelo y ejecuta la mejor en el emulador; la meta es el embedding de una captura con el inicial elegido [V, blog].
- **Resultado** [V, blog]: tarea mínima (elegir un inicial desde un savestate en el laboratorio de Oak); **52 de 100 planes** (100 semillas) consiguen un inicial, frente a **0 con botones aleatorios** y **1 con el predictor sin entrenar**. El autor admite que "pressing A repeatedly already works" desde ese estado y sospecha que la dificultad crece exponencialmente con el largo del plan.
- **Lección positiva**: es el único proyecto comunitario encontrado que reporta **tasa sobre 100 semillas con dos líneas base explícitas** (aleatorio y sin entrenar); justo el tipo de protocolo de ARES-U (V12).
- **Vicios**: datos con rutas scripted (V5 suave), inicio desde savestate (V6), horizonte diminuto. En HN, un comentario le recuerda que PPO ya ganó el juego completo con una red pequeña, y otro comentario cierra con una alusión a "The Bitter Lesson".

---

### 2.2 Claude Plays Pokémon (pregunta 3)

#### C01. Stream de Twitch `claudeplayspokemon` y su hilo en HN [V]
- **Autor**: Anthropic (David Hershey). **Inicio**: 2025-02-25 (lanzamiento con Claude 3.7 Sonnet). **URL**: https://www.twitch.tv/claudeplayspokemon . **Hilo HN**: "Claude Plays Pokémon", 75 puntos, 24 comentarios, https://news.ycombinator.com/item?id=43173825 [V].
- **Qué dice la comunidad en el hilo** [V]:
  - Lleva casi un día atascado en Mt. Moon, olvida dónde estuvo y casi siempre huye de los combates.
  - Confundió el gorro rojo del personaje con la alfombra de salida del Pokécenter.
  - Un usuario señala que el harness "is already cheating a little" porque hay un segundo LLM de verificación y compactación frecuente del contexto.
  - Otro pide comparar con Twitch Plays Pokémon: "Eleven years from TPP..." (`Philpax`).
  - Otro sospecha fuga de datos de entrenamiento ("leakage the entire corpus"), sin evidencia.
- **Datos de audiencia**: unos 2,000 espectadores en las primeras semanas (TIME, M03) [V artículo]. En mayo de 2026 iba en el **octavo intento** de la serie (descripción de un VOD de Kurukkoo, C08) [V].

#### C02. "How Claude Plays Pokémon was made" (Latent Space) [V]
- **Fecha**: 2025-03-03. **Tipo**: podcast en video ("lightning pod") con David Hershey, 37:37. **Vistas**: 6,068. **URL**: https://www.youtube.com/watch?v=DHP0WUEOqik . Versión escrita ya catalogada en `ESTADO_DEL_ARTE.md` (`latent.space/reading/claude-plays-pokemon`).
- **Aporte nuevo**: la descripción enlaza el **diagrama completo del harness** en Excalidraw, la misma pieza que en HN se discutió por la frase "parsed directly from the RAM" (ver §2.7) [V].

#### C03. "I Let An AI Play Pokémon! - Claude plays Pokémon Creator" (MLOps Podcast #297) [V metadatos]
- **Canal**: AAIF Live (MLOps Community, Demetrios con David Hershey). **Fecha**: 2025-03-05. **Tipo**: podcast, 46:58. **Vistas**: 235. **URL**: https://www.youtube.com/watch?v=nRHeGJwVP18
- **Temas declarados**: frameworks de agentes, optimizar prompts frente a fine-tuning, plataformas gestionadas [V, descripción]. Sin cifras propias.

#### C04. "Claude Plays Pokémon Hackathon: Escape from Mt. Moon!" (Latent Space) [V]
- **Fecha**: 2025-04-05. **Tipo**: evento presencial en SF transmitido, 1:15:00. **Vistas**: 1,507. **URL**: https://www.youtube.com/watch?v=zBPc6Ims1Bc
- **Qué es**: hackathon para construir "the Best Pokemon Agent" que salga de Mt. Moon, con créditos de Claude de Anthropic y el harness de Morph Labs (código en `morph-labs/morphcloud-examples-public`, carpeta `pokemon-example`) [V, descripción; el repo hoy da 404].
- **Vicio que ilustra**: la tarea se define como **escapar de un cuello de botella concreto**; lo natural sería partir de un savestate en Mt. Moon, pero la descripción no lo dice [NV]. Es el tipo de meta que mide harness + prompt sobre un tramo, no capacidad general (V1).

#### C05. "Lessons on AI agents from Claude Plays Pokemon" (Anthropic) [V]
- **Fecha**: 2025-04-24. **Tipo**: charla, Alex Albert y David Hershey, 44:50. **Vistas**: 25,024. **URL**: https://www.youtube.com/watch?v=CXhYDOvgpuU
- **Capítulos** [V]: *How Claude plays Pokemon technically* (4:30), *Memory systems and long-term storage* (10:20), *Evolution across different model versions* (14:00), *How Pokemon success translates to agentic capabilities* (17:15), *Funny failures and current limitations* (23:15), *Community response* (30:40), *Advice for building agents* (42:00).
- **Lectura crítica**: el encuadre es corporativo (Pokémon como proxy de "capacidades agénticas"); la generalización de Pokémon a trabajo real es una **tesis del canal, no un resultado medido** (vicio de antropomorfización/extrapolación, ver §3).

#### C06. "Why Anthropic's Claude still hasn't beaten Pokémon" (Kyle Orland, Ars Technica) [V]
- **Fecha**: 2025-03-21. **Tipo**: prensa con entrevista a Hershey. **URL**: https://arstechnica.com/ai/2025/03/why-anthropics-claude-still-hasnt-beaten-pokemon/
- **Datos y citas** [V, texto leído]:
  - El harness monitorea direcciones clave de la RAM además de la imagen.
  - Hershey: "Claude's still not particularly good at understanding what's on the screen at all".
  - Memoria: contexto de 200,000 tokens; al llenarse, se resume y se pierden detalles; a veces borra lo que no debía.
  - **Envenenamiento de la base de conocimiento**: de lo que anotó antes, Hershey dice que "it sort of trusts pretty blindly"; pasó horas explorando un cuadrado de coordenadas erróneas creyendo que era la salida del Viridian Forest.
  - **Variabilidad entre corridas**: a veces arma una estrategia coherente, pero "most of the time it doesn't".
  - Salida de Mt. Moon tras **más de 78 horas**.
  - Hershey cree que podría ganar con percepción perfecta de la pantalla.
- **Comentarios de lectores** (mismo artículo) [V]: loops de 6-8 h en Vermilion buscando el S.S. Anne, repitiendo el mismo circuito "20 times in a row"; otro lector pide probarlo en un juego "purged of pokemon" para descartar memorización; otro reporta que clones con modelos de Google y OpenAI **alucinaban el diálogo de NPCs** y creían que el juego estaba roto.

#### C07. "Claude Plays Pokémon Proves We Aren't Ready For AI Agents" (Elodine) [V metadatos]
- **Fecha**: 2025-03-30. **Tipo**: video de análisis, 16:34. **Vistas**: 50,327. **URL**: https://www.youtube.com/watch?v=Ux34V0MecyY
- **Tesis** [V, descripción]: el stream muestra por qué la IA agéntica no está lista; destaca quedar atascado en menús y **liberar Pokémon** por error; lo compara con el caos de Twitch Plays Pokémon y con el *hype cycle* de Gartner.
- **Vicios**: menu spam y acciones irreversibles (V9); el video también ilustra el riesgo inverso: usar un N=1 de entretenimiento para concluir sobre "agentes" en general (V12).

#### C08. Clips, highlights y VODs de la comunidad [V metadatos]
- **pookie**, "Claude Plays Pokemon Highlights: Is AGI Here Yet?" (2025-03-09, 25:06, 9,137 vistas, https://www.youtube.com/watch?v=pAkElOzD5Rw), #2 (2025-03-16, 3,185 vistas, https://www.youtube.com/watch?v=L_rcgv4qbWM) y #3 "Is AGI Confirmed?" (2025-03-29, 2,300 vistas, https://www.youtube.com/watch?v=ZYjmjyREUiQ). Describen límites en "spatial awareness, memory management, and multitasking" y un Claude que intenta una y otra vez hablar con rocas [V, descripciones]. Los títulos con "AGI" son ironía, pero muestran el encuadre antropomórfico dominante.
- **tendog**, "Claude uses Escape Rope to leave Mt. Moon (with chat)" (2025-02-26, 6:15, 8,763 vistas, https://www.youtube.com/watch?v=4panxmPVTjI) [V].
- **Kurukkoo**, "Claude Jr's Blackout Strategy, Mt. Moon shenanigans, and his escape!" (2025-12-12, 39:52, https://www.youtube.com/watch?v=-aHJ9IB-zmw): reconstruye el VOD perdido del 2025-03-07 sincronizando chat y clips; Twitch borró los metadatos del VOD [V, descripción]. La "blackout strategy" (desmayar al equipo a propósito para volver al Pokécenter) es el ejemplo canónico de **estrategia derivada de una creencia falsa** (V17). Kurukkoo también archiva VODs de 12 h de 2026 ("Claude Plays Pokémon VOD 5-15-26 Part 4", https://www.youtube.com/watch?v=i6jFyABObtE), cuya descripción indica que es el **octavo intento** y que el harness permite navegar automáticamente a una casilla visible [V].
- **El final del intento VIII**: "[FINALE] Claude Plays Pokémon VOD 5-17-26 Part 4 - Claude VIII completes his mission" (https://www.youtube.com/watch?v=8fbvzyYJueY): tras vencer a la Elite Four y a Blue y capturar a Mewtwo, Claude sigue completando la Pokédex; la descripción indica **Claude Opus 4.7** y el título original del stream de Twitch [V]. Pero el primer VOD del mismo intento ("VOD 5-3-26 Part 1", https://www.youtube.com/watch?v=O-UyiTLAZg0) lleva "Opus 4.7 (VIII)" en el título y, en la descripción, "Claude Opus 4.6" y el título de Twitch "Claude Opus 4.6 Plays Pokemon!" (VOD de 47:59:56, 10,261 vistas y 9,182 mensajes de chat) [V]. Es decir, **el intento que "ganó" con Opus 4.7 parece haber empezado con Opus 4.6**; no se verificó cuándo ni cómo se hizo el cambio [NV]. Si se confirma, la victoria mezcla dos modelos en una misma partida (V12, V13).
- **Valor**: estos archivos son la única forma de auditar corridas cuyo VOD oficial desapareció; sin ellos, las anécdotas (alfombra, blackout) y la etiqueta de modelo de cada tramo no tendrían registro.

#### C09. "La Vez Que la Inteligencia Artificial Intentó Jugar Pokémon - Claude Plays Pokémon (Parte 1)" [V metadatos]
- **Canal**: PKMN Champion Emerald. **Fecha**: 2025-05-03. **Tipo**: video en español (estreno), 1:07:18. **Vistas**: 115. **URL**: https://www.youtube.com/watch?v=MkYqQR2dKr0
- **Contenido**: recopilación comentada en español del stream; la descripción ironiza que Claude "continúa caminando en círculos" [V]. Muy baja audiencia: el tema casi no tiene cobertura hispanohablante en video.

#### C10. "Claude Fable 5 beats Pokémon FireRed only using vision" (canal oficial Claude) [V]
- **Canal**: Claude (canal oficial; publica los lanzamientos de producto de Anthropic) [V listado]. **Fecha**: 2026-06-09. **Tipo**: timelapse, 0:53. **Vistas**: 233,138. **URL**: https://www.youtube.com/watch?v=Ty_50J84fMY
- **Afirmación** [V, descripción]: completa FireRed "from start to finish using only raw game screenshots", sin mapas, ayudas de navegación ni estado extra; los modelos anteriores necesitaban "a complex helper harness".
- **Lo que falta**: el video no publica pasos, horas, costo ni número de intentos; un timelapse de 53 s es **la presentación más selectiva posible** (cherry-picking, V12). La réplica independiente de Qingze Hu (ya en `ESTADO_DEL_ARTE.md`) mostró memorización y un control fallido en Crystal (V8).

#### C11. "An Unreasonable Amount of Data on How Well Claude Plays Pokémon Red" (Finn Mertens) [V descripción]
- **Fecha**: 2026-11-15. **Tipo**: video de datos, 6:31. **Vistas**: 13. **URL**: https://www.youtube.com/watch?v=8t8u13X9C-8
- **Hallazgos** (resumen del autor) [V]:
  - Muchas corridas por modelo: se agrupan por tamaño de modelo pese a "a lot of variability between runs".
  - Más esfuerzo de razonamiento ayuda: Opus en *high* supera a Opus en *medium*.
  - Fable en *high* gana donde Fable en *medium* se queda en un **soft-lock**.
  - Sonnet en *high* logra 6 medallas, mientras que en *medium* "gave up on winning" y se dedicó a capturar Pokémon.
- **Valor**: es la única fuente encontrada que mide **variabilidad entre corridas** de un mismo modelo en Pokémon. Confirma que cualquier N=1 (C01, C10, G01) puede caer en cualquier punto de una distribución ancha (V12). Detalles del harness no verificados [NV].

---

### 2.3 Gemini Plays Pokémon (pregunta 3)

#### G01. Stream `gemini_plays_pokemon` y la reacción de Google [V parcial]
- **Autor**: Joel Zhang ("JoelZ"), desarrollador independiente. **URL**: https://www.twitch.tv/gemini_plays_pokemon . Prototipo publicado en HN el 2025-03-30; "Gemini 2.5 Pro won Pokémon Blue in 106k moves" en HN el 2025-05-03 [V, API de HN].
- **Google**: Logan Kilpatrick (AI Studio) celebró la 5.ª medalla aclarando que el siguiente modelo tenía 3 "with a different agent harness"; Sundar Pichai bromeó con "API, Artificial Pokémon Intelligence" y luego publicó en X "What a finish! Gemini 2.5 Pro just completed Pokémon Blue!" [V, posts originales leídos; ver X06]. La mención en el keynote de Google I/O 2025 aparece en prensa [S: PhoneArena, TechCrunch], no se verificó en el video del keynote [NV].
- **Descargo del propio autor**: "please don't consider this a benchmark" (FAQ de Twitch, citado por TechCrunch y Ars) [V].
- **Archivo**: en Crystal se publicaron **savestates de cada paso** de la corrida completa (G07) [V].

#### G02. "Google's Gemini has beaten Pokémon Blue (with a little help)" (Anthony Ha, TechCrunch) [V]
- **Fecha**: 2025-05-03. **URL**: https://techcrunch.com/2025/05/03/googles-gemini-has-beaten-pokemon-blue-with-a-little-help/
- **Puntos clave** [V, texto leído]: Zhang admite "dev interventions" pero dice que no son trampa; la única pista específica fue avisar que hay que hablar dos veces con un Rocket Grunt para obtener la Lift Key (un bug corregido en Yellow); el framework "continues to evolve" durante la corrida.
- **Vicio**: harness cambiado en vivo + intervención humana → la corrida no es reproducible ni comparable (V1, V12).

#### G03. "Why Google Gemini's Pokémon success isn't all it's cracked up to be" (Kyle Orland, Ars Technica) [V]
- **Fecha**: 2025-05-05. **URL**: https://arstechnica.com/ai/2025/05/why-google-geminis-pokemon-success-isnt-all-its-cracked-up-to-be/
- **Argumento** [V]:
  - El harness de Gemini marca qué casillas son transitables, genera un **minimapa textual** y agrega sub-agentes (BFS para laberintos y un solucionador del puzzle de boulders de Victory Road).
  - Zhang defiende el minimapa porque los humanos construyen mapas mentales y los LLM "can't do independently yet".
  - Ars concluye que cuanto más "hand-holding" (información externa, herramientas, harness), menos sirve el juego como prueba de razonamiento general.
  - Recuerda que el RL especializado vence Pokémon de forma eficiente y que un generador aleatorio lo vencería de forma muy ineficiente (afirmación de Ars, sin cifra) [V texto; la parte del generador aleatorio es [NV] y contradice lo dicho en R02].
- **Es la crítica de prensa más citada** sobre "el harness hace el trabajo" (V1).

#### G04. "Gemini Plays Pokémon -- Indigo Plateau & Credits" (HitokageWorshipper) [V]
- **Fecha**: 2025-05-09. **Tipo**: VOD, 2:12:21. **Vistas**: 2,478. **URL**: https://www.youtube.com/watch?v=Htob1nCDSlE
- **Contenido**: la Liga completa con marcas de tiempo por combate (Lorelei 0:12:26, Bruno 0:27:07, Agatha 0:38:28, Lance 0:52:10, Campeón 1:18:13, "Gemini is the Champion" 1:42:09) [V, descripción]. Útil como evidencia primaria del final de la run 1.

#### G05. "Gemini Plays Pokémon Situation: Did Agentic AI Get Better Or Is Something Else Going On?" (Elodine) [V metadatos]
- **Fecha**: 2025-06-01. **Tipo**: video de análisis, 13:02. **Vistas**: 6,678. **URL**: https://www.youtube.com/watch?v=zV6g9752xpk
- **Tesis** [V, descripción]: el éxito de Gemini depende de "the human behind the harness": arquitectura, prompts, bucles de retroalimentación y restricciones. Es la versión en video de la crítica de Ars (V1).

#### G06. "The Making of Gemini Plays Pokemon - Joel Zhang" (PokeAgent) [V]
- **Fecha**: 2025-10-26 (charla dada el 2025-09-13 en T01). **Tipo**: charla, 15:20. **Vistas**: 943. **URL**: https://www.youtube.com/watch?v=eG73wySkgac . Versión escrita: https://blog.jcz.dev/the-making-of-gemini-plays-pokemon (enviada a HN dos veces en agosto de 2025, con 1 comentario cada una) [V].
- **Temas** [V, descripción]: traducir el estado del juego a lenguaje natural, prompting para razonamiento estratégico e infraestructura del scaffolding.
- **Vicio**: la charla misma confirma que la representación textual del estado (y no la visión) es la base del sistema (V2, V16).

#### G07. "Gemini Plays Pokemon Crystal - Full Game Timelapse" (MrCheeze) [V]
- **Fecha**: 2026-01-26. **Tipo**: timelapse, 16:01. **Vistas**: 1,235. **URL**: https://www.youtube.com/watch?v=IU0nnR6dd64
- **Datos** [V, descripción]: Gemini 3 Pro con el harness de "continuous thinking" de jcz.dev; aclara que no es la primera vez que un LLM (ni Gemini) completa Crystal, pero sí **la primera corrida LLM completa con savestates de cada paso publicados** (`waylaidwanderer/gemini-plays-pokemon-public`, rama `crystal_continuous_thinking`; repo con 38 estrellas, actualizado el 2026-11-27) [V].
- **Lección positiva**: publicar el estado de cada paso permite auditar y reproducir cualquier tramo, lo opuesto al timelapse de C10 (contra V12).

---

### 2.4 GPT, otros LLM y comparaciones (pregunta 3 y 4)

> El stream de Clad3815 ("GPT Plays Pokémon") y la cifra de 6,470 pasos de GPT-5 ya están en `notas/llm_agents.md` §1.3 como [S]. Esta pasada leyó los posts originales en X de OpenAI Devs y de Clad3815 (P06), lo que **eleva a [V]** el resultado de Crystal (9,517 pasos) y agrega la corrida de GPT-6 Astra en FireRed. La cifra de 6,470 pasos en Red sigue en [S]: no se encontró el post original, y el post de Reddit "GPT 5 completed Pokémon Red in just 6,470 steps" (r/OpenAI), enviado a HN el 2025-08-16 con 1 comentario, no se pudo leer [NV].

#### P01. "nunu.ai plays Pokémon Emerald! // Gym 1" [V metadatos]
- **Canal**: nunu ai (empresa de testing de juegos con IA). **Fecha**: 2024-07-20. **Tipo**: stream, 1:56:06. **Vistas**: 1,431. **URL**: https://www.youtube.com/watch?v=MgHj3ZEHrR4
- **Qué es**: un agente LLM jugando Emerald hacia el primer gimnasio; el chat puede dirigirse a la IA empezando sus mensajes con "!" [V, descripción]. Es **siete meses anterior** a Claude Plays Pokémon: los streams "LLM juega Pokémon" no empezaron con Anthropic.
- **Vicio**: interacción del chat con el agente = posible intervención humana en línea (V12) [NV en qué medida].

#### P02. "Can ChatGPT Beat Pokémon?" (PaPaSea) [V metadatos]
- **Fecha**: 2024-05-25. **Tipo**: video, 28:17. **Vistas**: 134,219. **URL**: https://www.youtube.com/watch?v=nXhdBvPUTF0
- **Enfoque**: ChatGPT como **guía** de un jugador humano en Red/Blue [V, descripción]. No es un agente: el humano percibe y ejecuta.

#### P03. "I Let ChatGPT Play Pokémon Fire Red!" (Poepeo) [V metadatos]
- **Fecha**: 2025-12-23. **Tipo**: video, 22:16. **Vistas**: 1,286,118. **URL**: https://www.youtube.com/watch?v=6G0QN1EDA-U
- **Enfoque** [V, descripción]: ChatGPT decide combates, capturas, movimientos e ítems; los humanos "simply report what happens on screen" y ejecutan; regla de una sola derrota total (tipo Nuzlocke).
- **Vicio**: el humano hace la percepción y la ejecución (el "harness" es una persona), así que el resultado mide planificación textual, no juego autónomo (V1, V16). Es de los videos más vistos del tema y difunde esa confusión.

#### P04. "GPT 6 Astra Plays Pokémon Red without RAM Access" (Devon7925) [V]
- **Fecha**: 2026-09-08. **Tipo**: video, 1:34:04. **Vistas**: 750. **URL**: https://www.youtube.com/watch?v=CBXEGH2vtuU . Harness: https://github.com/Devon7925/ShoggothPokemon (descrito como "vibe coded").
- **Afirmaciones** [V, descripción]:
  - Llega a la medalla 1 con un harness de juego **general** que no accede a la RAM.
  - Critica que la mayoría de harnesses "read coordinates or arbitrate how long buttons should be held".
  - Usa la ROM original sin color para no dar ventajas.
  - El modelo solo tiene un comando para leer la pantalla y un archivo para fijar las entradas; corre en un sandbox sin internet usando el plan de suscripción (~10 % del límite semanal).
- **Valor**: es la crítica de creador más precisa al **V2 (RAM privilegiada)** y al control de botones por el harness. N=1 y solo hasta Brock.

#### P05. "Can ChatGPT play Pokémon Crystal? (with GPT-4V)" (piman's dumping ground) [V metadatos]
- **Canal**: canal secundario de pimanrules (autor de H05). **Fecha**: 2023-11-19. **Tipo**: video, 1:39:06. **Vistas**: 2,548. **URL**: https://www.youtube.com/watch?v=Dct7dffObpY . El sitio de PokeRL lo embebe en su portada [V].
- **Qué es** [V, descripción]: un primer intento con el modelo de visión de GPT-4 recién lanzado; el código es un gist "very hastily written" que advierte que, si falla la llamada a Discord, el bucle sigue gastando créditos de API sin avanzar el turno.
- **Valor**: junto con un post de X del 2023-11-08 (Sid Radcliffe, citado en el hilo X01: `gpt-4-vision-preview` en Blue, con solo 100 llamadas de API al día y problemas para ubicar al jugador) [V], es de los intentos LLM con visión más antiguos encontrados, unos 15 meses antes de Claude Plays Pokémon. Muestra que la idea existía y que el cuello de botella era la visión del modelo, no la infraestructura (V16).

#### P06. GPT Plays Pokémon: posts de OpenAI Devs y Clad3815 [V]
- **Canal**: https://www.twitch.tv/gpt_plays_pokemon (organizado por Clad3815, miembro de la comunidad; enlazado por Gigazine) [V enlace].
- **Posts verificados en X** (texto leído vía el endpoint público de embeds) [V]:
  - OpenAI Devs, 2025-05-27: "Watch o3 play Pokémon—live" y agradece a Clad3815 por armar el stream.
  - OpenAI Devs, 2025-08-07: "GPT-5 is now playing Pokémon", con un sitio acompañante que muestra prompts, razonamiento, herramientas y autocrítica en tiempo real; el post citado de Clad3815 menciona **apodos elegidos por el chat**.
  - Clad3815, 2025-08-25: GPT-5 vence a Red en Crystal en **9,517 pasos frente a 27,040 de o3**; 16 medallas en 9,205 frente a 22,334.
  - Clad3815, 2026-09-03: tiempo hasta Campeón en FireRed, **GPT-6 Astra (high) 18 h 12 min** frente a **GPT-5.6 Sol (max) 96 h 35 min**; GPT-5.5 no había terminado tras 218 h.
- **Lo que agrega la prensa** (mixed-news, 2026-09-21) [V artículo]: la cifra de Astra es **la mejor de cuatro configuraciones** (high 18:12, medium 19:06, low 21:21, max 24:37); Clad3815 describe el setup como "Screenshots only. No RAM, no hints, no walkthrough." y el artículo advierte que no hay harness común, réplica independiente, hardware, costo ni número de intentos publicados.
- **Vicios**: "pasos" sin definición común (V13), mejor-de-N configuraciones (presentación selectiva, V12), interacción del chat (menor). **Lección positiva**: el sitio acompañante con prompts y razonamiento por paso es transparencia real.

#### M01. Andreas Astrup: "Which AI model is ACTUALLY the best at playing Pokémon?" y "GPT-5.6 Sol vs Claude Opus 5: in Pokemon Fire Red" [V]
- **Fechas**: 2026-06-21 (8:17, 973 vistas, https://www.youtube.com/watch?v=tkZSo8450rQ) y 2026-09-05 (4:11, 908 vistas, https://www.youtube.com/watch?v=8UqJNF_5WhQ). Harness: https://github.com/andreasvig/ai-plays-pokemon ("using only screen vision").
- **Argumento** [V, descripción]: los streams de Gemini y Claude estaban "cheating" al leer el estado de la RAM y navegar por coordenadas; su harness usa **solo visión y botones**. Compara Gemini, GPT, Claude, Grok, Qwen y Mimo en una carrera desde el inicio de FireRed **hasta Viridian City**, incluyendo precio frente a desempeño.
- **Límites**: tramo muy corto del juego; N por modelo no declarado [NV]; los resultados concretos están en el video, no en la descripción [NV].

#### M02. PokéBench (canal, streams y leaderboard) [V]
- **Canal**: PokéBench. **Videos**: "An AI Beat Pokemon Red's First Gym in 279 Turns for $3.70" (2026-09-12, 2:25:04, https://www.youtube.com/watch?v=gOiwesocq_8); "New Record: An AI Beat Pokemon Red's First Gym in 244 Turns" (2026-09-29, 56:03, 837 vistas, https://www.youtube.com/watch?v=NvI0by9Lstc); stream "GPT 6.2 Sol plays Pokémon Red — 1000 turns, no hints" (2026-11-24, https://www.youtube.com/watch?v=Xg2mqWovPZ0). Harness MIT: https://github.com/VibeCodyH/pokebench (25 estrellas); leaderboard https://pokebench.tv
- **Reglas** [V, descripción]: una captura por turno, "the game state" y el texto en pantalla; sin pathfinding, sin pistas de mapa, sin aviso al chocar con una pared; hasta 6 acciones por turno; **1,000 turnos** de presupuesto; **10 hitos leídos de la RAM** (nadie juzga a mano); cada corrida publica el log completo (prompt, razonamiento, botones, tokens, costo y commit del harness).
- **Cifras** [V]: Gemini 3.8 Flash vence a Brock en el turno 279 por US$3.70 (2 h 16 min); GPT-6.1 Sol en el turno 244 (54 min); GPT-6 Astra 246; GPT-6 Sol 673.
- **Segunda categoría, "RVB" (Raw Vision + Buttons)** [V, descripción de https://www.youtube.com/watch?v=GItoow_c96w]: solo una captura por turno y hasta 6 de los 8 botones, sin estado del juego; la única memoria son las notas del modelo y sus turnos recientes; el mismo prompt (`rvb4`) para todos. Claude Opus 5.5 venció a Brock en el turno 444 con 2,521 botones, 2 derrotas totales, 1 h 43 min y US$7.62.
- **Récords en los títulos de los streams**: "Gemini 4 Neon at 243" (2026-11-24) y "Gemini 4 Argon at 203" (2026-11-08, https://www.youtube.com/watch?v=wYmTfieQsUU). No se verificó a qué categoría corresponde cada uno [NV]; que convivan dos récords muestra que "el récord" depende de cuánta ayuda se da (V13).
- **Lectura**: es el diseño comunitario **más cercano a un protocolo** (presupuesto fijo, hitos automáticos, logs públicos, "a bad result goes up the same as a good one"). Aun así: entrega estado del juego (V2), cubre solo hasta Brock, y la métrica "turno" con hasta 6 acciones no es comparable con otros "pasos" (V13).

#### M03. "Why the World's Best AI Systems Are Still So Bad at Pokémon" (Tharin Pillay, TIME) [V]
- **Fecha**: 2026-01-13 (enviado a HN el 2026-01-19). **URL**: https://time.com/7345903/ai-chatgpt-claude-gemini-pokemon/
- **Datos** [V, texto leído]:
  - Tres streams en paralelo en Twitch: GPT 5.2, Claude Opus 4.5 y Gemini 3 Pro.
  - Opus 4.5 llevaba "over 500 hours" y unos **170,000 pasos**; pasó **cuatro días** rodeando un gimnasio sin ver que debía cortar un árbol. Cruce propio: 48,854 pasos el 2025-12-09 (X03), ~170,000 el 2026-01-13 (TIME) y 230,000 el 2026-01-29 (Josh Snider, ya en `ESTADO_DEL_ARTE.md`) son coherentes con unos 145-165 pasos por hora si el stream fue continuo (estimación propia). A ese ritmo, 170,000 pasos son más de 1,000 h, así que el "over 500 hours" de TIME sería solo una cota inferior.
  - Zhang describe el harness como un traje de "Iron Man". Según el artículo, el de Gemini traduce la imagen a texto y el de Claude es más mínimo, "meaning its attempt tells us more about the model itself".
  - Cita de Whidden en R01.
  - Antropomorfización explícita: Gemini 3 Pro, al terminar Blue, escribió que volvería a casa para "talk to Mom one last time"; según el artículo, a Zhang le pareció conmovedor ("poignant").
- **Valor**: resume el consenso de fines de 2025 (los modelos saben qué hacer pero fallan en ejecución de largo horizonte) y a la vez reproduce el encuadre antropomórfico.

#### M04. "Jev Plays Pokémon Red (LIVE): an AI decision model plays the whole game" (Frigade) [V]
- **Canal**: Frigade (Christian Mathiesen). **Fecha**: 2026-09-25. **Tipo**: stream de 38:24:04; hoy figura como no reproducible. **Vistas**: 6,662. **URL**: https://www.youtube.com/watch?v=1HMOA3BawXg . Repo: https://github.com/christianmat/jev-pokemon (138 estrellas).
- **Cifras del README** [V]:
  - Gana Red en **37 h 40 min**, con **16,150 decisiones**, ~39.2 M tokens de entrada y ~US$1.65 de costo.
  - **16 derrotas totales del equipo, 14 de ellas en la Elite Four**, y 15 intentos de Elite Four.
- **Lo que el README admite** [V]:
  - La descripción del video dice "No scripts, no cheats". Pero el harness **lee la RAM**, lista las opciones legales con "facts" (tabla de tipos, estimaciones de daño, "2 areas toward the objective") y ejecuta **A\* y menús**.
  - Conoce los **hitos de la historia** (`milestones.ts`) y los verifica con los event flags reales.
  - Ante bucles, etiqueta las opciones ya probadas, **muestrea una alternativa** y, como último recurso, **recarga el último checkpoint**.
  - Solo funciona con la ROM US/EU cuyas direcciones salen de pret.
- **Lectura**: Jev elige entre opciones que el harness ya filtró y anotó, de modo que es un caso de manual de V1, V2, V5 y V6, presentado como "sin scripts". Corrige el [NV] de `ESTADO_DEL_ARTE.md` §2, donde solo se había leído el titular de Tom's Hardware.

---

### 2.5 Charlas y streams del PokéAgent Challenge (pregunta 5)

#### T01. "PokéAgent Challenge | Research Talks & Hackathon Launch" [V]
- **Canal**: PokeAgent. **Fecha**: 2025-09-13 (en vivo). **Tipo**: charlas, 1:25:45. **Vistas**: 720. **URL**: https://www.youtube.com/watch?v=CaXM2hr_n2A
- **Programa** [V, descripción]: Aaron Traylor ("Pokémon as an AI Problem"), Seth Karten ("Large Language Models and Strategic Search"), Jake Grigsby ("Reinforcement Learning with Human Battle Replays", es decir Metamon), Joel Zhang ("The Making of Gemini Plays Pokémon"). Lanza un hackathon de 48 h con US$2k en créditos de GCP; patrocinan Google DeepMind y el AI Journal.
- **Taller en NeurIPS 2025**: según la descripción de las finales (T03), se presentó en San Diego el **2025-12-07** con charlas de Seth Karten, Aaron Traylor, Minmin Chen y otros [V descripción]; no se encontró la grabación del taller en el canal [NV].

#### T02. "PokéChamp: An Expert-level Minimax Language Agent for Competitive Pokémon" [V]
- **Canal**: PokeAgent (Seth Karten). **Fecha**: 2025-11-05. **Tipo**: charla (ICML 2025 Spotlight), 33:09. **Vistas**: 3,819 (el video más visto del canal). **URL**: https://www.youtube.com/watch?v=Mys1Cg2W20w
- **Afirmaciones** [V, descripción]: nivel experto "without any task-specific training"; el LLM propone acciones, modela al oponente y evalúa estados dentro de un minimax; inferencia bayesiana sobre **más de 4 M de replays** para atributos ocultos; herramientas simbólicas para el cálculo de daño.
- **Vicio**: "sin entrenamiento específico" convive con 4 M de replays humanos y una calculadora de daño exacta; el conocimiento del dominio entra por el harness (V1). El Elo es proyectado (ya en `ESTADO_DEL_ARTE.md` §3.2).

#### T03. Streams oficiales del PokéAgent Challenge [V]
- **Canal**: PokeAgent. Fechas 2025-10 a 2026-03. Ejemplos:
  - "[PokeAgent Challenge] AI Speedruns Pokemon Emerald" (2025-10-29, 7:02:37, 660 vistas, https://www.youtube.com/watch?v=YYAIdwJt5wg).
  - Finales Gen 1 OU (2025-11-04, https://www.youtube.com/watch?v=dLtEZBbRyYY) y Gen 9 OU (2025-11-10, 7:37:23, https://www.youtube.com/watch?v=jmUqIzjLHVU), al mejor de 99 combates, US$1,000 al ganador.
  - "Testing new Gemini 3 Pro --VISION ONLY-- + in-context SLAM in Pokemon Emerald" (2025-11-25, 2:40:46, https://www.youtube.com/watch?v=MeLW9V4mYfw).
  - "[PokeAgent Challenge] CodexCLI Plays Pokemon Emerald" (2026-03-16, 2:09:27, https://www.youtube.com/watch?v=Xi_4kFv0uF4).
  - Clasificatorias diarias de Showdown (Day 1 a Finale).
- **Qué revelan** [V, descripciones]: el experimento con Codex CLI conecta el agente por MCP a herramientas de estado, botones y **navegación A\***, y la captura "failed and was restarted mid-run" (hay un salto de tiempo); el stream de Gemini 3 "vision only + in-context SLAM" ataca el reto de percepción que el paper declara abierto.
- **Valor**: los VODs son el registro público de las corridas que el paper resume; con 30 a 700 vistas, casi nadie los audita.

---

### 2.6 Combate: bots de Showdown y debate competitivo (pregunta 4)

#### B01. "I Fought The Most Terrifying AI in Pokemon Showdown History" (blunder) + blog de Jaxcalibur [V]
- **Fecha**: 2026-08-30. **Tipo**: video, 37:26. **Vistas**: 73,344. **URL**: https://www.youtube.com/watch?v=MfCbL0ozNDs . Capítulos: *Intro to the Bot*, *Battling vs the Bot!*, *Final thoughts on AI in Competitive Pokemon*; enlaza el hilo de Smogon sobre Jaxcalibur (#1 del ladder de Gen 9 Random Battles) [V].
- **Lo que agrega el blog del autor** (https://jaxcalibur.github.io/) [V]: estuvo arriba del ladder ~20 % del tiempo entre el 6 y el 27 de agosto, con 68 % de victorias contra otros jugadores top; un humano (MichaelderBeste2) estuvo arriba el 50 % del tiempo. El autor **retiene el código y los pesos** por temor a que se use para "climb the ladder, obtain suspect reqs, or potentially even win tournaments". Esto explica el "sin código ni pesos" de `ESTADO_DEL_ARTE.md` §3.2: la falta de reproducibilidad es una decisión deliberada del autor. En el blog la deja en suspenso para consultar a la comunidad [V]; según un post de Smogon visto solo en resultados de búsqueda, después decidió no liberarlo [S].
- **Vicio**: el éxito es en Random Battles de Showdown, no en el combate del juego (V11); no hay réplica posible (V12).

#### B02. Debate sobre bots en el ladder humano [V metadatos; foro S]
- **Videos**: Jimothy Cool, "Is A.I Killing Competitive Pokemon?" (2026-10-28, 30:49, 137,717 vistas, https://www.youtube.com/watch?v=MmL68qsj8J0), que habla de Showdown singles y del VGC de Pokémon Champions (Reg. M-C) "under fire from an existential threat"; Mr Craw, "AI BOTS ARE TAKING OVER POKEMON SHOWDOWN" (2026-09-01, 20:28, 14,923 vistas, https://www.youtube.com/watch?v=8R0_AKAv4lc), que enlaza el hilo "Bots are now capable of suspect reqs, do we care?"; DriftingTheTrainer, "Are AI Bots Taking Over Competitive Pokemon?" (2026-08-29, 27:51, 8,128 vistas, https://www.youtube.com/watch?v=QI6hajKB6wE) [V].
- **Foro de Smogon** (solo resultados de búsqueda; la página exige un desafío de navegador) [S]: preocupa que pocas cuentas-bot (se menciona 4) puedan inclinar un *suspect test* de un tier menor con ~30 votantes; se propone subir el GXE exigido a 80+ o poner un mínimo de partidas.
- **Vicio nuevo** (no está en V1-V17): **evaluar en el ladder humano** convierte el benchmark en un problema de integridad competitiva; los resultados de ladder además mezclan la fuerza del bot con la del pool de rivales de ese mes.

#### B03. "We Trained an AI to Play Pokémon Doubles (And it's scary)" (Wisp Coach) [V metadatos]
- **Fecha**: 2026-11-15. **Tipo**: video, 6:23. **Vistas**: 15,172. **URL**: https://www.youtube.com/watch?v=vTJieqlKlPQ
- **Afirmaciones** [V, descripción]: la primera versión (junio) "lost to a friend who had never played Pokémon"; cinco meses después estaba en el **top 10 del ladder de Showdown** (dobles Reg. M-C, *open team sheets*); la descripción menciona "the one idea that changed everything" sin detallarla [NV].
- **Vicio**: producto comercial (wispcoach.com) sin método ni datos publicados; el éxito en ladder no se separa del cambio de metajuego (V12, V13).

#### B04. "ChatGPT vs Google Gemini POKEMON BATTLE!!" (ZXMany) [V metadatos]
- **Fecha**: 2026-02-07. **Tipo**: video, 33:00. **Vistas**: 641,367. **URL**: https://www.youtube.com/watch?v=rITC6D-CyDg
- **Formato** [V, descripción]: cada LLM arma un equipo de 6 Pokémon de alquiler entre los 151 originales (sin Mew ni Mewtwo), con reglas de Pokémon Stadium 1 y permiso para buscar estrategias en internet; uno de los dos "starts to break down and hallucinate information".
- **Vicio**: N=1 y un humano que transcribe estado y acciones; las alucinaciones sobre mecánicas son el equivalente de las "versiones mezcladas" de Gemini (V8, V10).

#### B05. "I Created the Ultimate Pokémon AI (and used it to cheat)" (Astroid Videos) [V metadatos]
- **Fecha**: 2026-11-22. **Tipo**: video, 56:00. **Vistas**: 238,507. **URL**: https://www.youtube.com/watch?v=9agaG24SquM . Capítulos: "is he good enough for the tournament?", "so how do we get away with this?" [V].
- **Valor**: método no declarado [NV]; confirma que el público ya ve los bots de combate como **herramienta para hacer trampa** en torneos (el mismo temor de B01/B02).

#### B06. Jev contra Opus 5 en un combate competitivo (post en X) [V]
- **Autor**: Siddharth Arya (@sid19arya0). **Fecha**: 2026-09-17. **URL**: https://twitter.com/sid19arya0/status/2100458351440048258 (enviado a HN con 3 puntos).
- **Afirmación** [V]: en una partida, Jev le ganó a Opus 5; Jev costó US$0.0029 y 37 s de "pensar", Opus 5 US$2.35 y 6 min 29 s; de ahí el "~820x cheaper, 10x faster".
- **Vicio**: **una sola partida** usada para comparar modelos (V12); el costo por combate es relevante, pero la superioridad no está medida.

---

### 2.7 Análisis de la comunidad: hilos, blogs, X, prensa (pregunta 6)

#### X01. "Show HN: LLM plays Pokémon (open sourced)" [V]
- **Autor**: `adenta` (repo `adenta/fire_red_agent`, 165 estrellas). **Fecha**: 2025-02-26. **Tipo**: hilo de HN, 186 puntos, 59 comentarios. **URL**: https://news.ycombinator.com/item?id=43187231
- **Datos y críticas** [V, comentarios leídos]:
  - El autor llegó al Viridian Forest en FireRed y esquivó la mala visión **leyendo la RAM y describiendo la posición de los objetos**.
  - Debate sobre si Claude recibía la RAM: el diagrama del harness decía que esa información se extraía directamente de la RAM. Esto aclara el [NV] de `notas/infra_and_critiques.md` B2.
  - Un usuario sostiene que un sistema mucho más simple y barato que un LLM jugaría mejor; otro responde que el punto es la "G" de AGI; otro lo llama "publicity stunt".
  - Otro cita la razón del creador de Twitch Plays Pokémon para elegir el juego (ver H01): en Pokémon es difícil no progresar aunque se juegue mal.
- **Lectura**: desde el primer día la comunidad separó dos preguntas que la prensa mezcla: "¿puede un LLM general jugar?" frente a "¿es la mejor forma de jugar?".

#### X02. "So how well is Claude playing Pokémon?" (Julian Bradshaw, LessWrong) [V]
- **Fecha**: 2025-03-07. **Tipo**: blog, 175 puntos. **URL**: https://www.lesswrong.com/posts/HyD3khBjnBhvsp8Gb (leído vía GreaterWrong). No estaba en las notas, que solo citan posts posteriores de Bradshaw.
- **Argumentos y cifras** [V]:
  - Veredicto: "Worse than a 6-year-old would, definitely not PhD-level."
  - Cada "acción" del gráfico de Anthropic es un análisis completo; 35,000 acciones equivalen a unas 140 h de "pensamiento", frente a 26 h de un humano promedio para todo el juego (estimación del autor).
  - Run #2 se abandonó a las 35,000 acciones porque no entendía que debía hablar con Bill; Run #3 llevaba 24 h en Mt. Moon y borró sus notas de la cueva tras desmayarse, creyendo que ya la había pasado.
  - Tras Run #2 se agregó un sistema de memoria nuevo (el harness cambia entre intentos).
  - Tesis central: el problema es la **función ejecutiva** (metas, prioridades, atención), no solo la visión.
  - Un amigo del autor estima que con guía humana una vez por hora iría "10x" más lejos.
  - Nota al pie: las supuestas victorias con entradas aleatorias "involved some human direction/cheating".
- **Contracrítica en los comentarios** [V]: Sam Marks (103 puntos) cuestiona la **línea base humana**: muchos niños tampoco encontraban la salida del primer cuarto (hay que pararse en la alfombra y presionar abajo), y FireRed cambió ese sprite por eso.

#### X03. "Insights into Claude Opus 4.5 from Pokémon" (Julian Bradshaw, LessWrong) [V]
- **Fecha**: 2025-12-09. **Tipo**: blog destacado ("Curated"), 223 puntos, 24 comentarios. **URL**: https://www.lesswrong.com/posts/u6Lacc7wx4yYkBQ3r
- **Cifras y observaciones** [V]:
  - Ningún Claude entre 3.7 Sonnet y Sonnet 4.5 progresó más en la historia; solo llegaban antes al mismo punto.
  - Visión mucho mejor, pero con *ceguera por inatención* (ignora lo que no está mirando) y alucina el ascensor que busca cuando está desesperado.
  - En el gimnasio de Erika pasó ~4 días (~8,000 pasos de razonamiento) dando vueltas; volvió más de 13,000 pasos después y encontró el árbol cortable casi de inmediato.
  - Dependencia total de las notas, que el autor compara con una *amnesia anterógrada*.
  - Comparación: Gemini 2.5 Pro en Blue, 106,505 pasos en 813 h; Opus 4.5 iba en 48,854 pasos tras más de 300 h; GPT-5.1 terminó Crystal en 9,454 pasos y 108 h (~3 veces lo de un humano).
- **Sobre el harness** [V]: el de Claude cambió (soporte de Surf, menos prompts a medida, casillas giratorias marcadas como obstáculos), cambios que según el autor facilitan el escondite Rocket. Los harnesses ganadores fueron "relentlessly optimized by human trial and error". Cierre: "raw intelligence is not the only lever".
- **Comentarios útiles** [V]: Shoshannah Tekofsky nota que Claude **no explora** como un humano aburrido y sugiere que un análogo del aburrimiento evitaría loops (argumento a favor de la novedad intrínseca de E2).

#### X04. Hilo "Gemini 3 Pro vs. 2.5 Pro in Pokemon Crystal" [V]
- **Fecha**: 2025-12-16. **Tipo**: hilo de HN, 315 puntos, 92 comentarios (el más votado del tema encontrado). **URL**: https://news.ycombinator.com/item?id=46287848 (sobre https://blog.jcz.dev/gemini-3-pro-vs-25-pro-in-pokemon-crystal, ya en `ESTADO_DEL_ARTE.md`).
- **Críticas** [V, comentarios leídos]:
  - **Contaminación**, en al menos tres comentarios: tras la publicidad, Pokémon podría ser "explicitly part of the training regime" (`zurfer`). MrCheeze responde que las dificultades de Crystal se discutían en Twitch y Discord, que no se scrapean, pero que ese mismo post sí influirá en el próximo modelo.
  - **El prompt de "ignorar el conocimiento interno"**: se piden ablaciones. Un comentarista cita al autor: preguntado directamente, el modelo daba la solución correcta del puzzle de interruptores la mayoría de las veces, aunque en la partida no la ejecutó. Es evidencia de que el conocimiento está y el prompt solo lo enmascara (V8).
  - **N=1**: "a single run from each is not strong evidence" (`kqr`).
  - **Costo**: estimación de un comentarista de ~US$22,560 (1,880 M tokens × US$12/M de salida, supuesto de precio no verificado) [NV].
  - **TPP como vara**: un comentario afirma que Twitch Plays Pokémon tardó 13 días en Crystal frente a los 17 de Gemini 3 Pro [NV].

#### X05. Hilo "Show HN: Jev Plays Pokémon Red" [V]
- **Fecha**: 2026-09-25. **Tipo**: hilo de HN, 282 puntos, 124 comentarios. **URL**: https://news.ycombinator.com/item?id=49845172 (ver M04).
- **Críticas** [V, comentarios leídos]:
  - "more like watching a walk thru play the game" (`testaccount28`).
  - "harness is doing 99% of the work" (`hbrn`), que enumera: lectura de RAM, grilla de colisión, A\*, grafo de regiones, distancia al objetivo, detección de Pokécenters, NPCs e ítems, y opciones ya anotadas.
  - El autor reconoce haber escrito **33 metas de alto nivel** más submetas.
  - Las opciones son "very railroady" (`ViscountPenguin`).
  - Un usuario probó Jev con contexto mínimo y **no llegó ni al laboratorio de Oak**.
  - Otro sugiere usar un modelo al que se le haya borrado el conocimiento de Pokémon.
  - Los espectadores describen loops de entrar y salir por la misma puerta, y a Charizard olvidando Ember por Counter.
  - En la Elite Four predicen que reintentará sin cambiar de estrategia hasta estar sobrenivelado, y el README lo confirma con 15 intentos.
- **Ideas positivas en el hilo**: combinar un LLM lento para metas con un modelo rápido para acciones; es la jerarquía del ganador de PokéAgent (Heatz) y del profesor offline de ARES-U.

#### X06. Posts en X de Anthropic, Google y OpenAI [V]
- Leídos vía el endpoint público de embeds de X:
  - **@AnthropicAI**, 2025-02-25 (8,610 likes): hilo de lanzamiento ("Can Claude play Pokémon?").
  - **@AnthropicAI**, mismo día (1,684 likes): Claude, atascado en una esquina, "typed out a formal request to reset the game". En HN se bromeó con que parecía una inyección de prompt dirigida al operador.
  - **@OfficialLoganK**, 2025-04-18: 5.ª medalla de Gemini frente a 3 del siguiente modelo, "though with a different agent harness".
  - **@sundarpichai**, 2025-04-19: "We are working on API, Artificial Pokémon Intelligence".
  - **@sundarpichai**, 2025-05-03 (6,210 likes): "What a finish! Gemini 2.5 Pro just completed Pokémon Blue!".
  - Para OpenAI ver P06.
- **Lectura**: las cuentas corporativas amplifican corridas N=1 con harnesses distintos. Kilpatrick incluye la salvedad del harness en el mismo post que celebra la ventaja (V1, V13). El post del "reset" es el ejemplo oficial de antropomorfización.

#### X07. Prensa general (las piezas con ficha propia son C06, G02, G03 y M03) [V título y fecha; texto donde se indica]

| Medio (autor) | Título | Fecha | URL | Qué aporta |
|---|---|---|---|---|
| GamesRadar+ | "AI spends 7,000 hours beating Pokemon Red's first gym, but still can't find the second one after 50,000 hours" | 2023-10-23 | https://www.gamesradar.com/ai-spends-7000-hours-beating-pokemon-reds-first-gym-but-still-cant-find-the-second-one-after-50000-hours/ | Cobertura del video de Whidden (R01); la cifra de 7,000 h solo se leyó en el título [NV texto] |
| TechCrunch | "Anthropic's Claude AI is playing Pokémon on Twitch — slowly" | 2025-02-25 | https://techcrunch.com/2025/02/25/anthropics-claude-ai-is-playing-pokemon-on-twitch-slowly/ | Lanzamiento [NV texto] |
| Futurism | "One of the World's Most Advanced AI Agents Is Completely Stuck Trying to Beat a Pokémon Game for Children" | 2025-03-21 | https://futurism.com/advanced-ai-stuck-pokemon | Encuadre "juego para niños" [NV texto] |
| Tom's Guide | "Claude AI has been continously playing Pokémon Red for over a month — it still can't beat it" | 2025-04-01 | https://www.tomsguide.com/ai/claude-ai-has-been-continously-playing-pokemon-red-for-over-a-month-it-still-cant-beat-it | Duración como titular [NV texto] |
| Xataka (Javier Pastor), en español | "En Anthropic se han obsesionado con algo de apariencia trivial pero clave para la IA: que Claude se pase 'Pokémon'" | 2025-04-02 | https://www.xataka.com/robotica-e-ia/ia-claude-lleva-miles-horas-jugando-a-pokemon-se-ha-atascado-monte-selenite | [V texto] Resume a Ars en español; afirma que Claude "tenía que aprender sobre la marcha", confundiendo adaptación en contexto con aprendizaje |
| Engadget | "Claude isn't a great Pokémon player, and that's okay" | 2025-04-09 | https://www.engadget.com/ai/claude-isnt-a-great-pokemon-player-and-thats-okay-151522448.html | [V texto] Hershey tuvo que darle **lectura de memoria** por la mala visión; pudo haber programado una ruta, pero eso solo mediría seguir instrucciones; "I don't know how useful this will be internally as a benchmark." |
| mixed-news | "GPT-6 Astra beat Pokémon FireRed in 18 hours, then lost a Minecraft run to one creeper" | 2026-09-21 | https://mixed-news.com/en/gpt-6-astra-pokemon-firered-18-hours-minecraft-creeper-run/ | [V texto] Ver P06: "None of this is a controlled benchmark"; mejor de 4 configuraciones; sin costo ni número de intentos |
| Tom's Hardware | "Developer trains a small AI on a single RTX 3080 Ti gaming GPU to 'play' Pokémon Red..." | 2026-09-30 | https://www.tomshardware.com/tech-industry/artificial-intelligence/developer-trains-a-small-ai-on-a-single-rtx-3080-ti-gaming-gpu-to-play-pokemon-red-model-discovered-what-each-button-does-by-predicting-what-happens-next | [V texto] Ver R13; menciona al menos dos victorias con Jev |

#### X08. Reddit y foros cerrados [NV contenido]
- **No accesibles en esta pasada**: Reddit rechaza descargas sin sesión (HTTP 403), la herramienta de búsqueda no indexa reddit.com, y Smogon responde con un desafío de navegador. No se intentó evadir esos bloqueos.
- **Hilos conocidos por referencias cruzadas** [V existencia, por la API de HN o por citas]:
  - r/ClaudePlaysPokemon, "Gemini Plays Pokémon Blue – Megathread" (enviado a HN el 2025-04-10).
  - r/OpenAI, "GPT 5 completed Pokémon Red in just 6,470 steps" (HN, 2025-08-16).
  - r/ClaudePlaysPokemon, "Claude won a game of Civilization VI" (HN, 2026-11-10): el subreddit ya cubre "LLM juega X" en general.
  - El usuario `reasonosaur` de r/ClaudePlaysPokemon es la fuente de una figura de X03.
  - Sam Marks (X02) cita un hilo de Reddit sobre niños atascados en el primer cuarto.
  - Smogon: hilos de Jaxcalibur y de *suspect reqs* (B01, B02).
- **Pendiente**: leer los megathreads con acceso autenticado; es probable que contengan la discusión más detallada sobre cambios de harness.

---

### 2.8 Entretenimiento y "humano en el bucle" (pregunta 4)

> Son los videos con más vistas del tema, pero casi ninguno declara método ni cifras. Sirven para entender la percepción pública, no como evidencia.

| ID | Video | Canal | Fecha | Vistas | URL | Qué es / vicio |
|---|---|---|---|---|---|---|
| E01 | "I Forced Ai To Play Pokemon... Its Save Is Terrifying" | Purple Thunder | 2026-07-15 | 736,542 | https://www.youtube.com/watch?v=gGfbQl4aoJ8 | Revisa las partidas abandonadas que dejó la IA [V desc.]; método no declarado [NV] |
| E01 | "I Trapped An Ai In Pokemon... It Was Disturbing" | Purple Thunder | 2026-08-31 | 2,436,356 | https://www.youtube.com/watch?v=ny3E_tiiwgM | Encerró a la IA en el juego hasta que le ganara un combate [V desc.]; el video de IA y Pokémon **más visto de 2026** encontrado, sin método [NV] |
| E01 | "I Trapped An Ai In Pokemon. How Far Can It Get?" | Purple Thunder | 2026-09-28 | 515,800 | https://www.youtube.com/watch?v=711Y--gqUAc | Secuela; método no declarado [NV] |
| E02 | "¿ChatGPT puede ayudarme a pasar pokémon rojo fuego?" (ES) | aalecard | 2024-01-31 | 558,381 | https://www.youtube.com/watch?v=REz9dTfuhJc | ChatGPT elige el equipo; el humano juega [V desc.] |
| E02 | "¿Puede la IA vencer a TODOS los LÍDERES en Pokémon Rojo Fuego?" (ES) | SuperO4LIVE | 2025-08-24 | 289,040 | https://www.youtube.com/watch?v=z-8RH0S3SIY | ChatGPT elige el equipo y cada movimiento, con 5 "vidas"; "muchas de sus decisiones… no tienen ningún sentido" [V desc.] |
| E02 | "DEJÉ que la IA jugara Pokémon Rojo Fuego POR MÍ" (ES) | SuperO4LIVE | 2026-10-17 | 125,724 | https://www.youtube.com/watch?v=g2KmJ4qTpZU | Gemini elige inicial, capturas y movimientos; una derrota termina el reto [V desc.] |
| E02 | "Claude 3.7 Juega Pokémon y es DESESPERANTE" / "GEMINI se pasa POKÉMON AZUL!" (ES) | Paco Rodríguez | 2025-02-25 / 2025-05-04 | 247 / 418 | https://www.youtube.com/watch?v=BDECizVONnQ · https://www.youtube.com/watch?v=wiWGxRwfbyQ | Noticieros breves en español; repiten el encuadre de la prensa [V metadatos] |
| E03 | "Can My AI BEAT Pokémon Fire Red?" | PokeTech | 2026-11-11 | 4,279 | https://www.youtube.com/watch?v=hRA4N36MmJw | Corrida completa de un "Grok Bot"; la descripción adelanta varios intentos y falta de dinero, sin cifras [V desc.; NV cifras] |
| E04 | "AIs Play Pokemon Episode 1 (Starring ChatGPT, Claude, Gemini, and more!)" | AI World Championships | 2026-04-15 | 1,377 | https://www.youtube.com/watch?v=roUAuQ3tqPk | Formato de liga entre modelos; método no declarado [NV] |
| E05 | "Pokemon Crystal Playoff [PART 3] - Opus 5.5 / GPT 6.1 Sol / GPT 6 Astra / Fable 5.1" (4 partes) | Chris Izatt | 2026-10-01 → 10-04 | 152 (parte 3) | https://www.youtube.com/watch?v=DUB3mwvSZ20 | Ver abajo |

- **E05 en detalle** [V, descripción]: cuatro modelos compiten desde partidas nuevas de Crystal, y la meta se amplió de Campeón a Red en Mt. Silver. Las reglas dicen "no walkthrough scripts or human button inputs", pero:
  - **Pistas escritas del "control de carrera"** aparecían en la pantalla de cada jugador.
  - Las recargas estaban prohibidas, hubo **dos infracciones** anunciadas en vivo, y en la parte 4 se concedió **una recarga por jugador**.
  - **Se cambió de modelo a mitad de carrera** en dos asientos (Muse Spark 1.3 por GPT 6 Astra; GLM 5.3 por Fable 5.1).
  - Los "presentadores" eran otros LLM que leían el chat.

  Es un catálogo en vivo de V6, V12 y V13.
- **Lectura**: en E02 y en P02-P03, **el humano percibe y ejecuta**. Se mide la elección textual (equipo y movimiento), no el juego autónomo, y el público lo recibe como "la IA juega" (V1, V16).

---

### 2.9 Líneas base históricas y no-IA (pregunta 4: Twitch Plays Pokémon y TAS)

#### H01. Twitch Plays Pokémon (2014) y sus historias [V]
- **Datos** (Wikipedia, consultada vía API) [V]: lanzado el **2014-02-12** con Pokémon Red. Completado el **2014-03-01** tras **más de 16 días** continuos, con unos **1.16 M de participantes** (récord Guinness: 1,165,140), un pico de 121,000 participantes simultáneos y 55 M de vistas. El creador eligió Pokémon porque "Even when played very poorly it is difficult not to make progress in Pokémon".
- **Videos**: TheSmithPlays, "The Complete History Of 'Twitch Plays Pokemon'" (2014-03-03, 17:38, 1,405,920 vistas, https://www.youtube.com/watch?v=5Ec8zMxPVzk); en español, Darkraimola, "LA HISTORIA COMPLETA DE TWITCH PLAYS POKÉMON ROJO EN 1 VÍDEO" (2025-04-03, 1:08:39, 72,568 vistas, https://www.youtube.com/watch?v=ST8NZni2m0U) [V].
- **Por qué importa**: es la vara implícita de todos los streams de LLM. Engadget afirma que Claude Plays Pokémon se inspiró en parte en TPP (X07), y en HN se compara con Claude y con Gemini (C01, X04). La cita del creador explica por qué **progresar en Pokémon es barato** y por qué el avance en hitos, por sí solo, es una métrica débil (V13).

#### H02. Pi Plays Pokémon Sapphire (WinningSequence) [V]
- **Qué es** [V, descripción]: cada dígito de π (0-9) se asigna a un botón y se ingresa uno por segundo, 24/7, en un servidor de Google Cloud con transmisión en Twitch (creado por armityle29). Se inspira en Twitch Plays Pokémon y en "Fish Plays Pokémon".
- **Datos**: "PI FINALLY BATTLES MAY!" (2024-06-29, 12:23, 114,217 vistas, https://www.youtube.com/watch?v=RPy7PosIotI) viene del stream n.º 471 (2024-05-23). En noviembre de 2026 los streams iban en el **n.º 906** [V títulos]. En HN (2025-03-05) se dijo que llevaba más de 3 años sin medallas [V comentario].
- **Valor**: es la **línea base aleatoria pública**. Junto con los comentarios de R02 (el azar no sale de Pallet Town tras miles de millones de pasos) y la nota de X02 (las victorias "aleatorias" tuvieron ayuda humana), refuta la frase de Ars (G03) de que un generador aleatorio vencería el juego.

#### H03. "Can you beat Pokemon FireRed while blind and deaf?" (MartSnack / Keeyan Ghoreshi) [V]
- **Fecha**: 2022-10-08. **Tipo**: video, 47:35. **Vistas**: 2,239,214. **URL**: https://www.youtube.com/watch?v=6gjsAA_5Agk . Repo: https://github.com/KeeyanGhoreshi/PokemonFireredSingleSequence (203 estrellas).
- **Qué es** [V, README]: una **lista fija de entradas**, escrita casi toda a mano, que en teoría vence más del 99 % de las partidas de FireRed, sin manipular el RNG ni exigir precisión de frame, y que "does not require any information from the game". La lista completa está "technically untested".
- **Valor**: demuestra que **"terminar el juego" no exige percepción ni aprendizaje**: basta una ruta memorizada en lazo abierto. Es la versión extrema del problema de memorización de los LLM (V8) y de los scripts (V5). Por eso la evaluación de ARES-U mide generalización a contenido no visto, no completitud.

#### H04. "Pokemon Red / Blue Tool Assisted Speedrun History Explained" (Abyssoft) [V metadatos]
- **Fecha**: 2021-08-27. **Tipo**: video, 35:49. **Vistas**: 633,979. **URL**: https://www.youtube.com/watch?v=oYvg2yf7Xeg
- **Valor**: según la descripción, cada TAS nuevo cambia drásticamente la ruta. Es el techo "humano + herramientas" y recuerda que el cartucho es explotable con glitches (uno de los desajustes de V11). Un TAS no es un agente: es búsqueda offline con savestates (V6).

#### H05. "Pokémon Red/Blue AI Tournament - Scientifically Ranking the Pokémon Red Trainers" (pimanrules, "Elo World") [V metadatos]
- **Fecha**: 2020-06-15. **Tipo**: video, 20:46. **Vistas**: 933,217. **URL**: https://www.youtube.com/watch?v=8yUPhRJtNJM ; secuela "Elo World REDUX" (https://www.youtube.com/watch?v=247qD1qulSQ, 872,854 vistas). Código y datos: `jsettlem/elo_world_pokemon_red`, construido sobre pret/pokered [V, descripción].
- **Valor**: enfrenta a los entrenadores del juego entre sí **con su IA real**. Es la referencia divulgativa de lo simple que es el rival dentro del juego frente a un humano de Showdown (V11).

#### H06. "Te ENSEÑO como FUNCIONA la IA de Pokémon" (Yellowpop) [V metadatos]
- **Fecha**: 2025-11-19. **Tipo**: video en español, 25:57. **Vistas**: 24,862. **URL**: https://www.youtube.com/watch?v=UgsDtk5S3Qc
- **Valor** [V, descripción]: explica cómo deciden movimientos y cambios los entrenadores NPC en GBA y DS (rutinas y flags). Lo complementa Chuxsss, "How an Ai Became A Top Pokemon Player" (2025-07-11, 6,247 vistas, https://www.youtube.com/watch?v=zgjlfgyBzBU), que lo resume así: "An NPC in Pokémon won't switch out, predict…" [V desc.]. Ambos apoyan la distinción de V11 entre el rival del juego y el de Showdown, que en E1 se aborda con oponentes fijos y [LP] ajuste contra la IA de NPC.

---

## 3. Síntesis

### 3.a Vicios recurrentes que muestran los videos y la comunidad

Conteo = número de **entradas de este catálogo** (IDs de §1) que muestran o discuten el vicio con evidencia propia; una entrada puede contar en varios vicios. No es una muestra aleatoria: mide **cuánto aparece** cada crítica, no cuán grave es.

| # | Vicio (como lo dice la comunidad) | N.º de fuentes | Fuentes | Ejemplo representativo |
|---|---|---|---|---|
| 1 | El harness hace el trabajo | **15** | C01, C06, G02, G03, G05, M01, M03, M04, P04, T02, T03, X01, X03, X05, X07 | "harness is doing 99% of the work" (X05); el minimapa y los sub-agentes de Gemini (G03) |
| 2 | Estado privilegiado (RAM, casillas transitables, texto) | 10 | C06, G03, G06, M01, M02, M04, P04, X01, X05, X07 | Hershey le dio lectura de memoria por la mala visión (X07, Engadget) |
| 3 | Percepción visual débil | 10 | C01, C06, C08, G06, M01, P04, P05, X01, X03, X07 | El gorro rojo confundido con la alfombra (C01); "ceguera por inatención" (X03) |
| 4 | Bucles y atascos de horas o días | 9 | C01, C06, C07, C08, C09, M03, X02, X03, X05 | 78 h en Mt. Moon (C06); ~4 días en el gimnasio de Erika (X03, M03) |
| 5 | Métricas incomparables y "ganar" mal definido | 8 | G01, M02, M03, X02, X03, P06, H01, E05 | "Acción" = un análisis completo (X02); "turno" con hasta 6 botones (M02) |
| 6 | Costo y eficiencia de muestras | 8 | R02, R07, R08, M01, M02, M04, X02, X04 | 76 M de movimientos para Brock (R08); ~US$22,560 estimados en Crystal (X04) |
| 7 | Intervención humana o humano en el bucle | 7 | G02, P01, P02, P03, E02, E05, X02 | Pista de la Lift Key (G02); pistas del "control de carrera" (E05) |
| 8 | N=1, variabilidad, sin réplica | 6 | C06, C10, C11, X04, B06, P06 | "a single run from each is not strong evidence" (X04); variabilidad entre corridas (C11) |
| 9 | Memorización / contaminación / sin generalización | 6 | C01, C06, C10, M03, X04, X05 | El modelo conoce la solución del puzzle pero el prompt la enmascara (X04) |
| 10 | Antropomorfización y sobreinterpretación | 6 | C05, C07, C08, M03, X06, X07 | "formal request to reset the game" (X06); "aprender sobre la marcha" (Xataka, X07) |
| 11 | Cambios de harness o de modelo a mitad de corrida | 5 | G02, C08, X02, X03, E05 | Intento VIII de Claude etiquetado Opus 4.6 al inicio y 4.7 al final (C08); cambios de asiento en E05 |
| 12 | Rutas, metas o ayudas escritas a mano | 5 | R02, M04, X05, H03, R13 | 33 metas de Jev (X05); lista fija que vence FireRed sin mirar (H03) |
| 13 | Memoria frágil y notas envenenadas | 5 | C06, C08, X02, X03, X07 | "it sort of trusts pretty blindly" (C06); notas de Mt. Moon borradas (X02) |
| 14 | Savestates, checkpoints, recargas | 4 | R03, R13, M04, E05 | Jev recarga el último checkpoint (M04); recargas concedidas en E05 |
| 15 | Recompensa que contrabandea conocimiento del juego | 4 | R01, R02, R07, R08 | "smuggles in" conocimiento externo (R02) |
| 16 | Decisiones erráticas en combate | 4 | R02, M03, B04, X05 | "spamming options + picking damaging attacks" (R02); Charizard olvida Ember (X05) |
| 17 | Reward hacking y trampas de exploración | 3 | R01, R02, R08 | "PC Trauma" (R01); "The AI cheats" (R08) |
| 18 | Bots en ladders humanos (integridad competitiva) | 3 | B01, B02, B05 | Jaxcalibur retiene los pesos (B01) |
| 19 | Presentación selectiva | 2 | C10, P06 | Timelapse de 53 s (C10); mejor de 4 configuraciones (P06) |
| 20 | Línea base humana mal calibrada | 2 | X02, R02 | Niños atascados en el primer cuarto (X02) |
| 21 | Rival del juego ≠ rival de Showdown | 2 | H05, H06 | La IA de NPC no cambia ni predice (H06) |
| 22 | Representaciones que memorizan IDs | **0** | — | Nadie en la comunidad lo discute |

**Tres observaciones.**
1. El vicio n.º 1 domina por amplio margen y aparece en todas las familias de fuentes (prensa, HN, LessWrong, creadores y charlas). Es el consenso más sólido del tema.
2. Los vicios 10 (antropomorfización) y 18 (bots en ladders humanos) **no tienen id en V1-V17**, y el 11 (cambios a mitad de corrida) y el 20 (línea base humana) solo encajan en parte dentro de V12 y V13. Salen de mirar streams y foros completos, no de leer papers.
3. El vicio de IDs (V15) tiene **cero** menciones comunitarias. La hipótesis central de E1 ataca un problema que nadie fuera del paper está mirando, lo que refuerza su novedad pero obliga a motivarlo con evidencia propia (la ablación E1-e).

### 3.b Lecciones positivas que la comunidad encontró útiles

| Lección | Fuentes | Cómo se traduce en ARES-U |
|---|---|---|
| Publicar el log por turno, los savestates y el commit del harness | M02, G07, P06, M04, R13 | Logs fechados de E1/E2 y configuración completa publicada (V12) |
| Mismo harness, mismo presupuesto y mismos hitos automáticos para todos | M02, M01, E05 (parcial) | Presupuesto fijo de pasos y oponentes fijos (V13) |
| Varias corridas y líneas base explícitas (aleatorio y sin entrenar) | C11, R08, R13, R02 | 5 semillas en E1 y 3 en E2; Random y Greedy como líneas base (V12) |
| La RAM solo para evaluar (hitos leídos de memoria, nadie juzga a mano) | M02, M04 | "RAM solo para medir" en E2 (V2) |
| Separar categorías por nivel de ayuda (solo visión frente a estado) | M02 (RVB), P04, M01 | Tabla de componentes del harness on/off [LP] (V1, V5) |
| Recompensa sin conocimiento del juego (novedad) | R06, R13 | RND sobre el latente del AE en E2 (V3, V4) |
| Declarar ayudas y límites (README honesto, descargos) | G01, M04, X05, C06, X07 | Sección "Sin resolver" de cada vicio; tabla de scripts on/off (V5) |
| Jerarquía: LLM lento para metas y política rápida para acciones | X05, T01 (Heatz en `ESTADO_DEL_ARTE.md`) | LLM solo como profesor offline destilado [LP] (V1, V17) |
| Un análogo del aburrimiento o la curiosidad evita loops | X03 (Tekofsky) | Novedad intrínseca (RND) en E2 (V9) |
| Archivo comunitario de VODs y chat | C08, G04 | Guardar las trayectorias de evaluación como artefacto [LP] |
| Prudencia al liberar bots fuertes | B01 | E1 se evalúa localmente contra bots fijos, no en ladders públicos |

### 3.c Cómo se mapea cada vicio a las respuestas de ARES-U (`VICIOS_Y_SOLUCIONES.md`)

| Vicio comunitario (3.a) | Id(s) en `VICIOS_Y_SOLUCIONES.md` | Respuesta de ARES-U ya definida | Qué agrega esta revisión |
|---|---|---|---|
| 1. El harness hace el trabajo | V1 | Bridge crudo en E2; la red decide sola en E1; heurísticas solo como líneas base; [LP] tabla de componentes | X05 (Jev) y M02 (PokéBench) muestran que ni las corridas "sin scripts" cumplen: la tabla debe incluir **filtrado y anotación de opciones**, no solo mapas y pathfinders |
| 2. Estado privilegiado | V2 | Solo píxeles en E2; features públicas en E1; RAM solo para medir | M02 entrega "the game state" incluso en su categoría estándar; declarar explícitamente qué entra a la política |
| 3. Percepción débil | V16 | AE auto-supervisado; encoder transferido frente a desde cero | P05 y X03: la visión mejoró por generación de modelo, pero la *ceguera por inatención* persiste; E2 debería reportar errores de percepción, no solo celdas |
| 4. Bucles y atascos | V9 | RND; métricas de revisitas y entropía de botones | X03 (aburrimiento) apoya la novedad intrínseca como mecanismo, no solo como métrica |
| 5. Métricas incomparables | V13 | Oponentes fijos, presupuesto fijo, contabilidad de cómputo | Definir "paso" = una decisión de la política con K frames, y **nunca** reportar "pasos" de terceros sin su definición |
| 6. Costo | V14 | Modelos pequeños; DirectML | R08 (76 M de movimientos) y R07 (14 meses de juego) dan órdenes de magnitud para justificar el alcance reducido de E2 |
| 7. Intervención humana | V12 (intervención humana) | Configuración fija, sin intervención | Agregar al reporte una línea explícita de "intervenciones humanas: ninguna" con el commit |
| 8. N=1 y variabilidad | V12 | 5 y 3 semillas, IC 95 %, Holm | C11 y R13 confirman que la variabilidad entre corridas es grande; sostiene el protocolo |
| 9. Memorización / contaminación | V7, V8 | Entrenamiento desde cero; split por generación; zero-shot Gen2/Gen3; FR → EM | X04 muestra que la instrucción de *ignorar el conocimiento interno* no lo borra; H03 muestra que una ruta memorizada basta para terminar. Por eso el criterio es **generalización**, no completitud |
| 10. Antropomorfización | — (sin id) | No definida | Propuesta: regla de redacción en el informe ("la política", "el paso", "la tasa de victoria"; no "aprende sobre la marcha", "decide retirarse") |
| 11. Cambios a mitad de corrida | V12, V13 | Configuración publicada | Propuesta: congelar modelo, harness y prompt por corrida e identificarlos con hash o commit, como PokéBench (M02) |
| 12. Rutas, metas y ayudas a mano | V5 (y A1 para ARES-PKMN) | Cero ayudas; tabla de scripts on/off; orquestador apagado | H03 y X05 justifican que los hitos escritos a mano se usen **solo para evaluar** |
| 13. Memoria frágil / notas envenenadas | V17 | [LP] memoria recurrente; LLM solo offline | Sin cambios; no se demuestra en la ventana del curso |
| 14. Savestates, checkpoints, recargas | V6 | Archivo declarado; evaluación continua por semilla | E05 y M04 muestran recargas de emergencia: prohibirlas en la evaluación y contarlas si ocurren en exploración |
| 15. Recompensa que contrabandea conocimiento | V3 | ±1 en E1; RND en E2; sin constantes de juego | R02 (HN) es la cita comunitaria más directa a favor de esta decisión |
| 16. Decisiones erráticas en combate | V10 | Self-play con liga; tasa de panic switching | R02 y X05 agregan el spam de opciones y el olvido del único movimiento útil: medir también la tasa de acciones dominadas |
| 17. Reward hacking | V4 | Recompensa terminal; RND sobre el latente; [LP] E3B | Sin cambios |
| 18. Bots en ladders humanos | — (sin id; cercano a V11) | No definida | Propuesta: declarar que E1 no se despliega en ladders públicos (ética y validez) |
| 19. Presentación selectiva | V12 | Media ± desviación estándar entre semillas | Prohibir "mejor de N configuraciones" salvo que se reporten todas |
| 20. Línea base humana mal calibrada | V13 | No hay línea base humana en E1/E2 | Mantenerla fuera del alcance; si se usa, citar su dispersión (X02) |
| 21. Rival del juego ≠ Showdown | V11 | Daño exacto de Gen 3; [LP] ajuste contra la IA de NPC | H05 y H06 ofrecen material divulgativo para explicar el desajuste en el informe |
| 22. IDs memorizados | V15 | Tokens de entidad sin IDs; ablación E1-e | Brecha **no vista** por la comunidad: es el aporte más original de E1 |

**Brechas.** Los vicios 10, 18 y (en parte) 11 no tienen respuesta en `VICIOS_Y_SOLUCIONES.md`. Las propuestas de esta tabla son sugerencias para ese documento; no se editó.

---

## 4. Limitaciones de este catálogo y pendientes

1. **Sin transcripciones**: el contenido de los videos se infiere de su descripción, sus capítulos, los repos enlazados y las fuentes que los citan. El contenido del capítulo "RNG manipulation" (R01) o la idea clave de Wisp Coach (B03) quedan [NV].
2. **Reddit, Smogon y X**: Reddit y Smogon no fueron accesibles. De X solo se leyeron posts con ID conocido, vía el endpoint público de embeds. Falta el post original de GPT-5 en Red (6,470 pasos).
3. **Vistas y fechas**: las vistas son del 2026-11-27. Las fechas son las de publicación en YouTube (hora del Pacífico); un estreno o un VOD puede diferir del día del stream.
4. **Sesgo de idioma**: la cobertura en español es escasa y mayormente de entretenimiento (R05, R11, R12, C09, E02, H01, H06, Xataka). No se encontró un análisis técnico en español comparable a X02 o X03.
5. **Pendientes concretos**:
   - Confirmar el cambio de Opus 4.6 a 4.7 dentro del intento VIII (C08).
   - Identificar a qué categoría de PokéBench corresponde cada récord: los títulos citan 243 (Gemini 4 Neon) y 203 (Gemini 4 Argon) [NV].
   - Ver el método de Purple Thunder (E01) y de Wisp Coach (B03).
   - Verificar la mención de Gemini en el keynote de Google I/O 2025 (G01).
   - Leer el megathread de Gemini en r/ClaudePlaysPokemon.
