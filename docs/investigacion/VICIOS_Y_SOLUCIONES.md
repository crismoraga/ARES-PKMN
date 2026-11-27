# Vicios de los agentes de Pokémon y cómo los aborda ARES-U

> **Entregable central** de la investigación IPD434. Es un catálogo de los vicios, limitaciones y errores que se repiten en los agentes que juegan Pokémon, tanto publicados como propios (ARES-PKMN). Para cada vicio se indica la evidencia, la severidad, el mecanismo con que ARES-U lo aborda, el experimento o la métrica que lo verifica, y lo que **queda sin resolver**.
>
> **Fuentes**: `notas/*.md`, `ESTADO_DEL_ARTE.md`, `ESTADO_REPOSITORIO.md`, `PLAN_IPD434.md`. Las marcas son las mismas que en el estado del arte: [V] verificado en fuente primaria, [S] solo fuente secundaria, [NV] no verificado.
>
> **Severidad**:
> - **Crítica**: invalida la afirmación de "juega Pokémon" o de "generaliza".
> - **Alta**: sesga los resultados de forma importante.
> - **Media**: degrada la eficiencia o la interpretabilidad.
> - **Baja**: es un detalle de diseño.
>
> **Alcance honesto.** En la ventana del curso, ARES-U **verifica** dos cosas: **E1** (transferencia del combate entre generaciones, sin IDs) y **E2** (exploración desde píxeles con transferencia FireRed → Emerald). El resto de las soluciones son **arquitectura a largo plazo** y se marcan como **[LP]**: son diseño, no evidencia.

---

## Parte I — Vicios del estado del arte

### V1. El harness hace el trabajo
- **Descripción.** El "agente" recibe del harness estado ya procesado, mapas, pathfinders (A\*, BFS), solucionadores de puzzles, sub-agentes, críticos y resúmenes. El éxito no puede atribuirse al modelo.
- **Evidencia.**
  - Gemini 2.5 Pro en Blue usaba RAM convertida a texto, un mapa *fog-of-war* con tipos de tile leídos de RAM, `pathfinder` y `boulder_puzzle_strategist`. Tardó **813 h** (run 1) y 406.5 h (run 2) [V, arXiv 2507.06261].
  - El baseline de PokéAgent incluye `navigate_to` con A\* [V].
  - Bradshaw muestra que comparar Gemini con Claude no prueba nada porque los harnesses y los juegos son distintos [V].
  - PokéAgent: "By conflating harness with model capability…" (paráfrasis) [V, arXiv 2603.15563].
  - Sin harness, los VLM de frontera completan **~0 %** [V].
- **Severidad: Crítica.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: el entorno de E2 es el bridge **en modo crudo, sin orquestador**. El agente solo controla los botones; el harness no navega, no resuelve puzzles y no elige objetivos. En E1 el agente elige directamente la acción legal (4 movimientos o cambio), sin estimador de daño en la decisión: el daño analítico existe solo como **línea base** (Greedy).
  - *Verificación*: en E2 se comparan las celdas únicas visitadas por la política aprendida frente a Random **con el mismo presupuesto de pasos**. En E1 se mide el win rate de la red frente a Greedy y frente a `MLPGenome.prior`, que son exactamente "el harness/heurística haciendo el trabajo".
  - *[LP]*: en el reporte de cada corrida se publicará la **tabla de componentes del harness** (dimensiones de PokéAgent: estado, mapa, navegación, memoria, críticos e intervención humana), marcando cada uno como on u off.
- **Sin resolver.** E2 es exploración, no progreso en la historia. Que un agente sin harness complete una campaña sigue abierto para todo el campo, incluido ARES-U.

### V2. Información privilegiada de RAM
- **Descripción.** La observación incluye flags de eventos, party, inventario o tipos de tile leídos de RAM por direcciones específicas de un juego. Esto funciona como un oráculo y además impide portar el agente.
- **Evidencia.**
  - pokemonred_puffer embebe party (especie, tipos, movimientos), **2560 flags de eventos**, map ID e items [V].
  - Pleines usa flags de eventos y lo reconoce como limitación [V].
  - Gemini lee los tipos de tile desde RAM [V].
  - **ARES-PKMN observa solo RAM** a través de 16 símbolos pret por juego y unos 40 símbolos internos (`ffi_mgba.py:54-64`; `field_controls.py:135-842`) [V repo].
- **Severidad: Alta.** Es crítica para la transferencia.
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: en E2 la observación es **solo píxeles** (84×84 en gris, con apilado de frames). La RAM se usa **solo para medir** (mapa, x, y) y nunca entra a la política. En E1 la observación del combate son **features públicas y semánticas** (tipos, stats base, fracción de HP, estado, boosts), que equivalen a lo que un humano ve o puede inferir.
  - *Verificación*: los tests de E2 lo comprueban de forma automática:
    - `test_ram_is_not_in_observation`: la RAM no aparece en la observación.
    - `test_observation_only_depends_on_framebuffer`: la observación es función solo del framebuffer.
    - `test_learner_is_invariant_to_ram`: el learner da el mismo resultado si se altera la RAM.
    - `test_episodic_novelty_is_pixel_only`: la novedad episódica usa solo píxeles.

    *[LP]*: una **ablación con y sin adaptador simbólico** (feature dropout) para cuantificar "cuánto aporta la RAM".
- **Sin resolver.** En el combate, E1 asume que existe un extractor de entidades. Desde píxeles habría que reconocer especie y HP en pantalla, cosa que no se demuestra esta semana.

### V3. Recompensas hechas a mano y específicas del juego
- **Descripción.** La recompensa tiene decenas de términos ligados a eventos, mapas y objetos de un juego concreto, así que no se puede portar y "pone el juego sobre rieles".
- **Evidencia.**
  - PokeRL/puffer tiene **más de 25 parámetros**, unos 80 *required events* con nombres de Red y un boost por map ID, que los autores llaman su "least favorite" y describen como "on rails" [V].
  - Whidden V2: `event*4 + heal*10 + badge*10 + explore` [V].
  - PokeRL 2026 usa recompensas micro, meso y macro hechas a mano [V].
  - **ARES-PKMN**: la recompensa del canal profundo del Stadium es `ΔHP + 2·resultado` (`deep.py:100-105`), y el GFlowNet usa una recompensa heurística de cobertura (`gflownet_team.py:100-108`) [V repo].
- **Severidad: Crítica.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: en E1 la recompensa es **solo el resultado del combate** (±1), igual para cualquier generación. En E2 la recompensa es **intrínseca (RND)**, sin recompensa extrínseca específica del mapa.
  - *[LP]*: recompensa extrínseca a partir de **eventos genéricos multi-generación** (medalla, mapa nuevo, subir de nivel, captura), definidos de la misma forma por un adaptador para cada juego.
  - *Verificación*: el código de recompensa de E1 y E2 no contiene constantes de ningún juego (se revisa en el repositorio). En E2, el mismo código de recompensa corre en FireRed y en Emerald.
- **Sin resolver.** La novedad sola no lleva a completar la historia (ver Pleines: sin R_nav se logran 0 hitos y con 10×R_nav no se vence a Brock). A largo plazo todavía habrá que definir hitos "genéricos" por juego.

### V4. Reward hacking y trampas de exploración
- **Descripción.** El agente optimiza el proxy en lugar de la meta: farmea curaciones o niveles, juega en las tragamonedas, hace spam de A en los menús o queda hipnotizado por animaciones (*noisy TV*). También puede volverse averso tras un solo evento negativo.
- **Evidencia.**
  - Pleines: Bulbasaur con Leech Seed contra Zubats acumula **399 curaciones frente a 22.97 de un humano** y llega 0 veces a Cerulean [V].
  - puffer: tragamonedas, spam de A y farmeo en Lavender [V].
  - Whidden V1: el KNN de frames hace que el agente mire el agua y el pasto animados; tras un depósito accidental en el PC, la suma de niveles baja de 25 a 12 y "the AI will avoid it entirely" [V video].
  - Hamburg: con penalizaciones aparece "only safe behavior" [V].
  - PokeRL 2026: el agente aprendió a guardar la partida, así que se eliminó Start [V].
- **Severidad: Alta.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: en E1, con recompensa terminal ±1 no hay nada que farmear. En E2, RND calcula la novedad sobre **features del encoder AE**, no sobre píxeles crudos, para atenuar el *noisy TV*. Se usan solo recompensas positivas o de novedad, sin penalizaciones duras.
  - *[LP]*: **E3B**, cuyo embedding IDM captura solo lo controlable e ignora animaciones, y bonus con tope y decaimiento.
  - *Verificación*: en E1 se miden la longitud de los combates y la tasa de acciones inútiles (movimiento contra un inmune). En E2 se mide la fracción de pasos en celdas ya vistas y la entropía de la distribución de botones; un agente hipnotizado tendría mucha novedad y pocas celdas únicas.
- **Sin resolver.** Con RND el *noisy TV* solo se mitiga, no se elimina. E2 no mide si la exploración es útil para la historia.

### V5. Segmentos scripted, dinero infinito y HMs automáticos
- **Descripción.** El entorno resuelve las partes difíciles: usa HMs automáticamente, da dinero infinito, resuelve puzzles de Strength, regala Pokémon o empieza después del parcel.
- **Evidencia.**
  - El `config.yaml` de puffer tiene `infinite_money: True`, `auto_use_strength/surf`, `auto_solve_strength_puzzles`, `auto_flash`, `auto_pokeflute` y `auto_remove_all_nonuseful_items` [V]. **No hay corrida estable con todos los scripts apagados** [V].
  - Whidden y Pleines empiezan después del parcel de Oak [V].
  - Heatz, el ganador de PokéAgent, genera políticas scripted con un LLM [V].
- **Severidad: Crítica.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: E1 y E2 no tienen ayudas de juego. E2 parte de un **savestate fijo y declarado** (justo después de la intro) y no hay más ayudas.
  - *[LP]*: un principio de la Game Abstraction Layer: **el harness no ejecuta acciones de juego**, y cualquier ayuda se declara en una **tabla de scripts on/off** obligatoria en cada reporte.
  - *Verificación*: la configuración de E1 y E2 se publica completa en el informe.
- **Sin resolver.** Los HMs, los puzzles y los diálogos obligatorios no entran en el alcance de E1 ni de E2.

### V6. Swarming de savestates como currículum oculto
- **Descripción.** Cuando un env alcanza un hito, todos los envs cargan su estado. La "partida" deja de ser continua y la métrica pasa a ser "el primer env que llegó".
- **Evidencia.**
  - puffer: al cumplir un objetivo se guarda un savestate y **todos los envs lo cargan**; está inspirado en Go-Explore [V].
  - PokeGym reporta hitos como la primera vez que ocurren en algún env, por ejemplo la medalla 3 a 404M pasos [V].
  - PokeRL 2026 usa un currículum de savestates por secuencia [V].
  - **ARES-PKMN**: usa rewind a checkpoint hasta 6 veces (`orchestrator.py:183`, `:2802`) y reinicios del supervisor desde un checkpoint [V repo].
- **Severidad: Alta.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: el archivo de estados **se declara explícitamente** como un algoritmo de exploración (Go-Explore, que es computación evolutiva o QD). Se separa la **fase de exploración**, que puede restaurar estados, de la **evaluación**, que es una corrida continua desde un estado inicial fijo y no restaura nada.
  - *Verificación*: E2 evalúa episodios desde el savestate inicial, sin restauraciones intermedias, y reporta **por semilla** (3 semillas), no como "mejor env". *[LP]*: el reporte separa "progreso con archivo" de "progreso en partida continua".
- **Sin resolver.** Go-Explore no se implementa en la ventana del curso; es [LP].

### V7. Cero generalización entre juegos
- **Descripción.** Un agente por juego y por ROM. Nadie evalúa en un juego no visto.
- **Evidencia.**
  - Todos los RL de `ESTADO_DEL_ARTE.md` §1 se evalúan solo en Red o solo en Emerald [V].
  - Hamburg **cose modelos por segmento** [V].
  - Réplica de Fable 5: el control en **Crystal queda atascado 763 turnos** en Route 29 [V post].
  - Metamon: la observación queda ligada al formato y falló en Gen9 sin Team Preview [V].
  - **ARES-PKMN**: unas 90 ramas por juego y `Literal["firered","emerald"]` [V repo].
- **Severidad: Crítica**, porque es la meta misma del proyecto.
- **Cómo lo aborda ARES-U.**
  - *Mecanismo, E1*: se entrena **solo con Gen 1** (#1-151) y se evalúa **zero-shot** con especies de **Gen 2 (#152-251) y Gen 3 (#252-386)** nunca vistas. *E2*: se transfiere el encoder y la política de FireRed a Emerald.
  - *Verificación*:
    - E1: la **brecha de generalización**, calculada como puntaje en Gen1 (equipos nuevos de la distribución de entrenamiento) menos puntaje en Gen2/Gen3 contra los mismos oponentes. Se usan 5 semillas, IC 95% bootstrap sobre pares de equipos y un test de permutación (`ares_u/battle/paired.py`). Se compara con la ablación con IDs (H2).
    - E2: celdas únicas en Emerald, desde cero frente a transferido.
    - *[LP]*: protocolo **leave-one-game-out** entre Gen 1-4.
- **Sin resolver.**
  - E1 mantiene la mecánica de Gen 3 (el motor del Stadium), así que mide transferencia a **contenido nuevo** (especies, stats y movimientos), no a **reglas nuevas** de otra generación.
  - E2 transfiere entre dos juegos del **mismo motor**, algo análogo a la crítica que se le hace a Lumine.
  - La transferencia real entre plataformas (GB → GBA → NDS) queda en [LP].

### V8. Memorización y contaminación
- **Descripción.** Los LLM conocen los walkthroughs y "generalizan" porque recuerdan. Las wikis de otra versión empeoran el desempeño.
- **Evidencia.**
  - La réplica de Fable 5 anotó "Oak's Parcel" **141 turnos antes** de recibirlo, y un prompt que prohibía usar memoria no lo evitó [V post].
  - Gemini alucina TEA de FRLG mientras juega Blue [V].
  - PokéAgent: el retrieval de wikis a veces empeora el desempeño, y la contaminación solo se discute con argumentos [V].
  - lmgame-Bench reconoce la contaminación como problema [V].
- **Severidad: Alta** para los LLM. Es baja para modelos entrenados desde cero.
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: los modelos de E1 y E2 se entrenan **desde cero**, sin preentrenamiento web, así que no hay walkthroughs que memorizar. El split de E1 por generación **impide por construcción** que el modelo haya visto las especies del test.
  - *[LP]*: el LLM solo actúa como **profesor offline**, y la evaluación se hace en juegos o ROM hacks retenidos.
  - *Verificación*: en E1, la ablación **con IDs de especie** (línea base e) muestra el efecto de la memorización frente a la representación semántica.
- **Sin resolver.** Las features de E1 (stats base y tipos) vienen del dex y son "conocimiento de dominio" legítimo, pero conocimiento al fin.

### V9. Loops, menu spam y deambular improductivo
- **Descripción.** El agente repite ciclos de estado, pulsa A o abre menús sin sentido, y se queda atascado durante horas o miles de turnos.
- **Evidencia.**
  - PokeRL 2026: **41.2 %** de episodios en loop y A en el 32.1 % de las acciones [V].
  - Gemini en la Power Plant: el mismo payload **842 veces en 1,003 turnos** [V].
  - Claude 3.7: **52 h en Mt. Moon** [V].
  - Gemini 2.5 Pro en Crystal: 16,403 turnos en Olivine [V].
  - **ARES-PKMN**: 301 `repeated_state_cycle` y 720 `stall_detected` en Emerald, y 2,367 `campaign_blocked` en FireRed [V repo].
- **Severidad: Alta.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: la novedad intrínseca (RND) reduce el valor de estados repetidos. *[LP]*: memoria episódica (conteo de celdas por mapa), un núcleo recurrente (xLSTM o SSM) y un archivo Go-Explore para escapar con *return-then-explore*.
  - *Verificación*: en E2 se mide la tasa de revisitas y la entropía de botones. En E1 se miden los turnos por combate (un combate que se alarga sin fin delata un loop).
- **Sin resolver.** En E2, el agente entrenado con ~10^5–10^6 pasos no alcanzará los cuellos de botella reales (cuevas, puzzles).

### V10. *Panic switching* y *agent panic*
- **Descripción.** Ante un rival fuerte o con HP bajo, el agente cambia de Pokémon en cadena o toma decisiones erráticas.
- **Evidencia.**
  - PokéLLMon documenta el *panic switching* y cae a **18.75 %** frente a estrategias de desgaste (n=16) [V].
  - Gemini 2.5 tiene episodios de "agent panic" [V].
  - Fable 5 revive Pokémon débiles que caen al turno siguiente [S].
- **Severidad: Media.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: PPO con self-play y una **liga de oponentes** (instantáneas pasadas, Greedy y Random) penaliza implícitamente los cambios inútiles.
  - *Verificación*: la **tasa de panic switching** (cambios consecutivos) es una métrica anti-vicio declarada en E1 (`PLAN_IPD434.md` E1), junto con la tasa de acciones inútiles, y se compara contra todas las líneas base.
- **Sin resolver.** E1 no tiene ítems de mochila, que en el juego real cambian la lógica de curar o cambiar.

### V11. Desajuste entre Showdown y el combate del juego
- **Descripción.** Optimizar para Showdown (OU o VGC) no es lo mismo que combatir dentro del juego. Cambian el rival, la información disponible, los recursos que persisten, las reglas, la interfaz de menús y el formato.
- **Evidencia.** `notas/benchmarks_battle.md`, análisis transversal (6 puntos). PokéAgent separa los tracks por esta razón [V]. Metamon modela la IA de los NPC (`Gen1BossAI`, `EmeraldKaizo`) [V].
- **Severidad: Alta** para cualquier agente de campaña.
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: el Stadium usa **daño entero exacto de Gen 3** con datos exportados de la ROM. En E1 los equipos son 3v3 a nivel 50, sin cláusulas de Smogon.
  - *[LP]*: entrenar el Transformer de entidades con datos **de varias fuentes** (Showdown multi-gen, Stadium y combates in-game registrados) y ajustarlo **contra la IA de NPC** del juego, con un objetivo de costo (HP y PP gastados), no solo de victoria.
  - *Verificación [LP]*: el win rate y el HP residual en combates reales del emulador.
- **Sin resolver.** El Stadium no se ha validado combate a combate contra el emulador (`ESTADO_REPOSITORIO.md` §3). Hay que declararlo como "simulador aproximado".

### V12. N=1 y corridas no reproducibles
- **Descripción.** Una sola corrida, harnesses que cambian en vivo, modelos cerrados que cambian de versión e intervención humana.
- **Evidencia.**
  - Casi todas las corridas de LLM son N=1 [V].
  - Gemini run 1 tuvo intervenciones [V].
  - El harness de Claude cambió entre versiones [V].
  - Jaxcalibur no publica código ni pesos [V].
  - Pleines selecciona el "best data point" [V].
  - **ARES-PKMN**: entrenamiento con **1 semilla (42)** en ~80 s (`notas/repo_neural_stadium.md` §1.4), y un test que depende del estado de `artifacts/` (`tests/test_navigation_recovery.py:84`) [V repo].
- **Severidad: Alta.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: el protocolo exige **5 semillas por método** en E1 y 3 en E2, el **mismo presupuesto de pasos** y 500 combates por split y oponente, en pares de equipos con asientos intercambiados. Como todos los métodos juegan los mismos combates, el análisis principal es **pareado a nivel de combate**: permutación por cambio de signo e IC bootstrap sobre pares. Una familia de 4 hipótesis **pre-registradas** en el código (commit `53d61ec`) se corrige con Holm. La variabilidad entre semillas se reporta aparte (d.e. e IC $t$), junto con el tiempo de pared y el hardware. Se publican el código y las configuraciones.

    El diseño inicial (Mann-Whitney sobre 5 semillas con Holm sobre 12 comparaciones) se descartó: con 5 contra 5 el p mínimo alcanzable es 0.0079, y Holm lo eleva a ≥ 0.095, así que **nunca** podía rechazar. Ese error de diseño lo detectó la revisión adversarial.
  - *Verificación*: la tabla de resultados del informe muestra la media ± desviación estándar entre semillas y los IC.
- **Sin resolver.** Cinco semillas siguen siendo pocas para efectos pequeños; el paper GAME usa 20 réplicas.

### V13. Métricas que no se pueden comparar
- **Descripción.** Cada trabajo mide distinto: pasos, acciones, turnos, horas o tokens; botones múltiples por paso; emulador pausado; Elo "proyectado"; GXE local frente a GXE del ladder.
- **Evidencia.**
  - PokéAgent: "steps/actions/turns" no tienen definición común, y pregunta "Was Gemini 3 Pro's 173-hour completion better than Claude Opus 4.6 reaching Victory Road?" [V].
  - Metamon: el GXE local "no tiene conexión" con el del ladder [V].
  - El Elo de PokéChamp es proyectado [V].
- **Severidad: Media.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: en E1, el win rate se mide contra **oponentes fijos y públicos** (Random, Greedy y el prior), con lados alternados. En E2, las celdas únicas se miden con un **presupuesto fijo de pasos de agente** y un número fijo de frames por acción (K).
  - *[LP]*: reportar siempre **pasos de agente, frames emulados, tiempo de pared, GPU-horas y costo**.
- **Sin resolver.** No hay forma de compararse directamente con las cifras de LLM en campañas completas.

### V14. Cómputo extremo
- **Descripción.** Los resultados requieren entre 10^8 y 10^10 pasos, varias GPU de gama alta o miles de dólares en tokens.
- **Evidencia.**
  - puffer: `total_timesteps 10_000_000_000` ("100B for full games") en 4× RTX 4090 [V].
  - Whidden: más de 50,000 h simuladas [S].
  - Pleines: ~36 h por run, y la variante GRU en CPU ~24 días [V].
  - Gemini 3 Crystal: 1,880 millones de tokens [V].
  - Jaxcalibur: ~100M partidas en 1 H100 [V].
- **Severidad: Media** para la ciencia; **alta** para la reproducibilidad.
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: modelos pequeños (un Transformer de entidades y un AE con CNN pequeña) y entornos rápidos (el Stadium hace unos 130-250 combates/s por proceso con políticas ligeras). El learner usa **DirectML en la RX 6700 XT (~6× la CPU)** y los rollouts usan 16 hilos de CPU. Se aprovecha el hallazgo de que DirectML sí funciona (ver `ESTADO_REPOSITORIO.md` §1.1).
  - *Verificación*: el informe reporta el tiempo de pared y el hardware por método (ya exigido en el protocolo).
- **Sin resolver.** Una campaña completa sin scripts probablemente necesita órdenes de magnitud más cómputo que el disponible.

### V15. Representaciones que memorizan IDs
- **Descripción.** Los embeddings por ID de especie o movimiento (o el one-hot de 411 especies) no transfieren a especies nuevas ni a otra generación.
- **Evidencia.**
  - puffer usa embeddings de especie, map ID e items [V].
  - **ARES-PKMN**: el GFlowNet trabaja sobre 411 especies y 355 movimientos, y las observaciones se normalizan por 411 (`gflownet_team.py:33-45`; `observations.py:52,95`) [V repo].
  - Las ablaciones de este tipo no aparecen en la literatura revisada; no se encontró evidencia publicada de que alguien lo haya medido [NV].
- **Severidad: Alta** para la generalización.
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: los **tokens de entidad de E1 no tienen IDs**. Una especie se describe por sus tipos, stats base y estado, y un movimiento por tipo, potencia, precisión, categoría, PP, efectividad y STAB. Las acciones ilegales se enmascaran.
  - *Verificación*: la **ablación (e)**, un Transformer **con IDs de especie**, debería mostrar una brecha Gen1 → Gen2/Gen3 mayor que la del modelo sin IDs. Esa es la hipótesis central de E1.
- **Sin resolver.** Las habilidades y los efectos secundarios raros quedan resumidos de forma burda, y los movimientos que el motor no soporta se cuentan como `unsupported`.

### V16. Percepción visual débil y falta de lectura de texto
- **Descripción.** Los VLM confunden puertas, árboles cortables o alfombras. Los RL no leen diálogos y sustituyen la comprensión por recompensas de map ID.
- **Evidencia.**
  - Claude 3.7 pulsó A durante 12 h sobre una alfombra [V].
  - Opus 4.7 leyó una puerta como "Conveyor Belt" [V].
  - En Gemini, quitar la visión casi no cambia el desempeño [V].
  - PokéAgent: la visión es el principal modo de falla y "VLM-SLAM" es un reto abierto [V].
  - Whidden se queda en el parcel de Oak porque no puede leer el diálogo [V].
- **Severidad: Alta.**
- **Cómo lo aborda ARES-U.**
  - *Mecanismo*: el encoder visual se **preentrena de forma auto-supervisada** (AE en E2) y se evalúa por su desempeño en una tarea posterior, como pide la clase 05-3.
  - *[LP]*: un encoder compartido entre juegos más un IDM con acciones latentes (truco VPT/LAPO) y un mapa topológico aprendido.
  - *Verificación*: en E2, encoder transferido frente a encoder desde cero en Emerald.
- **Sin resolver.** La lectura de texto y diálogos no se aborda. A largo plazo sería trabajo del profesor LLM destilado.

### V17. Envenenamiento de contexto y memoria frágil (LLM)
- **Descripción.** Notas o resúmenes con creencias falsas que se refuerzan solas; no se ponen a prueba las hipótesis; se olvida lo planificado a largo plazo.
- **Evidencia.**
  - Gemini: *context poisoning* y "black-out strategy" [V].
  - Research notes: las notas refuerzan errores [V].
  - Opus 4.5 ignoró Gold Teeth, y por eso no consiguió Strength [V].
  - Por encima de ~100k tokens, Gemini repite acciones del historial [V].
- **Severidad: Media** (específica de los LLM).
- **Cómo lo aborda ARES-U.** *[LP]*: la política online no es un LLM. La memoria es un **estado recurrente aprendido** (xLSTM o SSM) más una memoria episódica estructurada (archivo de celdas e hitos), que no se puede "convencer" con texto falso. El LLM solo genera subgoals offline, y esos subgoals se validan con `success_cond(state)` al estilo de Heatz.
- **Sin resolver.** En la ventana del curso no se demuestra.

---

## Parte II — Vicios propios de ARES-PKMN

### A1. Campaña escrita a mano
- **Descripción.** El "qué hacer" viene de **101 milestones** (45 en FR y 56 en EM) con mapa, regex de script, flags, nivel y HMs; 27 de ellos tienen `puzzle=`. A eso se suman unas 90 ramas por juego y parches manuales por mapa.
- **Evidencia.** `htn_planner.py:133-246`; `docs/RECUPERACION_AUTONOMA.md`, que es un catálogo de bloqueos corregidos a mano; 17 derrotas contra el rival de Celeste resueltas grindeando (`lessons.json`); máximo de unas 3 medallas por juego (`ESTADO_REPOSITORIO.md` §5).
- **Severidad: Crítica.** Es el vicio V5 llevado al extremo.
- **Cómo lo aborda ARES-U.** No se extiende la campaña. E1 y E2 **no usan el HTN ni las FSMs**. *[LP]*: una política jerárquica que aprende subgoals, con un profesor LLM offline destilado; los hitos se usan **solo para evaluar**.
- **Verificación.** E1 y E2 corren con el orquestador apagado.
- **Sin resolver.** No existe todavía un reemplazo aprendido que complete campañas.

### A2. Redes neuronales decorativas
- **Descripción.** Los módulos neuronales existen y están bien implementados, pero no deciden casi nada.
- **Evidencia.**
  - Combate: `score = efe - daño*4 - logits*0.05` (`orchestrator.py:1001`).
  - H-JEPA: solo elige caminar o correr, y queda sobrescrito (`:1440-1444`).
  - GFlowNet: solo sesga capturas (`:1043,1120`).
  - `proved_decisions = 0` [V repo].
- **Severidad: Crítica** para cualquier afirmación de que el agente "aprende".
- **Cómo lo aborda ARES-U.** En E1 **la red decide sola** la acción, sin mezclarse con la heurística. Las heurísticas son **líneas base separadas**. En E2 la política de píxeles controla todos los botones.
- **Verificación.** Comparación directa contra Greedy y el prior (líneas base b y c de E1). Si la red no las supera, se informa tal cual.
- **Sin resolver.** —

### A3. Entrenamiento solo con datos sintéticos o de juguete
- **Descripción.** El JEPA se entrena con mapas sintéticos de 12×14 de un `SyntheticGBASiliconBridge`. El PPO, en una arena 1v1 **sin tipos, sin cambios y sin estados**, donde la política óptima es trivial (win rate 1.0).
- **Evidencia.** `core/training_data.py:16-59`; `psro_league_trainer.py:68-172`; `notas/repo_neural_stadium.md` §1.4 [V repo].
- **Severidad: Alta.**
- **Cómo lo aborda ARES-U.** E1 entrena en el **motor del Stadium** (daño exacto de Gen 3, 412 especies, learnsets reales). E2 entrena con **frames reales** de FireRed y Emerald en el emulador.
- **Verificación.** Los resultados se reportan sobre esos entornos, con líneas base fuertes.
- **Sin resolver.** El Stadium sigue siendo un simulador (V11).

### A4. Los tests usados como evidencia
- **Descripción.** El criterio de "terminado" fue que pytest, mypy y una demo dieran exit 0. Eso es verificación de software, no evidencia de aprendizaje.
- **Evidencia.** El prompt fundacional `gemini-code-1791938796398.xml` (`notas/course_and_local_docs.md` D3); **451 passed, 1 skipped**; mypy con 1 error. El curso advierte que un modelo sofisticado sin pregunta clara, línea base ni métricas no es evidencia (`00_Introduccion.md:211-213`) [V].
- **Severidad: Alta** (es metodológica).
- **Cómo lo aborda ARES-U.** Los tests se tratan como **higiene**. La evidencia es **experimental**: pregunta clara, líneas base, semillas e IC, como pide el curso (`04_RedesNeuronales.md:1784-1808`; `05_EA_Avanzado.md:322-336`).
- **Sin resolver.** —

### A5. Acoplamiento a Gen 3 y pret
- **Descripción.** Todo el pipeline asume el decomp pret con `.sym`, scripts `.inc`, JSON de mapas y structs de Gen 3. Un juego sin decomp es inobservable.
- **Evidencia.** `ESTADO_REPOSITORIO.md` §6; `ffi_mgba.py:130-187` (unos 30 métodos que habría que reimplementar por plataforma).
- **Severidad: Crítica** para la meta.
- **Cómo lo aborda ARES-U.** *[LP]*: Game Abstraction Layer pensada primero para píxeles. Los adaptadores simbólicos desde pret son **opcionales** y se usan solo para recompensa y evaluación. El espacio de acción de botones es universal.
- **Verificación.** E2 ya usa el bridge solo como "emulador + botones + píxeles".
- **Sin resolver.** No se han implementado backends PyBoy ni NDS.

### A6. Estado operativo engañoso
- **Descripción.** `status.json` dice `running: true` aunque no hay ningún proceso vivo desde el 17-nov. Además, la sesión actual empezó desde cero, lo que confunde los conteos con el máximo histórico.
- **Evidencia.** `notas/repo_health.md` §4 [V repo].
- **Severidad: Baja.**
- **Cómo lo aborda ARES-U.** Los resultados del informe vienen **solo** de corridas nuevas de E1 y E2, con logs fechados, y no del estado de las campañas.
- **Sin resolver.** Corregir el apagado limpio de las campañas queda fuera del alcance.

---

## Parte III — Tabla resumen

| # | Vicio | Severidad | Respuesta de ARES-U | Se verifica en |
|---|---|---|---|---|
| V1 | El harness hace el trabajo | Crítica | Entorno crudo sin orquestador; heurísticas solo como líneas base | E1 (contra Greedy y prior), E2 (contra Random) |
| V2 | RAM privilegiada | Alta | Píxeles en E2; features públicas en E1; RAM solo para medir | E2; [LP] ablación del adaptador simbólico |
| V3 | Recompensa a mano por juego | Crítica | ±1 de resultado (E1); RND (E2); [LP] eventos genéricos | Código de recompensa sin constantes de juego |
| V4 | Reward hacking y noisy TV | Alta | Recompensa terminal; RND sobre el latente AE; [LP] E3B | Acciones inútiles, revisitas, entropía |
| V5 | Scripts, dinero infinito, auto-HM | Crítica | Cero ayudas; tabla de scripts on/off | Configuración publicada |
| V6 | Swarming como currículum oculto | Alta | Archivo declarado; evaluación continua por semilla | E2 por semilla; [LP] Go-Explore |
| V7 | Cero generalización entre juegos | Crítica | Entrenar con Gen1 y evaluar zero-shot en Gen2/Gen3; FR → EM | **Brecha de generalización (E1)**, celdas en EM (E2) |
| V8 | Memorización y contaminación | Alta | Entrenamiento desde cero; split por generación | Ablación con IDs (E1-e) |
| V9 | Loops y menu spam | Alta | Novedad intrínseca; [LP] memoria y archivo | Revisitas (E2), turnos por combate (E1) |
| V10 | Panic switching | Media | Self-play con liga | Tasa de panic switching (E1) |
| V11 | Showdown frente al juego | Alta | Motor con daño exacto de Gen 3; [LP] ajuste contra la IA de NPC | [LP] combates en el emulador |
| V12 | N=1, no reproducible | Alta | 5 y 3 semillas, IC, Holm, código público | Tablas del informe |
| V13 | Métricas incomparables | Media | Oponentes fijos; presupuesto fijo; contabilidad de cómputo | Protocolo E1/E2 |
| V14 | Cómputo extremo | Media | Modelos pequeños; DirectML (~6× la CPU) | Tiempo de pared y hardware |
| V15 | Representaciones que memorizan IDs | Alta | Tokens de entidad sin IDs | **Ablación (e) de E1** |
| V16 | Percepción débil, sin texto | Alta | AE auto-supervisado; [LP] IDM y acciones latentes | E2 (encoder transferido frente a desde cero) |
| V17 | Envenenamiento de contexto | Media | [LP] memoria recurrente y LLM solo offline | — |
| A1 | Campaña escrita a mano | Crítica | Orquestador apagado; [LP] subgoals aprendidos | E1/E2 sin HTN |
| A2 | Redes decorativas | Crítica | La red decide sola | E1 frente a las líneas base |
| A3 | Entrenamiento sintético o de juguete | Alta | Stadium real y frames reales | E1/E2 |
| A4 | Tests como evidencia | Alta | Protocolo experimental | Informe |
| A5 | Acoplamiento a Gen 3 y pret | Crítica | [LP] GAL basada en píxeles | — |
| A6 | Estado operativo engañoso | Baja | Solo corridas nuevas | Logs de E1/E2 |

---

## Parte IV — Arquitectura objetivo ARES-U (visión a largo plazo)

> Esta parte es **diseño**, no resultado. E1 y E2 validan dos piezas: el Transformer de entidades sin IDs y el encoder de píxeles con exploración intrínseca. El resto son hipótesis de trabajo apoyadas en el estado del arte.

```
┌──────────────────────────── Evaluación (protocolo fijo) ───────────────────────────┐
│ leave-one-game-out · tabla scripts on/off · ≥5 semillas · pasos/frames/wall/GPU-h  │
└────────────────────────────────────────────────────────────────────────────────────┘
          ▲ métricas (RAM solo aquí)
┌─────────────────────────────── Game Abstraction Layer ─────────────────────────────┐
│ Backends: PyBoy (GB/GBC) · mGBA/shim ARES (GBA) · melonDS/py-desmume (NDS)          │
│           (stable-retro como API común opcional)                                   │
│ obs = píxeles (resize común) [+ simbólico OPCIONAL vía adaptador pret .sym, c/máscara]│
│ acción = {A,B,Start,Select,↑,↓,←,→,NOOP} × K frames  [+ touch macro NDS]            │
│ events() genéricos: medalla, mapa nuevo, nivel, captura, desmayo (reward/eval)      │
└────────────────────────────────────────────────────────────────────────────────────┘
          ▼ píxeles
┌──────── Encoder visual compartido (auto-supervisado: AE/VAE → JEPA/VQ; CNN residual) ┐
└────────────────────────────────────────────────────────────────────────────────────┘
          ▼ z_t                                   ▼ (en combate) extractor de entidades
┌──── Política jerárquica con memoria ────┐   ┌── Transformer de combate (tokens entidad, ┐
│ alto nivel: subgoal (cada N pasos)      │   │   sin IDs) entrenado cross-gen:          │
│ bajo nivel: botones                     │◄──┤   Showdown multi-gen + Stadium + in-game │
│ núcleo recurrente xLSTM / SSM (Mamba)   │   │   PPO self-play + liga; ajuste vs IA NPC │
│ memoria episódica (celdas, hitos)       │   └──────────────────────────────────────────┘
└─────────────────────────────────────────┘
          ▲ bonus intrínseco RND / E3B (IDM)      ▲ archivo Go-Explore (QD, save states)
          ▲ demostraciones destiladas  ◄── LLM profesor OFFLINE (subgoals + success_cond)
```

### Componentes, justificación y unidad del curso

| # | Componente | Qué resuelve | Base en la literatura | Unidad IPD434 | Estado |
|---|---|---|---|---|---|
| 1 | **Game Abstraction Layer**: píxeles primero; adaptadores simbólicos opcionales desde símbolos pret; acción universal por botones; backends PyBoy, mGBA y melonDS | V2, V5, A5; habilita V7 | `notas/infra_and_critiques.md` A7; PyBoy (×1014 tiempo real), stable-retro (GB/GBA/NDS); NitroGen (acción de gamepad unificada) | 05-1 (formulación MDP, API Gymnasium `reset/step`) | E2 usa el backend mGBA en modo crudo; el resto es [LP] |
| 2 | **Encoder visual auto-supervisado compartido** (AE/VAE en E2; JEPA o VQ a futuro), con CNN residual | V16, V4 (latente robusto al *noisy TV*), V7 | Campese & Moschitti (JEPA congelado de 2 a 8 veces mejor); V-JEPA 2; LAPO/Genie | **05-3** (AE, denoising, VAE, preentrenamiento, evaluación por tarea posterior), **05-4** (bloque residual, transfer learning: congelar la base y entrenar la cabeza; robustez bajo cambio de dominio) | **E2** |
| 3 | **Transformer de combate con tokens de entidad** (sin IDs), entrenado entre generaciones (Showdown multi-gen y Stadium) | V7, V8, V11, V15, A2 | Metamon (secuencias, offline RL), Jaxcalibur (Transformer + PPO), VGC-Bench (PSRO) | **05-2** (atención, Q/K/V, multi-head, máscaras de relleno y de acciones ilegales), 05-1 (RL; DQN como línea base) | **E1** (Stadium con daño Gen 3; Showdown es [LP]) |
| 4 | **Política jerárquica con memoria** (núcleo xLSTM o SSM y memoria episódica) | V9, V17, horizonte de 10^4–10^6 pasos | LRAM (xLSTM), Drama (Mamba); Hamburg (vector de hitos como memoria) | 05-1 (PPO/actor-crítico, explicado aparte), 04 (LSTM/RNN), 05-2 | [LP] |
| 5 | **Exploración intrínseca** (RND en E2; E3B con embedding IDM a futuro) | V3, V4, V9 | RND, NovelD, E3B; PokeRL (anti-loop) | 05-1 | **E2** (RND) |
| 6 | **Archivo Go-Explore** (celdas: mapa, x, y, medallas; *return-then-explore* con savestates; selección por novedad) | V6 (declarado), V9, detachment y derailment | Go-Explore; GAME/QD (Anne et al.) | **03/05 EA** (selección, archivo QD tipo MAP-Elites, criterios de reinicio y estancamiento) | [LP]; el Stadium ya tiene GAME/QD |
| 7 | **LLM solo como profesor offline destilado** (subgoals con `success_cond(state)`, demostraciones que se destilan a la política pequeña; nunca juega en línea) | V1, V8, V17, V14 (costo de inferencia) | **Lección de Heatz** (1º en PokéAgent T2: LLM → scripts → destilación + DQN); Continual Harness con Gemma (SFT + GRPO + DAgger) | 05-2 (Transformers, transfer learning), 04 (supervisado/BC) | [LP] |
| 8 | **Protocolo de evaluación**: leave-one-game-out; tabla de scripts on/off; ≥5 semillas; IC y Holm; contabilidad de pasos de agente, frames, tiempo de pared, GPU-horas y costo | V12, V13, V5, A4 | PokéAgent (dimensiones del harness); GAME (presupuesto igualado, 20 réplicas, Holm) | **04** (diseño experimental: línea base, semillas, presupuesto; reporte de variabilidad y recursos), **05 EA** (mediana, dispersión, tasa de éxito) | E1/E2 (versión reducida) |
| 9 | **Control difuso de la mezcla exploración/explotación** (opcional): reglas difusas que ajustan el coeficiente intrínseco β y el umbral de "atasco" a partir de la tasa de novedad y del progreso reciente, en lugar de los umbrales rígidos del watchdog actual (`progress_watchdog.py:44-62`) | V9, A1 (reemplaza umbrales a mano por reglas interpretables) | Propuesta propia, **sin evidencia en la literatura revisada** | **02 Sistemas difusos** (las notas no pudieron leer el `.md` de 02 en la rama 2026-2: verificar la notación antes de usarlo) | [LP], opcional |

### Mapa de unidades del curso

| Unidad | Dónde aparece en ARES-U | Experimento |
|---|---|---|
| **05-1 Deep RL** (MDP, Bellman, Q-learning, **DQN**, Gymnasium) | Formulación MDP de E1 y E2; **DQN-MLP como línea base (d) de E1**; PPO como método propuesto, que se explica aparte porque no está en el material | E1, E2 |
| **05-2 Transformers** | Transformer de entidades pre-norm escrito a mano, con máscaras de acciones ilegales; [LP] profesor LLM | **E1** |
| **05-3 AE/VAE/GAN** | Encoder AE preentrenado en frames de FireRed, evaluado por la tarea posterior en Emerald | **E2** |
| **05-4 CNN avanzadas** | CNN residual del encoder; transfer learning (congelar y ajustar); robustez bajo cambio de dominio (FR → EM) | **E2** |
| **03 / 05 EA** | Coevolución GAME/QD del Stadium (existente); [LP] archivo Go-Explore; `MLPGenome` neuroevolucionado como referencia | E1 (prior/QD como línea base), [LP] |
| **02 Difusos** | [LP] controlador difuso de β y de detección de atascos | — |

### Qué queda abierto aunque ARES-U funcione

1. **Completar una campaña sin scripts ni harness**: nadie lo ha logrado, y ARES-U no lo promete.
2. **Leer y comprender diálogos** sin un LLM en línea.
3. **Transferir reglas**, no solo contenido: mecánicas distintas entre Gen 1 y Gen 9, la doble pantalla de NDS y el control táctil.
4. **Validar el Stadium contra el emulador** combate a combate.
5. **Cómputo**: una campaña completa por aprendizaje probablemente exige entre 10^9 y 10^10 pasos (orden de magnitud de puffer), lejos del presupuesto disponible.
