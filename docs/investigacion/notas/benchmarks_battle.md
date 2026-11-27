# Benchmarks académicos, competencias y Battle AI (Pokémon): notas de investigación

Fecha de corte: 2026-11-24. Autor: agente de investigación (proyecto ARES-PKMN, IPD434 Soft Computing).
Verificación: abstracts vía arXiv API (export.arxiv.org), HTML de arXiv, READMEs de GitHub (gh api), sitios oficiales.
Convención: [V] = verificado contra fuente primaria en esta sesión; [NV] = no verificado.

---

## 0. Hallazgo: arXiv 2601.19562 NO es un paper de Pokémon [V]
- Título: "Tournament Informed Adversarial Quality Diversity" (Anne, Syrkis, Elhosni, ..., Risi). v1 27-01-2026, v2 15-05-2026, GECCO '26, cs.NE.
- Tema: Quality-Diversity adversarial sobre GAME (Generational Adversarial MAP-Elites). Propone (1) un torneo inter-variante para comparar conjuntos de soluciones con 6 medidas de calidad/diversidad y (2) dos métodos de selección de oponentes informados por torneo.
- Dominios: Pong, cat-and-mouse, pursuers-evaders. Pokémon no aparece.
- Relevancia indirecta para ARES: (a) evaluar por torneo y no contra un único oponente fijo; (b) seleccionar oponentes para que haya diversidad en el self-play/league de batalla, igual que el league training de Porygon2AI en PokéAgent.

---

## 1. PokéAgent Challenge (NeurIPS 2025) + paper arXiv 2603.15563 [V]
- URL: https://arxiv.org/abs/2603.15563 , https://pokeagentchallenge.com . Paper publicado el 16-03-2026 (v2).
- Dos tracks:
  - **Battling** (Showdown): foco en Gen 1 OU y Gen 9 OU; la infraestructura soporta más formatos (Gen 9 VGC aparece en el apéndice). Hay un modo "Extended Timer" para LLMs.
  - **Speedrunning**: Pokémon Emerald en emulador, desde el inicio hasta vencer a Roxanne (1er gimnasio) en 15 hitos. Se rankea por % completado, después por tiempo y el desempate es el número de acciones.
- Datos: más de 20M trayectorias de batalla (más de 4M derivadas de humanos y 18M de self-play), más de 200K equipos y más de 100K batallas de la competencia.
- Métrica principal de batalla: Full-History Bradley–Terry (FH-BT) con bootstrap y mínimo de 250 batallas en la clasificatoria. También reportan Glicko-1, GXE y Elo de Showdown; el apéndice B.4 compara 4 sistemas de rating sobre más de 1.6M partidas.
- Baselines de organizadores: PokéChamp (LLM+minimax), Metamon (30 checkpoints, desde RNN pequeñas hasta Transformers de 200M) y Foul Play (búsqueda). En el texto revisado no aparecen random/max-damage como baselines del leaderboard; sí aparecen como sparring en Metamon y PokéChamp.
- Participación: más de 100 equipos, más de 150 envíos, 35K batallas de clasificatoria en 2 semanas. De los 16 cupos, 13 fueron para extensiones de los baselines RL públicos (Metamon).
- Ganadores Battling:
  - Gen 1 OU: **PA-Agent** ganó 50–28 a 4thLesson (final al mejor de 99). Es offline RL tipo Metamon con Transformer; los replays humanos bajan de 100% a 10% a lo largo de 6 rondas; GXE de 80.35% en la clasificatoria. 4thLesson hizo fine-tune de SyntheticRL-V2 con el optimizador Kron.
  - Gen 9 OU: **FoulPlay** ganó 50–14 a Team Q. Usa MCTS sobre un motor en Rust, agrupa las tiradas de daño e infiere los sets rivales con estadísticas de uso de Smogon; reporta más de 80% GXE. Team Q usó currículum: primero heurísticos, después un coach experto y versiones pasadas de sí mismo.
  - Judge's Choice: Porygon2AI (league training tipo AlphaStar) y August (el mejor LLM puro).
- Ganadores Speedrunning (22 envíos válidos, 6 llegaron al 100%):
  - **Heatz**, 1º con 40:13 (Scripted Policy Distillation: un LLM escribe subgoals y scripts que se destilan en una DQN).
  - **Hamburg PokéRunners**, 2º (recurrent PPO + LSTM con un vector binario de hitos como memoria y goal-conditioning; solo recompensas positivas, porque las negativas volvían al agente demasiado cauteloso).
  - **anthonys**, el mejor harness LLM puro, con 1:29:17 (7 fases con un prompt cada una y A* sobre la grilla).
  - **Deepest**, 5º por tiempo, fue el que usó menos pasos: 649 contra 1,608 de Heatz.
  - Humanos: el mejor speedrunner hizo 18 min y el promedio humano 1:22:05. Heatz es 2.2× más lento que el mejor humano.
  - Sin harness, los VLM frontier logran en la práctica 0% de completitud.
- Otros números: sin harness, Gemini 3 Flash llega a 71% GXE y Gemma3-1B a 29%; con el harness de PokéChamp, Gemma3-1B sube a 53%.
- Análisis BenchPress: la matriz de 83 modelos × 49 benchmarks es casi rango 2, pero ese SVD de rango 2 explica solo 27% de la varianza del GXE en batalla. Ningún benchmark correlaciona fuerte con el GXE (ρ máximo de Spearman 0.77, |ρ| medio 0.45). Esto sugiere que batallar en Pokémon mide algo que los benchmarks estándar de LLM no capturan.
- Limitaciones declaradas: la visión es el principal modo de falla. El retrieval de wikis a veces empeoraba el desempeño por información contradictoria o de otra versión, por eso los baselines no usan búsqueda web. Los resultados miden modelo+harness juntos. "Steps/actions/turns" no tienen una definición común. La contaminación se discute solo argumentativamente, sin ningún test formal.

## 2. Metamon (Grigsby et al., RLC 2025; arXiv 2504.04395) [V]
- URL: https://arxiv.org/abs/2504.04395 , https://github.com/UT-Austin-RPL/metamon (149★, push 07-2026), https://metamon.tech
- Enfoque: reconstruye la perspectiva en primera persona a partir de replays de espectador de Showdown y entrena modelos de secuencia (RNN/Transformer, hasta 200M) con imitation learning, después offline RL y después fine-tuning offline con self-play. No usa búsqueda explícita. Formatos: Gen 1-4 OU (+ Gen 9 OU desde PokéAgent).
- Datasets: metamon-raw-replays con 2.7M batallas; metamon-parsed-replays con 5.3M trayectorias; self-play pac-base (11M), pac-exploratory (7M) y pac-tauros (4M, Gen1OU).
- Resultados (README, GXE en el ladder humano con equipos de muestra):

  | Modelo | Fecha | Tamaño | G1 | G2 | G3 | G4 | G9 |
  |---|---|---|---|---|---|---|---|
  | SyntheticRLV2 | sep 2024 | 200M | 77% | 68% | 64% | 66% | — |
  | Kadabra3 | sep 2025 | 57M | 80% | — | — | — | 64% |
  | Kakuna | dic 2025 | 142M | 82% | 70% | 63% | 64% | 71% |
  | TaurosV0 | may 2026 | 62M | 83% | — | — | — | — |

  TaurosEnsemble llegó al **#1 del ladder humano de Gen1OU**. El abstract del paper reporta top 10% de los jugadores activos.
- Baselines heurísticos (README): `RandomBaseline`, `BugCatcher` (elige el movimiento que menos daño hace), `Gen1BossAI` (imita la IA de los NPC de Gen 1), `Grunt` (max-damage usando la fórmula de daño y la tabla de tipos), `GymLeader` (Grunt más manejo de HP, boosts y curación), `PokeEnvHeuristic`, `EmeraldKaizo` (IA de un ROM hack de Emerald) y `BaseRNN` (IL). Para ARES es relevante que `Gen1BossAI` y `EmeraldKaizo` aproximan la IA de los NPC del juego real.
- Vicio y riesgo: hay un **sim2sim gap entre el servidor y los replays**, porque los replays omiten información que el servidor sí envía a los jugadores. El README dice que los replays sirven mejor como preentrenamiento offline-to-online. Los modelos de paper eran débiles en Gen9 porque asumían que no había Team Preview, es decir, el espacio de observación estaba atado al formato. Además, el GXE local "no tiene conexión" con el ladder público.

## 3. PokéChamp (Karten, Nguyen, Jin; ICML 2025 spotlight; arXiv 2503.04094) [V]
- URL: https://arxiv.org/abs/2503.04094 , https://github.com/sethkarten/pokechamp (191★, push 23-11-2026)
- Enfoque: búsqueda minimax donde el LLM reemplaza (1) el muestreo de acciones del jugador, (2) el modelado del oponente y (3) la función de valor. No entrena el LLM. Formato: Gen 9 OU.
- Resultados (abstract): con GPT-4o gana 76% contra el mejor bot LLM (PokéLLMon) y 84% contra el bot por reglas más fuerte (Abyssal). Con Llama 3.1 8B gana 64% contra PokéLLMon+GPT-4o. Elo **proyectado** de 1300–1500 en el ladder, que ubica al agente en el top 30%-10%.
- Dataset: más de 3M partidas reales, de las cuales más de 500k son de Elo alto. Agrega "battle puzzles" y benchmarks de habilidades específicas.
- Bots baseline en el repo: `abyssal` (Abyssal Bot), `max_power`, `one_step` (lookahead de un paso), `random`.
- Vicios: el Elo es "proyectado" y no viene de un ladder sostenido; depende de un LLM cerrado (costo y latencia por turno); el motor local fue modificado por los autores.

## 4. PokéLLMon (Hu et al., 2024; arXiv 2402.01118) [V]
- URL: https://arxiv.org/abs/2402.01118 , https://github.com/git-disl/PokeLLMon (sin actividad desde 10-2024)
- Enfoque: LLM (GPT-4) con in-context RL basado en feedback textual, knowledge-augmented generation (KAG) y consistent action generation (self-consistency). Formato: Gen 8 Random Battles.
- Win rate contra el heurístico de poke-env: Random 1.2%, MaxPower 10.4%, GPT-4 vanilla 26%, +KAG 58%, +SC 64%, humanos 59.84%. Contra humanos: 48.57% en el ladder (105 batallas) y 56% contra un jugador invitado (50 batallas).
- Vicios documentados: **panic switching** (cambios consecutivos ante un rival fuerte); vulnerable a estrategias de **desgaste/stall**: el win rate cae a 18.75% (n=16) frente a 53.93% en el resto; planificación miope; alucinación de tipos (GPT-4 acierta solo 84% de las ventajas de tipo); muestras chicas.

## 5. VGC-Bench (Angliss, Cui, Hu, Rahman; arXiv 2506.10326) [V]
- URL: https://arxiv.org/abs/2506.10326 , https://github.com/cameronangliss/vgc-bench (56★)
- Formato VGC (dobles). Los autores estiman alrededor de 10^139 configuraciones de equipo. Dataset de más de 700K logs humanos con open team sheets.
- Baselines: R (random), MBP (max base power), SH (SimpleHeuristics de poke-env), LLM (Llama-3.1-8B), BC, y PSRO (self-play, fictitious play, double oracle) inicializado al azar o desde BC.
- Resultado: en mirror match con un solo equipo vence a un jugador profesional de VGC. Al aumentar el número de equipos, el mejor algoritmo de un equipo rinde peor y se vuelve más explotable, aunque generaliza mejor a equipos no vistos.
- Protocolo: 200 batallas por par de agentes, pero solo 20 para el LLM; promedio de 5 runs.

## 6. PokaiTrainer (Max Yu; arXiv 2608.29197, ago-2026) [V abstract]
- Búsqueda de equilibrio en tiempo de decisión (estilo Student of Games / CFR sobre public belief states) para VGC. El motor PokaiEngine en Rust enumera la distribución completa de resultados con alrededor de 99% de paridad con Showdown.
- Resultados: con open team sheets gana 59% de 150 sets Bo3 contra humanos de alrededor de 1320 Elo, se mantiene en 1350-1400 y tuvo un pico de 1492 (top 500). Los autores dicen que es el primer agente VGC con rating en el ladder en vivo.
- Dato clave: "la política de la red sola pierde incluso contra un heurístico de búsqueda superficial"; la fuerza viene de la búsqueda.

## 7. Foul Play (pmariglia) [V parcial]
- URL: https://github.com/pmariglia/foul-play (387★, push 10-2026). Motor poke-engine en Rust (https://github.com/pmariglia/poke-engine).
- Juega singles de todas las generaciones, salvo dynamax y z-moves (README). Usa búsqueda MCTS/expectiminimax compilada por generación.
- Campeón de Gen 9 OU en PokéAgent. En el ladder de Gen9 Random Battle tuvo un pico de ~2341 Elo y 88 GXE, según el blog de Jaxcalibur [V vía tercero].
- Vicio: el motor se compila por generación, sin un modelo unificado multi-gen; todo el conocimiento está hard-codeado.

## 8. Jaxcalibur (blog, 2026) [V blog]
- URL: https://jaxcalibur.github.io/ . Gen 9 Random Battle. Transformer no causal de 8.5M con heads auxiliares (predice el item, la habilidad, los movimientos y el Tera del rival, y el win prob). PPO con self-play puro, alrededor de 100M partidas en 1 H100 durante ~1 semana, sobre un motor reimplementado en JAX. En test agrega búsqueda pUCT con 32 "mundos" muestreados; el autor estima que la búsqueda suma 100–150 Elo.
- Resultados: pico de 2557 Elo y 95.3 GXE; #1 del ladder alrededor del 20% del tiempo; 68% contra jugadores top. El mejor humano tiene 95.5 GXE.
- Limitaciones: no liberó código ni pesos; una sola ventana de ladder de 3 semanas.

## 9. poke-env y Pokémon Showdown como simulador multi-gen [V]
- URL: https://github.com/hsahovic/poke-env (542★, activo 11-2026). Interfaz Python para Showdown que ofrece RandomPlayer, MaxBasePowerPlayer y SimpleHeuristicsPlayer, más `cross_evaluate`. Casi todos los trabajos de batalla de esta lista dependen de él (Metamon, PokéChamp, PokéLLMon, VGC-Bench).
- Showdown simula las Gens 1–9 en singles/dobles y en formatos Smogon/VGC. PokéAgent excluye movimientos "Past" y aplica restricciones competitivas de DVs en Gen 1 [V].

## 10. lmgame-Bench (Hu et al., ICLR 2026; arXiv 2505.15146) [V abstract]
- Tiene una API Gym unificada y scaffolds de percepción y memoria; incluye Pokémon Red sobre PyBoy. Identifica 3 problemas: percepción visual frágil, sensibilidad al prompt y **contaminación de datos**, y propone scaffolds para mitigarlos. Evalúa 13 modelos. El RL sobre un solo juego transfiere a juegos no vistos.
- En las slides de ICLR (vía búsqueda) se dice que en Red el control de batalla es "too easy" y la navegación desde la imagen cruda "too hard" [NV en detalle].

## 11. Orak (Krafton, arXiv 2506.03610) [V]
- 12 juegos, Pokémon Red entre ellos, con interfaz MCP. En Red la meta es vencer a Brock, medido con 12 flags de historia; el estado es textual (ubicación, party, inventario, estado de batalla, texto en pantalla).
- Puntaje normalizado: GPT-5 88.9±4.8, humanos novatos 86.1±12.7, Gemini-2.5-pro 83.3, DeepSeek-R1 75.0, Claude-3.7 63.9±19.2, GPT-4o 38.9, Qwen2.5-72B 27.8, Llama-3.3-70B 16.7; o3-mini, GPT-4o-mini y los modelos chicos sacan 0.

## 12. VideoGameBench (Zhang, Griffiths, Narasimhan; arXiv 2505.18134) [V abstract]
- 10 juegos de los 90, entre ellos Pokémon Crystal [V vía búsqueda], con 3 juegos secretos. Solo usa visión cruda. El mejor modelo (Gemini 2.5 Pro) completa 0.48% de VideoGameBench y 1.6% de la versión Lite, que pausa el juego mientras el modelo infiere.

## 13. Otros papers 2025-2026 relevantes [V abstract]
- **Continual Harness / Gemini Plays Pokémon** (arXiv 2605.09998, may-2026): GPP completó Blue, Yellow Legacy (hard) y Crystal sin perder una batalla, con refinamiento del harness con humano en el loop. Continual Harness automatiza ese refinamiento online sin resets en Red y Emerald.
- **ASH** (arXiv 2605.14211): aprende de video de internet sin etiquetas usando un IDM. En Emerald alcanza en promedio 11.2/12 hitos contra 9.3/12 del mejor baseline, en 8 horas.
- **PokeRL** (arXiv 2604.10812): PPO en Red con wrapper anti-loop y anti-spam de menús y recompensa jerárquica densa; documenta loops, menu spam y deambular improductivo.
- **Automatic Generation of High-Performance RL Envs** (arXiv 2603.12145): traduce Showdown a **PokeJAX** y PyBoy a EmuRust; con 200M parámetros, el overhead del entorno cae bajo 4%. **TCGJax** se presenta como control de contaminación.
- **Porimon** (arXiv 2609.32544): LLM con Long/Short-Term KAG y una API de cálculo de daño; 15,000 batallas. Supera a PokéLLMon y al heurístico, pero la contribución del Long-Term KAG es "inconclusa".
- **PokéAI** (arXiv 2506.23689): multiagente LLM para Red. Su módulo de batalla gana 80.8% de 50 encuentros salvajes, 6 puntos menos que un humano experimentado.
- **PokeGym** (arXiv 2604.08340): benchmark visual 3D sobre Pokémon Legends: Z-A. G-EvoMAC logra 60.18% de éxito.
- **Teamwork under extreme uncertainty** (arXiv 2212.13338): llegó al puesto #33 mundial en un formato popular usando 4 servidores.
- Menores: LLM Pokémon League (2508.01623) y LLMs as Pokémon Battle Agents (2512.17308). Este último usa un simulador propio simplificado, sin Showdown, así que no es comparable.

---

## Análisis transversal

### Métricas
- En batalla: win rate contra un baseline fijo (PokéLLMon, VGC-Bench, PokéChamp); Elo, Glicko-1 o GXE del ladder de Showdown (Metamon, Jaxcalibur, PokaiTrainer); FH-BT (PokéAgent).
- El GXE local (contra un pool de modelos) **no es comparable** con el GXE del ladder (Metamon lo advierte).
- El Elo "proyectado" (PokéChamp) es más débil que un rating sostenido en ladder.
- En juego/RPG: % de hitos o flags (Orak 12 flags, PokéAgent 15 hitos, ASH 12), tiempo real, número de acciones o button presses. PokéAgent reconoce que "steps/actions/turns" no tienen una definición común.

### Baselines estándar
| Baseline | Dónde aparece |
|---|---|
| Random (acción legal uniforme) | poke-env, PokéLLMon, VGC-Bench, PokéChamp, Metamon |
| Max base power / MaxPower | poke-env, PokéLLMon, VGC-Bench, PokéChamp |
| Max-damage con fórmula+tipos (`Grunt`) | Metamon |
| SimpleHeuristicsPlayer (poke-env) | PokéLLMon (bot de referencia), VGC-Bench, Metamon |
| Abyssal Bot (reglas) | PokéChamp |
| One-step lookahead | PokéChamp |
| IA de NPC del juego (`Gen1BossAI`, `EmeraldKaizo`) | Metamon |
| Foul Play (búsqueda) | PokéAgent, Jaxcalibur (como referencia) |
| LLM vanilla | PokéLLMon, VGC-Bench, PokéAgent |
| Humanos (ladder / pro / speedrunner) | casi todos |

### Reproducibilidad
- Buena: Metamon (checkpoints, datasets y equipos en HF), VGC-Bench (código + 700K logs), PokéAgent (living leaderboard), poke-env.
- Mala: Jaxcalibur (sin código ni pesos); los harness de GPP iterados con humanos; las evaluaciones en ladder no se pueden repetir porque la población humana cambia y no es estacionaria.
- Los LLM cerrados cambian de versión, así que los resultados no se pueden reproducir exactamente. Varios trabajos usan muestras pequeñas: 20 batallas para el LLM en VGC-Bench, 16 en el análisis de stall de PokéLLMon.

### Contaminación
- lmgame-Bench la reconoce explícitamente como problema y diseña para mitigarla.
- PokéAgent solo argumenta que el conocimiento de preentrenamiento no basta, sin test formal.
- 2603.12145 usa TCGJax (spec privada) como control de contaminación.
- Riesgo para LLMs: Bulbapedia, Smogon y los walkthroughs están en el preentrenamiento. Los "hitos" de Red y Emerald están documentados paso a paso en internet. PokéAgent observa que el retrieval de wikis puede **empeorar** el desempeño por desajuste de versión.

### Desajuste Showdown vs batallas in-game (clave para ARES)
1. **Oponente**: en Showdown el rival es un humano o bot competitivo con equipos optimizados (OU/VGC). En el juego, los NPC tienen IA simple (Metamon modela `Gen1BossAI`, que mayormente elige al azar con preferencia por super-efectivos), equipos fijos y niveles bajos. Una política entrenada en OU está sobreespecializada para un rival mucho más fuerte y distinto.
2. **Información**: Showdown OU tiene Team Preview desde Gen 5; en Gen 9 los modelos de Metamon fallaban por asumir que no había preview. En el juego el rival suele estar oculto, pero sus equipos son conocidos y fijos (contaminación útil o "memorizable").
3. **Recursos persistentes**: en el juego el HP, los PP, los ítems y el estado persisten entre combates, se puede usar ítems en batalla, hay EXP, niveles, evoluciones, captura, huida y encuentros salvajes. Showdown resetea todo en cada partida y prohíbe ítems de bolsa. El objetivo cambia de "ganar 1 partida" a "progresar minimizando costo (tiempo, curaciones, riesgo de blackout)".
4. **Reglas y mecánicas**: Showdown aplica cláusulas Smogon (Sleep, Species, Evasion, OHKO) y restricciones de DV/EV competitivas [V parcial: DVs Gen1 en PokéAgent]. El cartucho no tiene cláusulas, tiene glitches propios y cambia mecánicas entre versiones (Showdown marca movimientos "Past"). Que los glitches del cartucho estén bien reproducidos en Showdown por generación queda [NV].
5. **Interfaz**: Showdown entrega estado estructurado y acciones discretas (4 movimientos + 5 cambios). El juego exige leer la pantalla y navegar menús con botones, y ahí aparecen el menu spam y los loops (PokeRL). El Battling Track y el Speedrunning Track de PokéAgent están separados justamente por esto.
6. **Formato**: singles 6v6 frente a dobles (VGC) frente a encuentros 1vN en el juego (los líderes llevan de 1 a 6 Pokémon). Lmgame-Bench sugiere que la batalla in-game de Red es "demasiado fácil" para los LLM fuertes, porque el cuello de botella está en navegar.

---

## Referencias
- arXiv 2601.19562 (no Pokémon): https://arxiv.org/abs/2601.19562
- PokéAgent Challenge: https://arxiv.org/abs/2603.15563 ; https://arxiv.org/html/2603.15563v2 ; https://pokeagentchallenge.com
- Metamon: https://arxiv.org/abs/2504.04395 ; https://github.com/UT-Austin-RPL/metamon ; https://metamon.tech
- PokéChamp: https://arxiv.org/abs/2503.04094 ; https://github.com/sethkarten/pokechamp
- PokéLLMon: https://arxiv.org/abs/2402.01118 ; https://github.com/git-disl/PokeLLMon
- VGC-Bench: https://arxiv.org/abs/2506.10326 ; https://github.com/cameronangliss/vgc-bench
- PokaiTrainer: https://arxiv.org/abs/2608.29197
- Foul Play: https://github.com/pmariglia/foul-play ; https://github.com/pmariglia/poke-engine
- Jaxcalibur: https://jaxcalibur.github.io/
- poke-env: https://github.com/hsahovic/poke-env ; Showdown: https://github.com/smogon/pokemon-showdown
- lmgame-Bench: https://arxiv.org/abs/2505.15146 ; https://github.com/lmgame-org/GamingAgent
- Orak: https://arxiv.org/abs/2506.03610 ; https://github.com/krafton-ai/Orak
- VideoGameBench: https://arxiv.org/abs/2505.18134
- Continual Harness (GPP): https://arxiv.org/abs/2605.09998
- ASH: https://arxiv.org/abs/2605.14211
- PokeRL: https://arxiv.org/abs/2604.10812
- PokeJAX/TCGJax: https://arxiv.org/abs/2603.12145
- Porimon: https://arxiv.org/abs/2609.32544
- PokéAI: https://arxiv.org/abs/2506.23689
- PokeGym: https://arxiv.org/abs/2604.08340
- GameVerse: https://arxiv.org/abs/2603.06656
- Teamwork under extreme uncertainty: https://arxiv.org/abs/2212.13338
