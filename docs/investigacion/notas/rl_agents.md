# Agentes RL puros para Pokemon (pixeles / RAM) -- notas de investigacion

Fecha: 2026-11-24. Estado: COMPLETO (v1).

## 1. Pleines et al. 2025 -- "Pokemon Red via Reinforcement Learning" (arXiv 2502.19920)
- Autores: Pleines, Addis, Rubinstein, Zimmer, Preuss, Whidden. v1 27-feb-2025, v2 11-mar-2025. Codigo: MarcoMeter/neroRL rama poke_red.
- Obs: pantalla gris 72x80 (downsample x2), stack de 3 frames + crop binario 48x48 de coordenadas visitadas + vector de estado (HP y nivel de party, flags de eventos). Sin especie/stats/movimientos.
- Accion: 7 botones discretos (sin Select); boton presionado 8 frames + 16 liberado = 1 decision cada 24 frames. ~9.403 steps/s crudos, ~392 decisiones/s (Ryzen 7 2700X).
- Simplificaciones: solo ~20% del juego (hasta Cerulean), arranca despues del parcel de Oak con Squirtle; modo "Fast" (texto rapido, sin animaciones); presupuesto de pasos 10.240 + 2.048 por evento.
- Recompensa: R_event=+2/evento; R_nav=+0.005 por coordenada nueva; R_heal=2.5*sum(dHP)/HPmax; R_lvl=0.5*min(sum lvl, (sum lvl-22)/4+22).
- Red: Nature CNN por entrada visual + FC para estado -> FC o GRU; ~2M params (~4M con GRU). PPO, 32 workers, batch 65.536, gamma 0.997, sin entropia.
- Compute: ~36 h por run (CPU+GPU); GRU solo-CPU ~24 dias/run. 5 semillas, eval 30 episodios x 25 puntos.
- Resultados (Tabla II): baseline 4.19/7 hitos; Brock 0.99, llega Cerulean 0.85, Misty 0.27, Cerulean completo 0.00. Fast 4.40. Humano 11.188 pasos a Cerulean vs agente 25.299.
- Reward hacking: Bulbasaur farmea Zubats con Leech Seed y cura -> 399 curaciones vs 22.97 humano, 0 llegadas a Cerulean; Charmander/GRU pelea cerca de Pallet y cura con la mama; 10x R_nav -> explora y no derrota a Brock (1.04 hitos); sin R_nav -> 0 hitos. Descuento temporal sesga a elegir Charmander (peor vs Brock).
- Nadie aprende Cut (HM01) -> gimnasio Vermilion inalcanzable.

## 2. PokeRL de Rubinstein/Whidden + PufferAI (drubinstein.github.io/pokerl, repo drubinstein/pokemonred_puffer)
- Claim: "As of February 2025" vence Pokemon Red con politica <10M params, "minimal simplifications" (v1 sitio: 4-mar-2025).
- Caveats propios (chapter-4/results): no hay run estable que gane con TODOS los scripts desactivados; si hay victorias quitando cada script por separado. Puzzles de STRENGTH (Sokoban) siguen por script. Usa swarming, muchas recompensas de conocimiento del juego, y recompensas por Map ID (su "least favorite addition").


- Scripts ACTIVOS segun chapter-2/env-setup: tirar items no clave si bolsa llena, DINERO INFINITO, enseñar/usar STRENGTH, resolver puzzles STRENGTH, enseñar/usar SURF fuera de batalla, Pokeflute automatica, FLASH automatico, bloquear salida de Indigo Plateau tras la liga.
- Scripts de desarrollo ya retirados: enseñar/usar CUT, stats maximos, meter bebidas al entrar a Celadon Mart, ascensores automaticos, desactivar batallas salvajes.
- Inicio: tras "Parcel Delivery"; starter Bulbasaur/Charmander (garantiza CUT); Lapras regalo garantizado en Silph.
- "Mini-episode" resets: limpian mapas de exploracion, sets de coords CUT/SURF/flute, warps, carteles, NPCs cada full_reset_frequency + jitter. Episodio termina solo con soft-lock o victoria.
- config.yaml (verificado via gh, repo drubinstein/pokemonred_puffer, push 2025-12-04): action_freq 24; auto_flash, auto_use_strength/surf, auto_teach_surf/strength, auto_solve_strength_puzzles, auto_pokeflute, auto_remove_all_nonuseful_items = True; infinite_money True; save_state True; swarm True; num_envs 288; batch 65.536; bptt_horizon 16; hidden 512 (LSTM); torch.compile; total_timesteps 10_000_000_000 con comentario "100_000_000_000 for full games". Entrenamiento "between 7 hours and 1 week" (sitio).
- Obs (chapter-2/observations): pantalla 72x80 gris + mascara de no-visitado + vector de eventos WRAM (con 4 eventos añadidos a mano) + direccion + blackout map ID + inventario + party + pasos Safari + flag Lapras. Regla declarada: nada que un humano no sepa (discutible: inventario/eventos RAM).
- Recompensa: ">25 parametros" (exploration por coord en mini-episodio, boost por map IDs importantes, signs, hidden objs, warps, ~80 required events, badges, seen/caught, moves, HM tiles, cut/surf/flute coords validas/invalidas, items requeridos/utiles, menus, a_press (retirada), pokecenter_heal, ball use, safari, nivel S<15: S; si no 30+(S-15)/4).
- Reward hacking reportado: slot machines del Game Corner (recompensa por objeto); spam de A por recompensa de menus/A; farmeo de niveles en Lavender Tower (boost condicionado a Silph Scope). Ellos mismos dicen que el boost de map ID puede ser "putting the game on rails".
- Swarming (chapter-3/swarm): al cumplir un objetivo requerido se guarda savestate y TODOS los envs cargan ese estado (inspirado en Go-Explore). Se resetea al ganar. Es esencialmente un curriculum por savestates compartidos -> no es un agente que juega una partida continua.

## 3. Peter Whidden -- PokemonRedExperiments (github.com/PWhiddy/PokemonRedExperiments) + video YouTube (youtu.be/DcYLT37ImBY, oct-2023)
- ~7.9k estrellas (gh, 2026-09). PyBoy + Stable-Baselines3 PPO.
- Video (33 min) y TechCrunch 18-oct-2023: ">50.000 horas" de juego simulado; recompensas por nivel, exploracion, batallas, gimnasios. Exploracion original = novedad de frames (KNN sobre frames) -> el agente se queda mirando el agua/pasto animados en Pallet (novelty trap / "noisy TV"). Anecdota PC: deposito accidental baja suma de niveles 25->12; tras un evento negativo asi "the AI will avoid it entirely" (evita el Pokecenter/PC). No podia leer dialogos: atascado con el parcel de Oak -> Whidden inicia despues de ese paso con Squirtle (savestate init.state). En el video llega a Mt. Moon; tras ajustes sale y llega a Cerulean.
- V2 (19-oct-2024, verificado en codigo v2/red_gym_env_v2.py): obs = screens 72x80x3 stack + health + level (fourier) + mapa de exploracion local 48x48 + recent_actions; recompensa = event*4 + heal*10 + badge*10 + explore 0.1*coords vistas + stuck -0.05; reemplaza KNN de frames por exploracion por coordenadas. baseline_fast_v2.py: 64 envs SubprocVecEnv, ep_length 2048*80=163.840 pasos, PPO MultiInputPolicy, gamma 0.997, ent 0.01, action_freq 24, init_state '../init.state'. README: V2 "Reaches Cerulean".
- Compute PokeRL (chapter-3/running): 4 maquinas donadas, cada una RTX 4090 + i9-14900K + 128 GB RAM; pico ~10.000 SPS ("10x"); run bajo de ~1 dia a 7 h tras tuning; Brock en <30 min; sweeps (CARBS) >1 mes; antes $200/mes en vast.ai.
- Politica (chapter-2/policy): PPO + LSTM 128 unidades; ~5M params (~20MB). CNN 3 capas (32/64/64, kernels 8/4/3 stride 2) sobre pantalla 72x80 + mascara visitada; embeddings de map ID, blackout map, items (x cantidad/100), party (especie, tipos, movimientos + stats) -> lineal 6; 2560 flags de eventos de RAM filtrados; one-hot direccion y tipo de batalla; concat -> 512 -> LSTM. NOTA: el sitio dice <10M en titular pero la pagina de policy dice ~5M; el config usa hidden 512.
- OBS: aunque declaran "no conocimiento que un humano no tenga", la party embebe especie/tipos/movimientos y 2560 flags de eventos via RAM: es fuertemente RAM-dependiente y especifico de Gen 1 (direcciones pokered).

## 4. PokeGym (PufferAI/pokegym, rama de xinpw8 "BET_pokegym_badge_3_400m"), 2023-11 a 2024-06
- Fork fuertemente modificado de Whidden sobre PufferLib 0.5/0.7. README (verificado gh): Badge 1 a 9.6M steps, Badge 2 a 35M, Bill salvado 53M, primer HM01 a 60M (86% envs obtienen HM01, 68% enseñan Cut), Badge 3 a 404M steps. Metricas como "first time" en algun env (no tasa por semilla) -> metrica optimista.

## 5. PokeRL (Mudireddy & Patibandla, arXiv 2604.10812, 12-abr-2026) -- codigo github.com/reddheeraj/PokemonRL
- Gymnasium + PyBoy, SB3 CnnPolicy ~1.07M params; obs 72x80 gris + mascara visitada por mapa centrada en el origen del mapa; stack 4 -> 72x80x8. Acciones: 7 (flechas, A, B, no-op); Start/Select ELIMINADOS ("the agent learned to save the game"). Movimiento = doble press (Gen1 gira primero y luego camina).
- RAM solo para reward/terminacion (pos Y 0xD361, X 0xD362, map 0xD35E, party 0xD163, battle 0xD057, HP 0xD16C).
- Recompensa jerarquica: micro (+1 tile nuevo, +0.2*dist euclidiana, +0.5 primera visita), meso (+10 nuevo mapa, +5, +2), macro (+20 pasto, +10 batalla, +50 captura/victoria). Penalizaciones suaves -0.02 a -0.2 (penalizaciones fuertes colapsaban el entrenamiento).
- Curriculum de 3 secuencias con savestates: salir de casa, llegar al pasto, primera batalla con rival.
- Anti-loop: penalizar visitas >3/>5 a misma casilla, detectar patrones en ventana de 20 acciones, loops de posicion. Episodios-loop: 41.2% -> 4.7% (n=1000). Distribucion antes: A 32.1%, no-op 28.4%, movimiento 27.2% (button spam); despues movimiento 68.2%. Entropia 1.21 -> 1.82 bits.
- Mascara visitada: posiciones unicas 34.2 -> 48.1; cobertura Pallet 12% -> 41%; revisit 4.8 -> 3.1 (300-500k steps).
- Resultados: salir de casa ~65% tras 150k steps; pasto ~60% a 500k; win vs rival ~50% a 500k. PPO lr 3e-4, gamma 0.999, n_steps 2048, batch 128, 10 epochs, 4 envs, 1 CPU+GPU.
- Vicios: solo early-game, savestates por secuencia, reward hecha a mano, admite "manual human trigger" en dialogos; 5 fallos vanilla documentados (loops infinitos, sparsity, button spam, semantica de movimiento, exploracion sin memoria).

## 6. PokeAgent Challenge (NeurIPS 2025), Track 2 speedrun Pokemon Emerald (arXiv 2603.15563, Karten et al.)
- Objetivo: primer gimnasio de Emerald; accion final debe salir de una red neuronal. Obs: frames + estado limitado (party, HP); puzzles/items/movimientos ocultos.
- "raw frontier VLMs achieve effectively 0% task completion" sin harness. Humano promedio 1:22:05; mejor speedrunner humano 18 min. 6 de 22 equipos al 100%.
- 1o Heatz 40:13: LLM descompone subobjetivos con success_cond(state), LLM escribe scripts por subobjetivo (puede consultar Qwen3-VL-8B), destilacion por imitacion a red neuronal y luego DQN con buffer sembrado con trayectorias expertas, epsilon experto 0.1->0. 
- 2o Hamburg PokeRunners 01:14:43: RL "puro" recurrent PPO + LSTM; frames gris 128x128 -> CNN; MLP con codificacion sinusoidal de x,y, one-hot de ubicacion, vector binario de 38 milestones. Recompensa SOLO positiva (sub-milestones, ubicaciones, exploracion, medallas, dinero, HP curado, subidas de nivel, ataques con daño); penalizaciones causaron "only safe behavior". Por compute limitado entrenaron varios modelos pequeños y los cosieron en inferencia (= politica por segmento, no generalista).
- Track 1 batallas: Metamon (RL offline + self-play; 30 checkpoints, RNN a Transformers 200M; >4M trayectorias humanas + 18M self-play) "competitive with top human players". Todos los top de batalla usan RL o busqueda, no LLM.

## 7. PyBoy (github.com/Baekalfen/PyBoy) -- emulador/gym base
- ~5.2k estrellas, activo (push 2026-10-30). Plugins de game wrapper incluyen game_wrapper_pokemon_gen1 y game_wrapper_pokemon_pinball; NO hay wrapper Gen2 (Gold/Silver/Crystal) ni GBA. set_emulation_speed(0) sin limite; README habla de speed-ups ~100x realtime (6.000 fps). Para Gen3 (Emerald/FireRed) hay que usar mGBA/libretro (PokeAgent usa su propio emulador/API).

## 8. JEPA multimodal sobre Pokemon Red (Campese & Moschitti, ICML 2026 workshop, openreview 8li5NOHh9S)
- Encoder JEPA pixeles+RAM preentrenado en trayectorias offline, congelado como extractor para PPO. 48 estados iniciales held-out. Frozen JEPA 2-8x la recompensa acumulada vs DINOv3, init aleatoria o fine-tuning completo. JEPA(init DINOv3) 4.48 +- 0.15 vs DreamerV3 end-to-end 4.19 +- 0.80 (unidad no especificada en la pagina). Fine-tuning degrada representacion (MSE progresion de eventos x3.7). Diversidad de datos > pericia: politica random sesgada con starts diversos supera trayectorias expertas. (Solo pagina ICML verificada; detalles del paper no verificados.)

## 9. Pokemon Gold / Crystal
- Tesis LIACS Leiden (Philip Edelaar, 20-ago-2025, supervisor Mike Preuss): "Testing Game Experience Using Deep RL and Curriculum Learning" -- dos agentes RL, uno con curriculum, sobre Pokemon Gold simplificado; abstract: el juego simplificado sigue siendo representativo "for most of the game". Numeros: no verificado.
- Bas de Haan (Medium, mayo 2024): "RL 101: AI plays Pokemon" -- agente en Pokemon Gold, recompensas desde RAM; alcance del agente no verificado.
- matiaszanolli/PokemonCrystalRL (3 estrellas, 2025): hibrido LLM-RL, PyBoy, recompensas ad hoc (+100 primer Pokemon, +500 medalla), sin resultados reportados. Evidencia de que Crystal casi no tiene trabajo RL serio.
- Emerald 2018 (Game Developer, Caleb Compton 2020): NO es RL; Gym Retro + RAM + A* y exploracion de frontera; batalla no implementada.

## 10. Otros
- PokeAI (arXiv 2506.23689): multi-agente LLM para Red (fuera de alcance RL puro).
- Metamon (arXiv 2504.04395 "Human-Level Competitive Pokemon via Scalable Offline RL..."): solo batallas Showdown.
- "AI Finds A Way" (arXiv 2608.23875, Dharna...Clune, ago-2026): 26 anecdotas de soluciones inesperadas/reward hacking; no verificado si incluye Pokemon.

# SINTESIS: vicios de los agentes RL de Pokemon
1. Reward shaping a mano, especifico de juego (critico): >25 parametros en PokeRL, ~80 required events con nombres de Red, boosts por map ID ("on rails"). No transfiere a otro juego ni version.
2. Reward hacking documentado (alto): curacion farmeada (399 vs 23 curas humanas), Leech Seed en Zubats, mama en Pallet, slot machines, spam de A/menus, farmeo de niveles en Lavender, "noisy TV" con agua/pasto animados (KNN de frames), auto-guardar partida (PokeRL 2026).
3. Aversion por un unico evento negativo (medio): deposito en PC -> evita la accion para siempre; penalizaciones negativas -> "only safe behavior" (Hamburg) o colapso (PokeRL 2026).
4. Scripts/hacks que vacian el problema (critico): dinero infinito, HMs y puzzles de Strength automaticos, Flash/Pokeflute automaticos, inicio tras Parcel, Lapras garantizado, tirar items; "beat the game" nunca demostrado con todos los scripts apagados.
5. Savestates/swarming como curriculum oculto (alto): todos los envs saltan al estado del mejor -> no hay una partida continua desde el inicio; evaluaciones con "first env to reach X".
6. Dependencia de RAM de Gen 1 (alto): flags de eventos, party, inventario por direcciones pokered; observacion no generaliza a Gen2/3 ni a hacks.
7. Compute enorme (alto): 10^10 pasos configurados (10^11 "for full games"); 4x4090 donadas; >50.000 h simuladas (Whidden); 24 dias/run GRU CPU (Pleines). Badge 3 a 404M pasos (PokeGym).
8. Sin generalizacion (critico): un juego (Red), un ROM, mismo RNG/semillas; ningun trabajo evalua en otro juego. Hamburg cose modelos por segmento.
9. Evaluacion debil (alto): metricas de "algun env alcanzo X", pocas semillas, sin humano baseline salvo Pleines; eval tan cara como entrenar; claims del sitio (<10M params, "beat") vs paper (Cerulean, 0% Cerulean completo).
10. Lectura de texto/dialogos inexistente (medio): parcel de Oak, Map ID rewards como sustituto de entender texto (dicho por autores).
11. Exploracion por conteo de coordenadas (medio): requiere coords de RAM, resets de mini-episodio, mascara visitada; no captura progreso semantico.
12. Semantica de acciones mal modelada (bajo): doble-press de Gen1 (giro vs paso), Start/Select eliminados por spam.

# Ideas para ARES (resumen)
- Observacion universal: pixeles (+ opcional pequeño vector RAM via adaptador por juego) para que la politica no dependa de direcciones Gen1; RAM solo para recompensa/eval.
- Recompensas genericas multi-juego: novedad de estado semantico (pantalla/mapa) + progreso de hitos comunes (medallas, Pokedex, dinero, nivel) definidos por adaptador por juego; evitar recompensas positivas farmeables (curar/niveles con tope y decaimiento).
- Exploracion: RND/curiosidad sobre embeddings robustos a animaciones (evitar noisy TV), archivo tipo Go-Explore declarado explicitamente.
- Encoder preentrenado (JEPA/autoencoder) en trayectorias de varios juegos -> representacion compartida; congelar para PPO (evidencia Campese & Moschitti).
- Evaluacion honesta: run continuo desde partida nueva sin savestates, lista explicita de scripts on/off, >=5 semillas, tasa por semilla, steps a hito y comparacion con humano.
- Accion con macro-acciones (paso de 1 tile, abrir menu) y masking por contexto (overworld/batalla/dialogo).

# Referencias
- https://arxiv.org/abs/2502.19920 ; https://arxiv.org/html/2502.19920v2 ; https://github.com/MarcoMeter/neroRL/tree/poke_red
- https://drubinstein.github.io/pokerl/ (chapter-2/env-setup, observations, rewards, policy; chapter-3/running, swarm; chapter-4/results)
- https://github.com/drubinstein/pokemonred_puffer (config.yaml)
- https://github.com/PWhiddy/PokemonRedExperiments (v2/red_gym_env_v2.py, v2/baseline_fast_v2.py) ; https://youtu.be/DcYLT37ImBY
- https://techcrunch.com/2023/10/18/after-50000-hours-this-ai-can-play-pokemon-red
- https://github.com/PufferAI/pokegym
- https://arxiv.org/abs/2604.10812 ; https://github.com/reddheeraj/PokemonRL
- https://arxiv.org/abs/2603.15563 ; https://pokeagent.github.io/track2.html
- https://icml.cc/virtual/2026/83721 ; https://openreview.net/forum?id=8li5NOHh9S
- https://github.com/Baekalfen/PyBoy
- https://theses.liacs.nl/pdf/2024-2025-EdelaarPPhilip.pdf
- https://medium.com/ordina-data/reinforcement-learning-101-ai-plays-pok%C3%A9mon-e0626bd6beae
- https://github.com/matiaszanolli/PokemonCrystalRL
- https://gamedeveloper.com/programming/how-i-taught-an-ai-to-play-pokemon-emerald
- https://arxiv.org/abs/2504.04395 ; https://arxiv.org/abs/2608.23875 ; https://arxiv.org/abs/2506.23689
