# Auditoria: stack neuronal + Stadium (coevolucion)

Estado: EN PROGRESO (notas incrementales). Fecha auditoria: 2026-11-24. HEAD = 1031f3a "WIP".

## 1. Modelos neuronales (core/)

### 1.1 H-JEPA world model — core/world_model/hjepa.py
- Arquitectura: MLP encoder 96->128->128 (LayerNorm+SiLU), target encoder EMA (decay .996), embedding de 16 acciones, predictor MLP 256->256->128 (hjepa.py:19-33).
- Loss: VICReg-style (MSE pred vs target EMA *25 + varianza*25 + covarianza) (hjepa.py:63-76); multi-horizonte (1,2,4) en forward_sequence_loss (hjepa.py:91-114).
- Es JEPA "real" en el sentido minimo (online/target EMA, no generativo, latente), pero NO jerarquico: el nombre "H-JEPA" no corresponde a ninguna jerarquia temporal/espacial. Es un MLP de ~60k params.
- Planificacion: plan_actions = argmin de distancia latente a goal, sobre K candidatos dados (hjepa.py:116-130).
- Datos: core/training_data.py:16-59 -> transiciones de un `SyntheticGBASiliconBridge` (mapas sinteticos 12x14 con bordes, acciones aleatorias de 4 direcciones). No son datos del juego real (ROM).
- Consumo en campaña: orchestrator.py:1424-1447 `_plan_movement` solo elige walk vs run entre 2 candidatos sobre un path YA calculado por A*/BFS; y en nativo con Running Shoes se sobreescribe `selection = 1` (orchestrator.py:1440-1444). => DECORATIVO / impacto minimo.
- Online learning de JEPA en campaña (orchestrator.py:1406-1421) cada 32 transiciones si online_learning.

### 1.2 GFlowNet de equipo — core/planner/gflownet_team.py
- Species policy MLP 411->128->128->411; move policy MLP (411+355)->128->355; log_Z escalar (gflownet_team.py:33-45).
- Loss: Trajectory Balance (MSE(logZ + logPF - logPB, logR)) con PB uniforme (log(step+1)) (gflownet_team.py:94-98). TB correcto en forma.
- Reward: coverage_reward heuristico = exp(#tipos/6 + cobertura - riesgo) (gflownet_team.py:100-108). Reward hecho a mano, no viene de resultados de batalla.
- Consumo en campaña: orchestrator.py:727-765 `_refresh_roster` -> `desired_species` / `desired_moves`. desired_species se usa en 1043 y 1120 (decidir capturar especie). desired_moves solo se guarda/registra (orchestrator.py:759, 3192). Ademas desired_species se mezcla con `planner.counter_targets` (763) y targets heuristicos (2495). => influencia debil (sesgo en que capturar), no hay evidencia de beneficio medido.

### 1.3 Battle policy "Mamba2 + Entity Transformer" — core/battle/mamba_entity_net.py
- Entity encoder MLP 48->128, role embedding (7), TransformerEncoder 3 capas, 4 heads, d=128, ff=256, pre-norm (mamba_entity_net.py:150-167).
- SelectiveSSMBlock: implementacion Mamba-2 portable en PyTorch puro: A escalar por head, dt por head, B/C agrupados, conv1d causal depthwise, gated RMSNorm, D skip, cache recurrente; forward_sequence con kernel semiseparable por chunks (mamba_entity_net.py:34-141). Es una reimplementacion plausible del SSD (no usa kernels Triton oficiales; no hay test de equivalencia contra mamba_ssm visto aqui).
- IMPORTANTE: el SSM se aplica a 1 token por turno (step_recurrent, mamba_entity_net.py:197) => funciona como RNN recurrente sobre turnos; durante PPO tambien se usa step a step (psro_league_trainer.py:226). forward_sequence (SSD por chunks) no se usa en el entrenamiento.
- Cabezas: action 9 (4 moves + 5 switch), target 3, value 1.
- Consumo en campaña: orchestrator.py:971-975 se computan logits, pero el score final de cada move es `efe - expected_damage/maxhp*4 - logits*0.05` y target `*0.02` (orchestrator.py:1000-1003). La decision la dominan EFE causal + estimacion de dano heuristica; la red es un tie-breaker de peso 0.05 => practicamente DECORATIVA.

### 1.4 PSRO league trainer — core/battle/psro_league_trainer.py
- PPO real: GAE (lambda .95), ratio clipping .2, value loss, entropia .01, grad clip .5, approx_kl (psro_league_trainer.py:207-305). Recurrente con cache SSM por paso.
- PSRO: freeze_current_policy, payoff matrix simetrica por ambas orientaciones, meta-solver = fictitious play (psro_league_trainer.py:308-371), best response vs mezcla Nash (373-387). Estructura PSRO correcta en forma.
- ENTORNO = `BatchedIntegerBattleArena` (psro_league_trainer.py:68-172): 1v1, HP=100 fijo, 4 movimientos con potencias permutadas {20,40,80,120}, sin tipos, sin switches, sin estados. Rival baseline: siempre elige el move mas fuerte con potencias {20,40,70,90} (asimetria a favor del agente). Es un juguete: la politica optima es "elige la potencia mas alta" (observable en features 29:33).
- Entrenamiento (scripts/train_neural_modules.py): 128 updates JEPA+TB + ciclos PPO de 8 updates x 32 steps x 2 envs, eval 64 episodios. metrics.jsonl: run completo ~80 s (t 1792083755 -> 1792083834), win_rate final 1.0 en arena juguete; training.log: win_rate 0.39 -> 0.625, VICReg 63.6 -> 39.3, TB 3662 -> 180. Una sola seed (42), sin baselines ni varianza.

### 1.5 Veredicto stack neuronal de campaña
- Los cuatro modulos son implementaciones reales y correctas EN FORMA (JEPA+EMA+VICReg, TB-GFlowNet, Mamba-2 SSD portable, PPO+GAE+PSRO/fictitious play). Tests unitarios existen.
- Pero en la campaña nativa son practicamente decorativos: battle policy pesa 0.05 frente a EFE+dano heuristico (orchestrator.py:1000-1003); JEPA solo elige walk/run y se sobreescribe (orchestrator.py:1440-1444); GFlowNet solo sesga que especies capturar (orchestrator.py:1043,1120), reward heuristico.
- El PPO "de campaña" se entrena en una arena juguete (psro_league_trainer.py:68-172) donde win 1.0 es trivial; no hay evidencia de transferencia a combates reales. "H-JEPA" no es jerarquico. El nombre "Mamba2" es correcto en parametrizacion pero se usa como RNN de 1 paso por turno.
- No usar estos numeros (training.log, metrics.jsonl) como resultado del informe: 1 seed, ~80 s, entorno trivial, sin baseline.

## 2. Stadium (core/stadium/*)

### 2.1 Motor Gen-3 (engine.py, 1474 lineas)
- Singles 6v6 deterministas por seed (engine.py:261-299, rng=random.Random(seed)); clone() para contrafactuales (engine.py:295).
- Dano: SIEMPRE via `GbaIntegerSafetyVerifier.compute_context_damage_reference` con roll 85-100 (engine.py:856, 1351); tabla de tipos de la ROM en orden nativo; ~124 ramas de efectos (`effect ==/in`); habilidades relevantes; estados, clima, pantallas, Puas, Sustituto, Proteccion (engine.py:1214), criticos por etapas (engine.py:929).
- Limites declarados (docs/STADIUM_ADVERSARIAL.md sec.3): Metronomo, Transformacion, Relevo, etc. -> evento `unsupported` contado (engine.py:687-688); Patada Baja peso medio; sin objetos de mochila. Dex desde data/stadium_dex.json (exportado de ROM) (dex.py:66).
- Tests: tests/test_stadium.py (24) + tests/test_stadium_rules.py (8): 32 passed en 17 s (ejecutado en esta auditoria).
- Fidelidad: alta para un simulador propio (dano entero verificado), pero NO validada batalla-a-batalla contra el emulador; no es Showdown. Suficiente para un experimento de soft computing si se declara como "simulador Gen-3 aproximado con dano exacto".

### 2.2 Politicas (policies.py, features.py)
- MLPGenome: option-scorer 25 features -> 16 -> 8 -> 1, LeakyReLU, + temperatura evolucionable (policies.py:35, 53-95). 561 parametros. Mutacion paper: gaussiana sigma .1 sobre 30% de pesos (policies.py:84-90).
- Features por opcion (features.py:27-30): dano estimado, prob KO, precision, efectividad CREIDA (aprendida de mensajes), estado, boosts, heal, prioridad, switch, hp, velocidad, amenaza... => features muy ingenieriles (gran parte del "conocimiento" esta en las features).
- Prior experto embebido exactamente (policies.py:36-39, 110-128).
- Baselines existentes: RandomPolicy (policies.py:172), GreedyDamagePolicy (183-199: max dano*precision, switch solo forzado), NeuralPolicy (202-229, AresBattlePolicyNetwork, 532,837 params), HybridPolicy (232-252).
- Contrato de informacion: rival con stats estimadas publicas (features.py:39-42); test anti-fuga mencionado en docs.

### 2.3 Algoritmos de coevolucion (game_qd.py, league.py, learning.py, deep.py)
- GAME / MTMB-ME (Anne et al., arXiv:2601.19562): GrowingArchive por tarea (game_qd.py:50-103), seleccion de tareas ranking (k-means sobre vectores de ranking normalizados, game_qd.py:208-212, 294-295), pareto (230), behavior, random. Es QD estilo MAP-Elites con archivo creciente (CVT-like por cercania), no grid clasico.
- Lecciones contrafactuales: rollouts de ambas ramas desde clones con CRN (learning.py:60-117); operador lamarckiano: fine-tuning supervisado (softmax sobre opciones, Adam manual en numpy, ancla L2) (learning.py:119-188).
- Canal profundo (deep.py): StadiumEnvironment vectorizado (deep.py:38-116), reward = delta de margen de HP + 2*resultado (deep.py:100-105); behavior cloning de elites -> PPO recurrente (reusa PSROLeagueTrainer) contra elites rivales -> destilacion a MLP estudiante (deep.py:118-211). league.py:374-403.
- NOTA: aqui NO hay PSRO propiamente dicho en el Stadium; el PSRO solo existe en la arena juguete. La coevolucion es GAME (QD adversarial).

### 2.4 Velocidad (medida en esta auditoria, 1 proceso, scratchpad bench)
- MLP/greedy/random: ~130-250 combates/s por proceso (greedy vs random 203/s; prior-MLP vs greedy 133/s; random vs random 255/s).
- NeuralPolicy (533k params, torch 1 hilo, CPU): ~5.7 combates/s (=> 25-40x mas lento). Implicacion: PPO con la red grande en 1 semana es viable solo con presupuestos modestos (~20k combates/hora/proceso).
- Liga historica: run-20261027-232544 -> 520,187 combates en 5531 s (~94/s con pool paralelo; docs dicen ~120/s con 10 procesos).

### 2.5 Baselines medidas (bench propio, n=150, equipos FireRed/Emerald alternando lados)
- greedy vs random: 0.713 win side0. prior-MLP (no evolucionado) vs greedy: 0.42. random vs random: 0.467.
- Untrained NeuralPolicy (sampling) vs random, mirror FireRed, n=30: 0.80 -> Random es un baseline muy debil (cambia sin sentido); greedy es el baseline relevante.
- Rosters muy pequenos: FireRed 9 mons, Emerald 5 mons (team_bank). Asimetria fuerte ROJO >> AZUL (Venusaur vs Swampert x4). Para experimento limpio: usar mirror matches o rosters fijos y alternar lados.

### 2.6 Resultados existentes en artifacts/stadium
- compare-20261027-234444/report.md: 6 replicas, Ranking/Pareto >> Behavior/Random en win_rate y expertise de ROJO (significativo, Mann-Whitney+Holm). Pareto > Ranking en win_rate. AZUL: Pareto mejor.
- ablation-20261028-002210/report.md (ablation-v2.log): 5 replicas x 10 gen, "completo" (GAME + lecciones + deep + intercambio) vs "game": TODAS las metricas en negrita = NO hay diferencia significativa. Ventaja media +0.048 ROJO, +0.014 AZUL, 3/5 replicas > 0. => Las extensiones propias (lecciones, deep, intercambio) NO muestran mejora estadisticamente significativa.
- Liga larga run-20261027-232544 (330 gen): anclas greedy/random = 1.0 para ROJO al final; deep: imitation_accuracy 0.524, ppo_win_rate 1.0 (ROJO vs elites AZUL, problema facil), student_accuracy 0.898. Docs: aprendiz de AZUL solo 5% vs elites ROJO; la red profunda va por detras de los MLP evolucionados.
- Progreso absoluto vs ancla greedy no crece en la liga larga (docs sec.8: ROJO 0.91->0.86).

### 2.7 WIP commit (1031f3a, 17-nov)
- 18 archivos, +1573/-409: deep.py nuevo (239), learning.py reducido (-340/+), dex/engine/narration/teams, tests/test_stadium_rules.py nuevo (125), scripts .bat/.sh, data/stadium_dex.json y stadium_teams.json (snapshot para correr sin ROM). Tests pasan.

## 3. Recomendacion para el proyecto del curso (1 semana)
- Base mas prometedora: el Stadium (motor + baselines + harness de evaluacion con workers + estadistica Mann-Whitney/Holm ya escrita), NO la campaña nativa.
- Experimento propuesto: "Politica neuronal de combate vs baselines" en el Stadium con multiples seeds:
  - Brazos: Random, Greedy (heuristico), Prior experto (MLP con pesos a mano), MLP neuroevolucionado (GAME/ES simple), MLP entrenado por PPO/BC, red Transformer+SSM (opcional, cara).
  - Metrica: win-rate vs greedy y vs pool fijo de oponentes, mirror matches + intercambio de lados, >=5 seeds, IC bootstrap.
  - Ojo con "cross-game": rosters FireRed/Emerald comparten mecanica Gen-3; generalizacion "entre juegos" aqui = generalizacion entre equipos/rosters (entrenar con FireRed, evaluar con Emerald), que es un experimento honesto y barato.
- Evitar afirmar que JEPA/GFlowNet/Mamba controlan la campaña: no lo hacen (ver 1.5).
