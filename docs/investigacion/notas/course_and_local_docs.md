# Material del curso IPD434 (rama 2026-2) + documentos locales de referencia

Fuente del curso: `TEL-USM/IPD434@2026-2`, descargado vía `gh api .../contents/<path>?ref=2026-2` (download_url). Números de línea = líneas del .md descargado.

## A. Lo que el curso exige (Clases/00_Introduccion/00_Introduccion.md)

- Nota final: `NF = 0.5 C + 0.3 PR + 0.2 LP` (00_Introduccion.md:~165-175). El proyecto vale 30 %.
- Resultados de aprendizaje (00_Introduccion.md:67-77): analizar complejidad, distinguir criterios de aprendizaje/búsqueda, **seleccionar y aplicar** difusa/EC/NN, **evaluar con métricas, líneas base y supuestos explícitos**, comunicar en formato proyecto + artículo breve.
- Ruta del curso (00_Introduccion.md:79-89): delimitar problema -> representación -> mecanismo -> evaluar (métricas, línea base, costo) -> defender (supuestos, **variabilidad**, límites).
- Proyecto (00_Introduccion.md:199-215): formulación + estado del arte, técnica + justificación, **protocolo experimental reproducible**, **resultados vs línea base**, artículo IEEE + presentación. Advertencia literal (paráfrasis): un modelo sofisticado sin pregunta clara, línea base o métricas pertinentes no es evidencia.
- "Suficientemente buena" debe traducirse a un criterio medible (00_Introduccion.md:~125).

## B. Conceptos NN enseñados (notación exacta)

### 04_RedesNeuronales.md
- ML supervisado, particiones train/val/test, hiperparámetros vía validación (:253-450).
- Métricas clasificación (matriz de confusión, precisión, recall, F1, ROC AUC) y regresión (MAE, MSE, RMSE, R^2 "comparación con línea base") (:452-738).
- Perceptrón, MLP, capas densas, activaciones (ReLU, Leaky ReLU, sigmoide, tanh, softmax) (:888-1150).
- Pérdida: entropía cruzada + softmax (:1152-1180); mini-batch GD, épocas (:1182-1206); **SGD y Adam** (:1208).
- Backprop derivado a mano (:1237-1376); ciclo PyTorch (:1401-1480).
- Sub/sobreajuste, penalización, dropout, early stopping (:1545-1632).
- LSTM/RNN (:1647-1685), CNN: convolución, pooling, CNN CIFAR-10 (:1686-1783).
- **Diseño experimental antes de entrenar** (:1784-1796): definir tarea/métrica; **construir línea base simple**; separar datos; **fijar presupuesto, semillas, arquitectura e hiperparámetros**. "Una red más profunda solo es mejor si supera una línea base bajo comparación justa".
- **Reporte** (:1797-1808): seleccionar con regla definida antes de mirar test; evaluar test una vez; **informar variabilidad entre semillas, tiempo de entrenamiento, recursos**.

### 05-1_ReinforcementLearning.md (Deep RL)
- Ciclo agente-entorno `s_t --a_t--> (r_{t+1}, s_{t+1})` (:99-110).
- Elementos: agente, entorno, estado, acción, recompensa, política `pi(a|s)` (:112-132).
- **MDP** `M=(S,A,P,R,gamma)`, propiedad de Markov (:134-158).
- Retorno `G_t = R_{t+1} + gamma R_{t+2} + ...` (:160-172); `V^pi(s)` (:174-185); **Ecuación de Bellman** (:187-215).
- Mundo grilla determinista/estocástico (:217-275); **Value Iteration** (:276-288); política óptima argmax (:290-300).
- **Q-learning** `Q(s,a) <- Q + alpha[r + gamma max Q(s',a') - Q]` (:302-314); `V=max_a Q`, política codiciosa (:316-327).
- **DQN**: red de política `Q_theta`, red objetivo `theta^-`, objetivo `y_t = r_{t+1} + gamma(1-d_t) max Q_{theta^-}(s_{t+1},a')` (:329-366); epsilon-greedy, replay memory, minibatches aleatorios, copia periódica a red objetivo, decaimiento de epsilon (:368-376).
- **Gymnasium** API `reset()`, `step()` -> `terminated`/`truncated` (:378-388); notebook DQN CartPole PyTorch (:390-399).
- **NO se enseña**: policy gradient, REINFORCE, actor-critic, PPO, A2C, ni RL intrínseco/curiosidad (grep sin coincidencias). Implicancia: usar PPO es legítimo pero debe justificarse/explicarse fuera del material; DQN es lo "canónico" del curso.

### 05-2_Transformers.md
- Tokenización (BPE, WordPiece, SentencePiece), Word2Vec, RNN y sus límites (:34-350).
- Atención, autoatención, Q/K/V, atención producto punto escalado (dividir por sqrt(d_k), :746-760), multi-head (:807-846), bloque Transformer, residual + norma, FFN por posición, máscaras de relleno y causal (:848-920), posicionales (:960), encoder/decoder BERT/GPT/T5 (:972-985), KV cache, FlashAttention, MoE (:986-1050).
- Síntesis (:1054-1066). No hay ViT ni Decision Transformer; transfer learning sólo en NLP de modo implícito.

### 05-3_GAN_AE.md
- AE `z=f_theta(x), x_hat=g_phi(z)` (:74-92); usos: reducción de dimensión, anomalías, **preentrenamiento/extracción de características** (:93-110).
- Denoising AE (:111-127), AE convolucional (:128), VAE + ELBO + KL (:173-205), reparametrización (:206-222), GAN minimax (:223-266).
- Evaluación: "Representación -> desempeño en tarea posterior" (:268-282).

### 05-4_AdvancedCNNArchitectures.md
- Vanishing gradients, **bloque residual `y = F(x;W)+x`** (:82-141), Inception, 1x1, EfficientNet/MBConv, BatchNorm, data augmentation, mixup, label smoothing (:153-282).
- **Transferencia de aprendizaje**: cargar base preentrenada sin cabeza, congelar, entrenar cabeza, descongelar parcial con LR menor (:284-296).
- **Comparar arquitecturas**: calidad, parámetros, FLOPs, latencia, **robustez bajo cambio de dominio** (:298-311).

### 05_EA_Avanzado.md / 03_ComputacionEvolutiva.md (skim)
- Selección, mutación, recombinación, ACO, PSO (05_EA_Avanzado.md:74-306).
- **Diseño experimental avanzado** (05_EA_Avanzado.md:322-336): mismo presupuesto de evaluaciones, **varias semillas independientes**, comparación con heurística simple, **mediana, dispersión y tasa de éxito**, tiempo/memoria. Comparar contra una sola ejecución no permite concluir superioridad.
- 03: formulación, exploración vs explotación, metaheurísticas, criterios de reinicio/estancamiento (:339-577).

### 02_SistemasDifusos
- El .md no existe en la rama 2026-2 al path pedido? (curl devolvió vacío); el directorio contiene .ipynb, .html, .pdf, .md listados por la API. (pendiente si relevante; fuera del foco NN).

## C. Tabla concepto -> archivo/sección -> aplicación en ARES

| Concepto del curso | Archivo:líneas | Cómo aplicarlo en ARES (proyecto 1 semana) |
|---|---|---|
| MDP `(S,A,P,R,gamma)`, retorno, Bellman | 05-1_ReinforcementLearning.md:134-215 | Formular el subproblema (p.ej. batalla 1v1 o navegación a objetivo) explícitamente como MDP en el paper: S = vector derivado de RAM (`core/observations.py` encode_battle/encode_world_state), A = acciones discretas de botón/menú, R definida y justificada. |
| Q-learning / DQN (red objetivo, replay, epsilon-greedy) | 05-1:302-376 | **Línea base natural y alineada al curso**: DQN sobre el mismo vector de estado. El Blueprint descarta "PPO/DQN directo" (Blueprint p.2 §1.1) -> convertir esa afirmación en hipótesis testeable: método propuesto vs DQN. |
| Gymnasium `reset/step/terminated/truncated` | 05-1:378-388 | Envolver un sub-entorno ARES (p.ej. `core/stadium/deep.py:38 StadiumEnvironment`) con API Gymnasium -> baseline DQN y reproducibilidad. |
| MLP, softmax+CE, Adam, backprop | 04_RedesNeuronales.md:933-1376 | Cabezas de política/valor; clonación de comportamiento (supervisado) sobre trazas como baseline adicional. |
| Dropout, early stopping, sub/sobreajuste | 04:1545-1632 | Regularización y selección de época por validación. |
| **Diseño experimental + baseline + semillas** | 04:1784-1808; 05_EA_Avanzado.md:322-336 | Mismo presupuesto de pasos de entorno, N>=5 semillas, mediana + IQR + tasa de éxito, tiempo y memoria. Test una sola vez (p.ej. juego/mapa no visto). |
| Atención, Q/K/V, multi-head, máscaras | 05-2_Transformers.md:400-920 | Encoder de entidades (Pokémon/movimientos como tokens), máscara de relleno para slots vacíos y máscara de acciones ilegales. Encaja con el "Entity-Transformer" del Blueprint. |
| Transfer learning (congelar base, nueva cabeza, fine-tune) | 05-4_AdvancedCNNArchitectures.md:284-296 | **Cross-game**: entrenar en FireRed, congelar encoder, ajustar cabeza en Emerald (o zero-shot). Forma más defendible y medible de "jugar cualquier Pokémon". |
| Robustez bajo cambio de dominio | 05-4:298-311 | Métrica central: desempeño en juego no visto vs visto (gap de generalización). |
| Bloque residual | 05-4:106-141 | Encoder con residuales si se usan píxeles. |
| AE / denoising AE / VAE como preentrenamiento | 05-3_GAN_AE.md:74-222, 268-282 | Latente de estado compartido entre juegos; evaluar por "desempeño en tarea posterior". Alternativa enseñada a H-JEPA/VICReg (no enseñado). |
| CNN, pooling | 04:1686-1783 | Sólo si se opta por píxeles (más general entre juegos, más costoso). |
| EA / QD | 05_EA_Avanzado.md:74-306; 03:416-577 | Opcional: neuroevolución/MAP-Elites como baseline u oponentes diversos (ver paper 2601.19562). |

Conceptos usados por ARES que **no están en el curso**: PPO/policy gradient/actor-critic, PSRO, H-JEPA/VICReg, GFlowNet (Trajectory Balance), Mamba/SSM, Active Inference/EFE, SCM/do-calculus, Z3/SMT, HTN. Requieren justificación y explicación propia en un informe de sólo 5 páginas.

## D. Documentos locales

### D1. G:\ARES\2601.19562v2.pdf
- **Título**: "Tournament Informed Adversarial Quality Diversity". Autores: T. Anne, N. Syrkis, M. Elhosni, F. Turati, A. Manai, F. Legendre, A. Jaquier, S. Risi (IT Univ. Copenhagen + armasuisse S+T). arXiv 2601.19562v2 [cs.NE], 15 May 2026; preprint aceptado en GECCO '26 (DOI 10.1145/3795095.3805154) (p.1-2).
- **Método**: extiende GAME (Generational Adversarial MAP-Elites, coevolución QD con MTMB-ME). Propone selección de tareas informada por torneo (**Ranking**, **Pareto**) vs **Behavior** (original) y **Random**; y 6 métricas de QD adversarial (Win rate, ELO, Robustness, Coverage, Expertise, AQD score) (p.1-6).
- **Setup**: Pong, Cat-and-mouse, Pursuers-and-evaders en JAX; políticas = MLP (32,16) evolucionado por mutación gaussiana; descriptor de comportamiento vía embedding CLIP del video (p.6).
- **Resultados**: 20 réplicas por variante, presupuesto igualado (1.5M evaluaciones), mediana + cuartiles, Holm-Bonferroni p=0.05. Ranking es mejor en Win rate, ELO, Expertise y AQD en los tres problemas (Tablas 1-3, p.7-8).
- **Relevancia**: (1) candidato válido para la componente **LP (artículo <=5 años)** (00_Introduccion.md:217-227): EC + NN + coevolución; (2) **plantilla de protocolo** (presupuesto igualado, 20 semillas, mediana/IQR, corrección múltiple); (3) relevancia técnica a Pokémon moderada: liga/oponentes diversos (ARES ya tiene `core/battle/psro_league_trainer.py`), métricas ELO/win-rate. No trata Pokémon ni generalización cross-game.

### D2. G:\ARES\ARES_PKMN_Master_Architecture_Blueprint_v3.pdf (20 pp., skim p.1-4)
- "Blueprint Maestro v3.0-FINAL": sistema neuro-simbólico para Gen 3 GBA con **Inferencia Activa (EFE), SCM do-calculus, H-JEPA (VICReg), GFlowNets (TB), Entity-Transformer + Mamba-2 SSM, verificación Z3/SMT entera ARM7TDMI, HTN, A* 3D**, sobre libmgba FFI (p.1-3).
- §1.1 (p.2): rechaza "RL libre de modelo (PPO/DQN directo de píxeles o RAM)" por recompensas dispersas en horizonte >10^6 frames; declara no depender de LLM externos.
- 4 niveles jerárquicos (p.3); 10 correcciones A-01..A-10 de direcciones/offsets de memoria (p.4).
- **Visión original = específica a Gen 3** (direcciones RAM, structs de pret). Esto **contradice** la visión actual "jugar CUALQUIER Pokémon": el nivel 1 entero depende de la ROM.

### D3. G:\ARES\gemini-code-1791938796398.xml (17 KB, skim)
- Es un **prompt de sistema** para un agente de código ("Codex Agentic Harness — GPT 6.1 SOL", FULLY_AUTONOMOUS) con el Blueprint como "single source of truth": construir todo sin humano; prohíbe stubs/TODO; exige doble backend (libmgba real + `SyntheticGBASiliconBridge` que simula EWRAM/IWRAM); impone A-01..A-10; termina cuando pytest + mypy + `scripts/run_autonomous_verification_demo.py` dan exit 0. Secciones: system_directive, mission_objective, strict_engineering_rules, repository_blueprint, locked_api_contracts, subagent_orchestration_graph, verification_gate.
- Implicancia: el criterio de "terminado" fue **verificación de software (tests verdes)**, no evidencia experimental de aprendizaje -> lo que el curso dice que no basta (00_Introduccion.md:211-213).

## E. Visión original vs implementado (evidencia rápida)

| Componente Blueprint | Archivo | Observación |
|---|---|---|
| H-JEPA + VICReg | core/world_model/hjepa.py:16 (130 líneas) | Instanciado en `core/orchestrator.py:217` en `.eval()`. Entrenado por `scripts/train_neural_modules.py:81` con `collect_world_transitions` (`core/training_data.py:15-26`): transiciones en **`SyntheticGBASiliconBridge` con mapas sintéticos** (12x14, bordes de colisión, metatiles constantes), no del juego real. Validez ecológica dudosa. |
| GFlowNet TB | core/planner/gflownet_team.py:31 (108 líneas) | Usado en orchestrator.py:218, 738-764. |
| Entity-Transformer + Mamba-2 | core/battle/mamba_entity_net.py:35 `SelectiveSSMBlock` (201 líneas) | Sin clase "Transformer"; `AresBattlePolicyNetwork` (orchestrator.py:27). SSM propio simplificado. |
| Z3 verifier | core/safety/z3_integer_verifier.py (orchestrator.py:69-71, :214) | Existe. |
| Active Inference / EFE | orchestrator.py:991 `self.causal.evaluate_free_energy` | Presente. |
| HTN | core/planner/htn_planner.py | Existe. |
| RL (no en Blueprint) | core/stadium/{learning,deep,league}.py, core/battle/psro_league_trainer.py | PPO/PSRO presentes pese a que el Blueprint los descartó. **DQN: 0 archivos**. |
| "Cualquier Pokémon" | — | Sólo FireRed/Emerald (mismo motor Gen 3). |
- Historial git: 6 commits.

## F. Conclusiones para el proyecto del curso
1. El curso premia **pregunta clara + baseline + semillas + métricas**, no arquitectura sofisticada (00_Introduccion.md:211-213; 04:1784-1808; 05_EA_Avanzado.md:322-336).
2. El RL del curso termina en DQN; PPO es aceptable pero debe explicarse. DQN debería ser al menos la **baseline**.
3. "Jugar cualquier Pokémon" en 1 semana debe reducirse a algo medible: **transferencia FireRed -> Emerald** con encoder compartido + transfer learning (05-4:284-296) y robustez bajo cambio de dominio (05-4:298-311).
4. La pila del Blueprint (JEPA, GFlowNet, Mamba, Z3, EFE) es mucha superficie no enseñada y difícil de evaluar en 5 páginas; aislar UN componente neural y compararlo contra baseline.
5. Paper 2601.19562 sirve como LP y como plantilla de protocolo.
