# Agentes generalistas cross-game + métodos NN modernos aplicables (IPD434 Soft Computing)

Fecha de corte: 2026-11-24. Autor: agente de investigación (workflow ARES-PKMN).
Leyenda: **[V]** = verificado por mí contra fuente primaria (abstract arXiv vía API, README de GitHub vía `gh api`, post oficial). **[NV]** = no verificado (recuerdo o fuente secundaria; NO citar como dato duro).

Objetivo de estas notas: qué se transfiere entre juegos, cómo unificar acciones, memoria, cómputo, mejor repo open-source, y qué unidad del curso demuestra cada técnica (Deep RL; Transformers y transfer learning; AE/VAE/GAN; CNN avanzadas; computación evolutiva). Cierre con un stack concreto factible en ~1 semana en 1 GPU de consumo.

---

## 1. Agentes generalistas cross-game (escala industrial)

### 1.1 VPT — Video PreTraining (OpenAI, 2022) [V]
- URL: https://arxiv.org/abs/2206.11795 · repo: https://github.com/openai/Video-Pre-Training (MIT, ~1.8k stars [V])
- Enfoque: se entrena un **Inverse Dynamics Model (IDM)** con poco dato etiquetado (contratistas), que **pseudo-etiqueta** video de internet; luego Behavior Cloning sobre ese video y fine-tune con BC y RL. Interfaz nativa: mouse+teclado a 20 Hz [V abstract].
- Resultados: primeras herramientas de diamante en Minecraft (~24.000 acciones, >20 min humanos) [V]. Modelo BC entrenado con 70.000 h de video etiquetado por IDM; <2.000 h usan datos de contratistas con etiqueta real [V blog OpenAI].
- Cómputo: enorme (cifra exacta de GPUs/días **[NV]**; el apéndice da 30 epochs, batch 880 según búsqueda secundaria [NV]).
- Transfiere: el *truco del IDM* (pocos datos etiquetados -> mucho video etiquetado). Para Pokémon: grabar ~horas propias con BizHawk/PyBoy registrando botones, entrenar IDM (frame_t, frame_t+k -> botón), etiquetar longplays de YouTube de *todas* las generaciones.
- Vicio: un solo juego; sin memoria larga explícita; política "sistema 1".
- Unidad: Deep RL + Transformers/transfer learning + CNN.

### 1.2 SIMA (DeepMind, mar-2024) y SIMA 2 (dic-2025) [V]
- URLs: https://arxiv.org/abs/2404.10179 · https://arxiv.org/abs/2512.04797 · blog https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/
- SIMA 1: imagen + instrucción en lenguaje -> teclado/mouse, sin API del juego [V]. 9 juegos comerciales de 8 estudios + 4 entornos de investigación; 600 habilidades básicas; ~1.500 tareas [V blog]. Generalista ≈165% del especialista; en juego no visto ≈90% del especialista; sin lenguaje ≈30% (lectura del gráfico del blog) [V blog, aprox.].
- SIMA 2: sobre Gemini; razona metas, conversa; **auto-mejora**: Gemini genera tareas y recompensas en mundos nuevos [V abstract]. Cifras exactas de brecha con humanos [NV].
- Transfiere: grounding lenguaje-acción; interfaz humana común. Cerrado (sin pesos).
- Vicio: no reproducible; cómputo inalcanzable; evaluación parcialmente con jueces humanos.
- Unidad: Transformers y transfer learning.

### 1.3 Genie 1 / 2 / 3 — latent actions y world models (DeepMind) [V parcial]
- Genie 1: https://arxiv.org/abs/2402.15391 — 11B params; tokenizer espacio-temporal + dinámica autoregresiva + **Latent Action Model (LAM)** aprendido sin etiquetas de acción; el espacio latente permite imitar comportamiento de videos no vistos [V]. Datos: videos de plataformas 2D (cifras de horas [NV]). Repo oficial: no hay (404 `google-deepmind/genie`) [V]; existen reimplementaciones comunitarias [NV].
- Genie 2 (blog 4-dic-2024): mundos consistentes hasta ~1 min (mayoría 10-20 s), control teclado/mouse, recuerda zonas fuera de vista; probado con SIMA dentro de mundos generados [V blog].
- Genie 3 (5-ago-2025): tiempo real 720p a 24 fps, consistencia de unos minutos, memoria visual ~1 min; research preview limitado [V vía búsqueda de blog oficial].
- Transfiere: **acciones latentes** = unificación de acción sin etiquetas, clave para "cualquier Pokémon" (Gen 1-3 GB/GBA: D-pad+A/B/Start/Select; DS: + táctil).
- Vicio: cerrado; horizontes de minutos, no de 30 h de un RPG.
- Unidad: AE/VAE/GAN (VQ-VAE tokenizer, LAM tipo VQ) + Transformers.
- Versión barata: **LAPO** (https://arxiv.org/abs/2312.10812, repo schmidtdominik/LAPO) recupera la estructura del espacio de acciones real solo desde video en Procgen y luego fine-tune con pocos datos etiquetados o con RL [V]. **LAPA** (https://arxiv.org/abs/2410.11758) usa VQ-VAE para acciones latentes discretas entre frames [V].

### 1.4 NVIDIA NitroGen (ene-2026, CVPR 2026 oral) [V]
- URL: https://arxiv.org/abs/2601.02427 · repo https://github.com/MineDojo/NitroGen (~2.3k stars [V]) · HF nvidia/NitroGen
- Enfoque: 40.000 h de gameplay de >1.000 juegos; acciones extraídas automáticamente de **overlays de gamepad** en videos públicos; BC a gran escala; benchmark multi-juego [V abstract]. Arquitectura derivada de GR00T N1, SigLIP 256x256 [NV-secundario]. README: "500M parameter DiT that can only see the last frame" [V README].
- Resultados: hasta 52% de mejora relativa en tasa de éxito vs. desde cero en juegos no vistos [V].
- Limitaciones (del propio README) [V]: sin memoria (solo último frame), no planifica horizontes largos, no juega end-to-end, sistema-1; solo juegos Windows; licencia NVIDIA (no-OSI).
- Transfiere: **espacio de acción gamepad unificado** — encaja casi perfecto con consolas Nintendo. Para Pokémon: punto de partida para el "motor reactivo" (movimiento/menús), no para estrategia.
- Unidad: Transformers/transfer learning (+ difusión como modelo generativo).

### 1.5 Game-TARS (ByteDance, oct-2025) [V]
- URL: https://arxiv.org/abs/2510.23691 (familia UI-TARS: https://github.com/bytedance/UI-TARS, ~11.7k stars [V]; pesos de Game-TARS [NV])
- Enfoque: espacio de acción unificado anclado a teclado-mouse nativo; preentrenamiento continuo >500B tokens (OS, web, juegos); *decaying continual loss* contra confusión causal; *Sparse-Thinking* [V].
- Resultados: ~2x éxito sobre SOTA previo en Minecraft abierto; cercano a humanos novatos en juegos web 3D no vistos; supera GPT-5, Gemini-2.5-Pro y Claude-4-Sonnet en benchmarks FPS [V abstract].
- Unidad: Transformers y transfer learning.
- Vicio: escala industrial; razonar en cada paso es caro (de ahí Sparse-Thinking).

### 1.6 Lumine (ByteDance Seed, nov-2025) [V]
- URL: https://arxiv.org/abs/2511.08892
- Enfoque: VLM end-to-end; píxeles a 5 Hz -> acciones teclado-mouse a 30 Hz; razonamiento adaptativo solo cuando hace falta [V].
- Resultados: completa la historia principal de Mondstadt (5 h) en Genshin Impact a nivel humano; zero-shot: misiones de 100 min en Wuthering Waves y el primer capítulo completo (5 h) de Honkai: Star Rail [V].
- Relevancia Pokémon: es lo más cercano a "terminar un RPG largo"; demuestra transferencia entre juegos del mismo género (mismo publisher/UI parecida — sesgo a considerar).
- Vicio: escala y datos propietarios; transferencia entre juegos muy similares.

---

## 2. World models (aprender a imaginar)

### 2.1 DreamerV3 [V]
- https://arxiv.org/abs/2301.04104 (publicado en Nature, cita en README [V]) · repo danijar/dreamerv3 (~4k stars, MIT, activo 2026 [V])
- Una sola configuración en >150 tareas; diamantes en Minecraft desde cero sin datos humanos [V]. RSSM (GRU + latentes categóricos), symlog, normalización de retornos.
- Cómputo: configs por tamaño (p.ej. `size50m` en README [V]); entrenable en 1 GPU para Atari/Crafter (tiempos [NV]).
- Unidad: Deep RL + AE/VAE (encoder/decoder variacional discreto).

### 2.2 IRIS y DIAMOND [V]
- IRIS https://arxiv.org/abs/2209.00588: autoencoder discreto + Transformer autoregresivo; Atari 100k HNS medio 1.046; supera humanos en 10/26 [V]. Repo eloialonso/iris (GPL-3.0).
- DIAMOND https://arxiv.org/abs/2405.12399: world model de **difusión**; Atari 100k HNS 1.46 (mejor para agentes entrenados solo en el WM); motor neural jugable de CS:GO [V]. Repo eloialonso/diamond (MIT, ~2.1k stars).
- Relevancia Pokémon: un WM de un juego GB (160x144, 4 tonos) es barato; útil para "soñar" batallas/menús y para detectar novedad (error de predicción).
- Unidad: AE/VAE/GAN (tokenizer VQ, difusión como modelo generativo) + Transformers.

### 2.3 Drama (Mamba) [V]
- https://arxiv.org/abs/2410.08893 · repo realwenlongwang/Drama
- World model SSM (Mamba) O(n); Atari100k competitivo con solo 7M parámetros; "entrenable en laptop" [V]. Mejor opción de WM para presupuesto estudiante.

### 2.4 V-JEPA 2 (Meta, jun-2025) [V]
- https://arxiv.org/abs/2506.09985 · repo facebookresearch/vjepa2 (MIT, ~4.9k stars)
- Preentrenamiento auto-supervisado en >1M horas de video; SSv2 77.3 top-1; V-JEPA 2-AC con <62 h de video robot para planificar zero-shot [V].
- Uso realista aquí: **encoder congelado** de features (o solo como inspiración de objetivo JEPA: predecir en espacio latente, no píxeles). Para pixel-art GB probablemente sobredimensionado; JEPA pequeño propio es un buen experimento de curso.
- Unidad: AE/VAE (auto-supervisado) + Transformers/transfer.

---

## 3. Secuencia y memoria

### 3.1 Decision Transformer [V]
- https://arxiv.org/abs/2106.01345 · repo kzl/decision-transformer (MIT)
- RL como modelado de secuencias condicionado al return-to-go; iguala/supera offline RL en Atari/Gym/Key-to-Door [V].
- Uso: preentrenar offline con trayectorias de varios Pokémon (de RL previo + IDM-labeled video), condicionar en "progreso deseado".
- Vicio conocido: depende de la calidad del dataset; no extrapola bien más allá del mejor retorno visto [NV-general].

### 3.2 Mamba / xLSTM para RL [V]
- LRAM https://arxiv.org/abs/2410.22391: xLSTM como "large recurrent action model"; 432 tareas, 6 dominios; comparable a Transformers con inferencia lineal [V]. Repo NX-AI/xlstm (Apache-2.0).
- Drama (arriba) para WM con Mamba [V].
- Por qué importa en Pokémon: episodios de 10^4–10^6 pasos; un Transformer con contexto de 1k frames no alcanza; un estado recurrente (LSTM/xLSTM/Mamba) + **memoria episódica externa** (mapa visitado, flags) es más barato.

---

## 4. Exploración (vicio #1 de los agentes Pokémon)

- **RND** https://arxiv.org/abs/1810.12894 [V]: bonus = error de predecir features de una red aleatoria fija; primer método > humano promedio en Montezuma sin demos.
- **NovelD** (NeurIPS 2021, Zhang et al.) [V vía proceedings]: meta-criterio sobre RND que premia cruzar la frontera de lo explorado (diferencia de novedad entre estados consecutivos) + término episódico; SOTA en MiniGrid y varias tareas de NetHack. Paper: https://papers.neurips.cc/paper_files/paper/2021/file/d428d070622e0f4363fceae11f4a3576-Paper.pdf · código github.com/tianjunz/NovelD [NV repo]. Precursor BeBold https://arxiv.org/abs/2012.08621 [V].
- **E3B** https://arxiv.org/abs/2210.05805 [V]: bonus elíptico episódico sobre un embedding aprendido con **IDM** (captura lo controlable, ignora ruido); SOTA en 16 tareas MiniHack; también VizDoom y Habitat. Repo facebookresearch/e3b.
- **Go-Explore** https://arxiv.org/abs/2004.12919 (Nature) [V]: identifica *detachment* y *derailment*; archivo de celdas, volver (restaurando estado) y explorar. Encaja perfecto con emuladores (save states gratis). Repo uber-research/go-explore.
- Pokémon: la recompensa por "coordenadas nuevas" de PokemonRedExperiments es un count-based tabular; los vicios (loops, menú spam, farmear grass) son exactamente lo que E3B/NovelD/Go-Explore atacan.
- Unidad: Deep RL (+ **computación evolutiva**: Go-Explore es un algoritmo de archivo tipo quality-diversity / MAP-Elites).

---

## 5. Infraestructura RL eficiente

### PufferLib [V]
- https://arxiv.org/abs/2406.12905 · repo PufferAI/PufferLib (~6.9k stars, MIT, activo oct-2026 [V])
- Wrappers + vectorización rápida; PPO propio y tuning de hiperparámetros [V README].
- Pokémon Red: drubinstein/pokemonred_puffer (~215 stars [V]). El proyecto afirma vencer Pokémon Red con política <10M params (feb-2025) [NV — fuente secundaria cita su página], pero su página de resultados admite que **no** hay corrida estable con todos los scripts desactivados, los puzzles de STRENGTH siguen scripteados, usan "swarming" y muchas recompensas de conocimiento del juego; las recompensas por Map ID son "least favorite" porque el agente no lee texto [V página de resultados].
- Pleines et al. "Pokemon Red via RL" https://arxiv.org/abs/2502.19920 [V]: baseline hasta Cerulean City; ablaciones muestran **explotación del reward shaping**.
- PokeRL https://arxiv.org/abs/2604.10812 [V]: agentes degeneran en loops de acción, menu spam y vagabundeo; propone wrapper anti-loop, máscaras de mapa y reward jerárquico denso; solo tareas tempranas (salir de casa, primera batalla rival).

---

## 6. VLMs pequeños + RL (GRPO y variantes)

- **G1 / VLM-Gym** https://arxiv.org/abs/2505.13426 [V]: Qwen2.5-VL-7B con RL puro (G0) y con cold-start perceptual (G1); supera a Claude-3.7-Sonnet-Thinking en sus juegos; percepción y razonamiento se refuerzan mutuamente. Repo chenllliang/G1.
- **GiGPO** https://arxiv.org/abs/2505.10978 [V]: GRPO de dos niveles (episodio + "anchor states" repetidos) para crédito por paso; Qwen2.5 1.5B/3B/7B; >12% sobre GRPO en ALFWorld, >9% en WebShop. Muy relevante: Pokémon tiene muchísimos estados repetidos (mismos menús) -> grupos por estado ancla gratis.
- **Odysseus** https://arxiv.org/abs/2605.00347 [V]: Super Mario Land, 100+ turnos; PPO con critic ligero a nivel de turno **supera a GRPO y Reinforce++** en estabilidad y eficiencia; priors del VLM reducen ingeniería de acciones; ≥3x progreso medio vs. modelos frontera; mejora cross-game.
- **"RL Teaches VLMs Where to Attend but Not How to Perceive"** https://arxiv.org/abs/2610.23983 [V]: el encoder visual casi no cambia con RL; mejora la atención de las capas LLM a tokens visuales. Implicancia: congelar el encoder visual y aplicar LoRA en el LLM es razonable.
- Vicio: GRPO puro es inestable en horizontes largos (Odysseus); costo de inferencia por paso alto (~1-3B params a pocos Hz).
- Unidad: Transformers y transfer learning (+ Deep RL).

---

## 7. Tabla de transferencia

| Método | Qué transfiere entre juegos | Unificación de acción | Memoria | Cómputo | Repo | Unidad |
|---|---|---|---|---|---|---|
| VPT | prior de comportamiento desde video | teclado/mouse nativo | corta (Transformer) | masivo | openai/Video-Pre-Training | DRL, Transf. |
| SIMA/SIMA 2 | grounding lenguaje-acción | teclado/mouse | Gemini | masivo, cerrado | — | Transf. |
| Genie/LAPO | acciones latentes desde video | latente discreto (VQ) | contexto limitado | Genie masivo; LAPO 1 GPU | schmidtdominik/LAPO | VAE/VQ |
| NitroGen | BC multi-juego (1.000 juegos) | gamepad | ninguna (último frame) | medio p/ fine-tune | MineDojo/NitroGen | Transf. |
| Game-TARS/Lumine | VLM + acción nativa | teclado/mouse | razonamiento esparso | masivo | UI-TARS | Transf. |
| DreamerV3/Drama | algoritmo único robusto | discreta por juego | RSSM / Mamba | 1 GPU | danijar/dreamerv3 | DRL, VAE |
| DIAMOND/IRIS | WM generativo | discreta | Transformer/difusión | 1 GPU (días) | eloialonso/diamond | VAE/GAN-difusión |
| V-JEPA 2 | features de video | — | — | inferencia 1 GPU | facebookresearch/vjepa2 | AE auto-sup. |
| DT / LRAM | política offline multi-tarea | tokens de acción | Transformer / xLSTM | 1 GPU | kzl/decision-transformer, NX-AI/xlstm | Transf. |
| RND/NovelD/E3B | bonus intrínseco genérico | — | episódico | barato | facebookresearch/e3b | DRL |
| Go-Explore | archivo de estados | — | archivo explícito | CPU-bound | uber-research/go-explore | Evolutiva/QD |
| PufferLib PPO | infra | — | LSTM | 1 GPU, alto SPS | PufferAI/PufferLib | DRL |
| VLM + GRPO/PPO | conocimiento del mundo del LLM | texto -> botones | contexto | 1 GPU con LoRA 3B (justo) | chenllliang/G1 | Transf. |

---

## 8. Stack recomendado (~1 semana, 1 GPU consumo tipo RTX 3060-4090 + CPU multi-core)

Principio: **separar "qué se transfiere" (percepción + acciones + exploración) de lo específico de cada juego (RAM/recompensas)**, y evitar los vicios documentados (scripts, map-ID rewards, menu spam, loops, una sola ROM).

1. **Entorno multi-juego unificado** (Deep RL): PyBoy (GB/GBC) + mGBA/BizHawk (GBA) con *misma* interfaz: obs = frame 144x160 escala de grises/4 tonos + stack corto; acción = 8 botones gamepad (U/D/L/R/A/B/Start/NOP), frame-skip ~24. Save states para Go-Explore. Vectorizado con PufferLib (CPU paralela).
2. **Encoder visual compartido** (CNN avanzada + AE/VAE): pequeña CNN tipo IMPALA-ResNet o ViT-tiny, **preentrenada auto-supervisada** (VQ-VAE o JEPA pequeño) sobre frames de Red/Crystal/Emerald (y longplays). Demuestra la unidad AE/VAE y es reutilizable entre juegos.
3. **Acciones latentes / IDM** (VAE + transfer): IDM entrenado con trayectorias propias (botón conocido) -> etiqueta longplays de YouTube de otras generaciones (truco VPT) o LAPO si no hay etiquetas. Sirve también como embedding de E3B.
4. **Política recurrente** (Deep RL + memoria): PPO (PufferLib) con núcleo LSTM o xLSTM/Mamba pequeño (~5-10M params) + memoria episódica explícita (conteo de celdas por mapa). Inicializada por BC sobre los datos del IDM.
5. **Exploración sin scripts**: bonus E3B (embedding IDM, episódico) o NovelD sobre RND en vez de "coordenadas nuevas" + recompensa extrínseca mínima *genérica* (badges, nivel del equipo, eventos de flags leídos de RAM con un mapa por juego). Anti-loop/anti-spam estilo PokeRL como penalización.
6. **Go-Explore como capa evolutiva** (Computación evolutiva): archivo de celdas (mapa, x, y, badges) con selección por novedad; restaurar save states para evitar detachment/derailment; las mejores celdas generan demostraciones para robustificar la política (como Go-Explore fase 2).
7. **Opcional / extensión** (Transformers): Qwen2.5-VL-3B con LoRA como planificador lento de alto nivel (texto de diálogos, metas) a baja frecuencia, entrenado con GiGPO/PPO de critic por turno (Odysseus) solo si sobra GPU; o Decision Transformer offline sobre trayectorias multi-juego.

Presupuesto orientativo [NV, estimación propia]: día 1-2 entorno + datos; día 2-3 VQ-VAE/IDM (horas de GPU); día 3-6 PPO+E3B+Go-Explore en Red y Crystal en paralelo (~10^8 pasos con PufferLib es realista en CPU de 16+ núcleos [NV]); día 7 evaluación zero-shot/fine-tune en un tercer juego (p.ej. Emerald o Yellow) midiendo transferencia (pasos hasta 1ª medalla vs. desde cero, como NitroGen mide mejora relativa).

Métricas para el informe: % de transferencia (estilo NitroGen 52%), cobertura de mapas, tasa de loops/menu-spam (estilo PokeRL), badges/hora, sin scripts.

---

## 9. Vicios transversales observados (con evidencia)
- **Dependencia de scripts y conocimiento del juego**: pokemonred_puffer mantiene scripts (STRENGTH), swarming y rewards de map-ID [V].
- **Explotación del reward shaping**: Pleines et al. 2025 [V].
- **Loops, menu spam, vagabundeo**: PokeRL 2026 [V].
- **Sin memoria**: NitroGen solo ve el último frame [V README].
- **Escala no reproducible / cerrado**: SIMA, Genie 2/3, Game-TARS, Lumine [V].
- **Transferencia "fácil"**: Lumine transfiere entre juegos muy similares del mismo publisher [V abstract; interpretación propia].
- **GRPO inestable en horizontes largos**: Odysseus [V].

---

## Referencias
- VPT: https://arxiv.org/abs/2206.11795 · https://openai.com/blog/vpt/ · https://github.com/openai/Video-Pre-Training
- SIMA: https://arxiv.org/abs/2404.10179 · https://deepmind.google/discover/blog/sima-generalist-ai-agent-for-3d-virtual-environments/
- SIMA 2: https://arxiv.org/abs/2512.04797
- Genie: https://arxiv.org/abs/2402.15391 · Genie 2: https://deepmind.google/discover/blog/genie-2-a-large-scale-foundation-world-model/ · Genie 3: https://deepmind.google/discover/blog/genie-3-a-new-frontier-for-world-models/
- LAPO: https://arxiv.org/abs/2312.10812 · https://github.com/schmidtdominik/LAPO · LAPA: https://arxiv.org/abs/2410.11758
- NitroGen: https://arxiv.org/abs/2601.02427 · https://github.com/MineDojo/NitroGen · https://nitrogen.minedojo.org/
- Game-TARS: https://arxiv.org/abs/2510.23691 · UI-TARS: https://github.com/bytedance/UI-TARS
- Lumine: https://arxiv.org/abs/2511.08892
- DreamerV3: https://arxiv.org/abs/2301.04104 · https://github.com/danijar/dreamerv3
- IRIS: https://arxiv.org/abs/2209.00588 · https://github.com/eloialonso/iris
- DIAMOND: https://arxiv.org/abs/2405.12399 · https://github.com/eloialonso/diamond
- Drama: https://arxiv.org/abs/2410.08893 · https://github.com/realwenlongwang/Drama
- V-JEPA 2: https://arxiv.org/abs/2506.09985 · https://github.com/facebookresearch/vjepa2
- Decision Transformer: https://arxiv.org/abs/2106.01345 · https://github.com/kzl/decision-transformer
- LRAM (xLSTM): https://arxiv.org/abs/2410.22391 · https://github.com/NX-AI/xlstm
- RND: https://arxiv.org/abs/1810.12894 · NovelD: https://papers.neurips.cc/paper_files/paper/2021/file/d428d070622e0f4363fceae11f4a3576-Paper.pdf · BeBold: https://arxiv.org/abs/2012.08621 · E3B: https://arxiv.org/abs/2210.05805 · https://github.com/facebookresearch/e3b
- Go-Explore: https://arxiv.org/abs/2004.12919 · https://github.com/uber-research/go-explore
- PufferLib: https://arxiv.org/abs/2406.12905 · https://github.com/PufferAI/PufferLib · https://github.com/drubinstein/pokemonred_puffer · https://drubinstein.github.io/pokerl/docs/chapter-4/results
- Pokemon Red via RL: https://arxiv.org/abs/2502.19920 · PokeRL: https://arxiv.org/abs/2604.10812
- G1: https://arxiv.org/abs/2505.13426 · GiGPO: https://arxiv.org/abs/2505.10978 · Odysseus: https://arxiv.org/abs/2605.00347 · RL & VLM perception: https://arxiv.org/abs/2610.23983
