# ARES-ADVERSARIAL: agentes Pokémon que coevolucionan y aprenden uno del otro

ARES-ADVERSARIAL enfrenta, de forma continua, a dos linajes de agentes: **ROJO**, con los Pokémon que crió el agente de la campaña de FireRed, y **AZUL**, con los de Emerald. Combaten en un *Stadium* propio, se comparan en tiempo real y cada uno aprende tanto de lo bueno como de lo malo del otro.

La base es *Tournament Informed Adversarial Quality Diversity* (Anne et al., GECCO'26, arXiv:2601.19562), con GAME y la selección de tareas por **Ranking**. Sobre ella se añaden tres canales de aprendizaje complementario entre adversarios y entre la evolución y las redes neuronales profundas.

```powershell
.\ARES.ps1 stadium                       # liga sin fin (Ctrl+C detiene; el estado queda guardado)
.\ARES.ps1 stadium -Continue             # reanuda la última liga donde quedó
.\ARES.ps1 stadium -Quick                # comprobación corta (4 generaciones pequeñas)
.\ARES.ps1 stadium -Background -Workers 14
.\ARES.ps1 dashboard                     # luego abre http://127.0.0.1:8765/stadium
.\.venv\Scripts\python.exe -m scripts.run_stadium --mode compare --replications 5   # réplica del paper
.\.venv\Scripts\python.exe -m scripts.run_stadium --mode teams                      # equipos cosechados
```

## 1. Qué hace en cada generación

1. **Iluminación (MTMB-ME).** El lado activo, que alterna ROJO y AZUL, busca muchas estrategias distintas y buenas contra *N_task* élites fijas del rival. Cada tarea tiene un archivo creciente de *N_cell* celdas (Algoritmo 3 del paper). Los padres salen del archivo completo y el operador de variación es el del paper: ruido gaussiano σ = 0.1 sobre el 30 % de los pesos, más mutaciones legales del equipo.
2. **Selección de tareas por torneo (Ranking).** Todas las élites combaten contra las tareas anteriores. A partir de esa matriz se calcula el vector de *rankings*, normalizado a [-1, 1], y se agrupa con k-means; la élite de cada grupo pasa a ser una tarea del rival. También están implementadas las variantes Pareto (frentes no dominados con *crowding*), Behavior (k-means sobre el comportamiento, el GAME original) y Random. Las evaluaciones del torneo se reutilizan como *bootstrap* de la generación siguiente.
3. **Duelo en vivo.** Los dos campeones juegan al mejor de N y el combate se narra en español en el dashboard.
4. **Lecciones contrafactuales entre adversarios.** El duelo guarda una copia exacta de la batalla en cada decisión. En las posiciones donde los dos agentes elegirían distinto, se simulan las dos ramas con los mismos números aleatorios.
   - Si la jugada del rival era mejor, el agente la imita («lo bueno del otro»).
   - Si la propia era mejor, es el rival quien aprende de ella («lo malo del otro», expuesto).

   Los campeones corregidos vuelven al archivo como candidatos (◆ en el gráfico). Además, cada lado guarda un **banco de lecciones** (las últimas 400) que funciona como **operador de variación lamarckiano**: el 25 % de los descendientes de MTMB-ME se ajusta con una muestra del banco antes de evaluarse. Así lo aprendido del rival moldea toda la población, no solo al campeón.
5. **Aprendices profundos.** La red de combate de la campaña (Transformer espacial + SSM selectivo tipo Mamba, `AresBattlePolicyNetwork`, iniciada desde `artifacts/checkpoints/neural.pt` si existe) pasa por tres fases:
   - imita las decisiones de las élites del archivo (QD → RL);
   - se entrena con PPO recurrente contra las élites del rival;
   - se destila de vuelta en un MLP estudiante que entra al archivo (RL → QD, ▲ en el gráfico).
6. **Intercambio de conocimiento.** Cada lado aprende la tabla de tipos **solo de los mensajes de combate** («súper eficaz», «no afecta», Levitación…) mediante la propagación de restricciones de la campaña (`TypeChartMemory`). Lo aprende únicamente de sus propias batallas de entrenamiento. Después se fusiona lo aprendido por los dos (unión idempotente) y se mide cuántas celdas ganó cada uno gracias al otro.
7. **Medición.**
   - Elo online de todos los agentes y de dos anclas fijas.
   - Progreso absoluto contra un jugador codicioso y uno aleatorio, porque el fitness adversarial es relativo.
   - Las **seis métricas del paper** entre las tareas actuales de ambos lados: tasa de victoria, ELO, robustez, cobertura, pericia y AQD-score.

## 2. Por qué es distinto

- **GAME + Ranking en un dominio real.** El paper usa Pong, gato y ratón, y perseguidores y evasores. Aquí el problema es un juego de información imperfecta con decisiones simultáneas (las estrategias mixtas importan, de ahí la temperatura evolutiva), equipos y estado oculto.
- **Aprendizaje complementario de dos vías.** La diversidad de calidad alimenta a la red profunda por imitación de élites, y la red profunda devuelve al archivo, por destilación, las estrategias que encontró el gradiente. Los dos procesos se corrigen en lugar de competir.
- **Variación lamarckiana adversarial.** El operador del paper (ruido gaussiano) se combina con un ajuste por gradiente hacia las jugadas que el rival demostró mejores, de modo que la evolución hereda lo aprendido del adversario.
- **Lecciones contrafactuales simétricas.** Cada desacuerdo entre adversarios se decide con despliegues desde la posición exacta (*common random numbers*) y enseña siempre a quien jugó peor. El conocimiento fluye en las dos direcciones a la vez.
- **Conocimiento epistémico compartido.** Ningún agente lee la tabla de tipos de la ROM: la deduce de lo que ve y la intercambia con su rival.
- **Equipos reales.** Los equipos no son inventados. Son los Pokémon que criaron los agentes de campaña, leídos de sus checkpoints sin escribir RAM. Se proyectan al nivel del Stadium (50 por defecto) con su línea evolutiva real (`gEvolutionTable`) y sus estadísticas Gen-3 exactas (IV, EV y naturaleza).

## 3. El Stadium (motor de combate)

`core/stadium/engine.py` implementa combates individuales Gen-3 deterministas por semilla:

- cambios y orden de turno (prioridad, velocidad con etapas, parálisis, Nado Rápido, Clorofila);
- precisión y evasión con las tablas nativas, y críticos por etapas;
- sueño, congelación, parálisis, veneno, tóxico, quemadura, confusión y retroceso;
- más de 150 efectos de movimiento, con etapas, pantallas, clima, Púas, Sustituto, Protección, movimientos de dos turnos y de recarga;
- daño residual, bayas y Restos;
- las habilidades que cambian combates (Intimidación, Levitación, Absorbe Agua/Elec., Absorbe Fuego, Superguarda, Electricidad Estática, Sincronía, Rastro, Sequía/Llovizna/Chorro Arena, Cuerpo Puro…).

**El daño nunca se recalcula aparte**: cada golpe usa `GbaIntegerSafetyVerifier.compute_context_damage_reference`, la referencia entera ya verificada contra la implementación ARM7. La tabla de tipos se lee de la ROM en su orden nativo (el orden cambia el redondeo).

**Límites explícitos.** Los efectos no modelados (Metrónomo, Transformación, Relevo…) siguen haciendo su daño si son ataques, o fallan si son de estado. Cada uso queda registrado como `unsupported`. No hay objetos de mochila, igual que en los combates por cable de Gen 3. Patada Baja usa un peso medio porque el Stadium no lee la tabla de pesos.

## 4. Contrato de información

Cada agente ve:

- su equipo completo;
- del rival: especie, nivel, barra de PS, estado, etapas, y los movimientos y habilidades que ya se revelaron en combate.

Las estadísticas del rival se **estiman** con datos públicos (IV 15, sin EV, naturaleza neutra). Un test (`test_observations_never_leak_hidden_opponent_data`) comprueba que cambiar movimientos o estadísticas ocultas del rival no altera ninguna entrada de la política.

## 5. Archivos y dashboard

| Ruta | Contenido |
|---|---|
| `artifacts/stadium/team_bank.json` | Equipos cosechados de los checkpoints de campaña (`--refresh-teams` los relee) |
| `artifacts/stadium/<run>/live.json` | Estado completo en vivo (lo lee `/stadium`) |
| `artifacts/stadium/<run>/history.jsonl` | Una línea por generación: fitness, QD-score, anclas, métricas, Elo, lecciones |
| `artifacts/stadium/<run>/duel-NNNN.json` | Duelo de campeones narrado de cada generación |
| `artifacts/stadium/<run>/state.json`, `learner-*.pt` | Checkpoint para `-Continue` |
| `artifacts/stadium/<run>/deep-*.pt` | Red de combate entrenada en el Stadium (**no** se carga sola en las campañas) |
| `artifacts/stadium/compare-*/report.md` | Comparación de variantes con medianas, cuartiles y Holm-Bonferroni |

El panel `/stadium` muestra:

- los campeones con equipo y movimientos;
- el duelo narrado y reproducido línea a línea;
- las curvas de Elo, la calidad del archivo, el progreso contra las anclas y el conocimiento aprendido e intercambiado;
- el archivo de élites en el espacio de comportamiento, con ejes elegibles y la forma del punto según el origen;
- la matriz del torneo con las seis métricas;
- el feed de lecciones, la tabla de tipos aprendida y la clasificación Elo.

## 6. Robustez

- La escritura de `live.json` reintenta mientras el dashboard tenga el archivo abierto (Windows).
- Una generación que falla se registra, se restaura el último estado guardado y la liga sigue.
- Si muere un proceso evaluador (cierre forzado, falta de memoria), el pool se reconstruye y el lote se repite.
- Los procesos evaluadores no importan Torch: con unos 22 procesos que importaban Torch, el equipo de 24 GB agotó la memoria comprometida. La red profunda vive solo en el proceso principal.
- `-Continue` sin estado previo arranca de cero en la misma carpeta.
- Los procesos de evaluación usan `spawn` y un hilo de Torch cada uno.
- Las campañas no se tocan: el Stadium es un paquete aislado (`core/stadium/`) que nunca escribe RAM ni progreso de partida y que no se importa desde el orquestador de campaña.

## 7. Verificación

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_stadium.py -v
.\.venv\Scripts\python.exe -m mypy --strict core scripts tests
```

Los tests cubren:

- tabla de tipos idéntica al oráculo de la ROM; evoluciones (incluidas las de Wurmple por personalidad y por amistad); identificadores de habilidades iguales al decomp;
- determinismo y copias que no alteran la batalla original; daño dentro de la referencia entera; inmunidades; Forcejeo; Intimidación; Púas;
- estadísticas Gen-3 iguales a las de las partidas reales (las de 13 de los 15 Pokémon cosechados coinciden exactamente; en los otros dos faltan los EV ganados desde la última subida de nivel, como en el juego);
- legalidad de las mutaciones;
- aprendizaje e intercambio de la tabla de tipos; ausencia de fugas de información;
- el MLP experto y el operador de mutación del paper;
- el Algoritmo 3, los vectores de ranking y Pareto;
- el ciclo GAME con las cuatro variantes;
- las seis métricas calculadas a mano;
- las lecciones contrafactuales y el entrenamiento;
- PPO con la red de la campaña;
- la liga completa con publicación y reanudación.

## 8. Resultados medidos (27-28 de octubre de 2026, PC de referencia: 16 núcleos, 24 GB)

### Réplica del experimento del paper en Pokémon

`--mode compare --replications 6 --n-gen 4 --tasks 6 --cells 6 --budget 384 --seed 11` (115 152 combates + torneo inter-variante 144 x 144; informe en `artifacts/stadium/compare-20261027-234444/report.md`). Mediana [Q1, Q3] sobre 6 réplicas; **negrita** = no distinguible del mejor (Mann-Whitney + Holm-Bonferroni, p = 0.05):

| ROJO · FireRed | Ranking | Pareto | Behavior | Random |
|---|---|---|---|---|
| tasa de victoria | 0.888 [0.793, 0.927] | **0.966 [0.960, 0.975]** | 0.446 [0.383, 0.482] | 0.402 [0.284, 0.510] |
| pericia | **0.796 [0.638, 0.805]** | **0.806 [0.795, 0.821]** | 0.467 [0.280, 0.475] | 0.355 [0.174, 0.489] |
| robustez | **0.473** | **0.499** | **0.215** | **0.195** |
| AQD-score | **3.5 [3, 4]** | **75 [3.5, ∞]** | **1** | **1** |

| AZUL · Emerald | Ranking | Pareto | Behavior | Random |
|---|---|---|---|---|
| tasa de victoria | 0.344 [0.337, 0.376] | **0.456 [0.453, 0.474]** | 0.223 [0.205, 0.296] | 0.278 [0.261, 0.290] |
| pericia | **0.114** | **0.123** | 0.070 [0.060, 0.088] | **0.094** |

Conclusiones:

- La tesis central del paper se reproduce en un dominio mucho más rico: **la selección de tareas informada por torneo (Ranking y Pareto) produce élites mucho más fuertes que la selección por comportamiento (GAME original) o aleatoria**. En ROJO se ve en la tasa de victoria y la pericia (diferencias significativas) y en la robustez y el AQD (tendencia). Los conjuntos de Ranking y Pareto necesitan de 3 a ∞ contra-soluciones para ser derrotados, frente a 1 de Behavior y Random.
- A diferencia del paper (Pong, gato y ratón, perseguidores), aquí **Pareto supera a Ranking** en tasa de victoria en ambos lados. Ranking empata con Pareto en pericia y robustez.
- La cobertura no distingue variantes en este dominio (en el paper, Random y Behavior obtenían la mayor cobertura en Pong y perseguidores).
- Con 3 réplicas (`compare-20261027-233724`) la dirección era la misma, pero ninguna diferencia era significativa: hacen falta al menos 5 réplicas para que Mann-Whitney + Holm puedan rechazar.

### Liga en vivo (`run-20261027-232544`, 72 generaciones, 138 739 combates, ~120 combates/s con 10 procesos)

- **Carrera armamentística visible.**
  - El fitness medio del archivo de AZUL contra las élites de ROJO era 0.15 en la generación 2; en la generación 8 su campeón alcanzaba 0.51 y ganó el duelo de campeones 4-1.
  - Después la dinámica se estabiliza en un equilibrio de «Reina Roja»: los dos mejoran a la vez y el fitness relativo queda casi constante (ROJO ~0.61, AZUL ~0.16 de media en el archivo).
  - El problema es asimétrico, como el gato y el ratón del paper: los dos Swampert de AZUL reciben ×4 de las hojas de Venusaur. En 72 duelos ROJO ganó 158 combates y AZUL 26.
- **Estrategias descubiertas sin ser programadas.**
  - AZUL pasó a usar Danza Lluvia con Azumarill (potencia el Agua) y Luz Lunar con Dustox (recuperación).
  - ROJO adoptó Síntesis + Desarrollo + Drenadoras en Venusaur (desgaste y potenciación) y Remolino en Pidgeot y Butterfree (forzar cambios).
- **Aprendizaje entre adversarios.**
  - ROJO aprendió 217 lecciones contrafactuales de AZUL y AZUL aprendió 253 de ROJO.
  - Algunas soluciones de tipo «lección» se convirtieron en campeonas (por ejemplo, la #1933 en la generación 5).
  - Los dos lados dedujeron 60 celdas de la tabla de tipos (todas las que permiten los tipos presentes en sus equipos) sin leer la ROM.
- **Límites honestos.**
  - El progreso absoluto frente al ancla codiciosa no crece en la liga larga: ROJO 0.91 → 0.86, AZUL 0.20 → 0.20. La coevolución optimiza contra el rival, no contra un jugador fijo.
  - El aprendiz profundo de AZUL apenas gana a las élites de ROJO con PPO (5 %); el de ROJO, el 50 %.
  - La red profunda sigue por detrás de los MLP evolucionados: imita a las élites con un 50 % de acierto, y sus estudiantes MLP alcanzan un 83-85 % de fidelidad.
