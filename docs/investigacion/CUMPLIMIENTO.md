# Cumplimiento: lo pedido frente a lo entregado

Auditoría honesta del 27-nov-2026. Estados: **Cumplido**, **Parcial** (cumplido en parte, lo que falta se explica) y **No cumplido**. Cada fila indica dónde está la evidencia.

## 1. Pedido original del proyecto

| # | Pedido | Estado | Evidencia y comentario |
|---|---|---|---|
| 1 | Revisar el estado actual de ARES: cómo está configurado y cómo funciona | **Cumplido** | `ESTADO_REPOSITORIO.md`, `notas/repo_*.md`. Tests 451/1 skip, mypy con 1 error. Campañas: máximo ~3 medallas; sesión actual 1/8 (FR) y 2/8 (EM). La observación es solo RAM Gen 3, hay 101 hitos escritos a mano y las redes son decorativas (peso 0.05). |
| 2 | Usar el material de IPD434 para presentarlo como proyecto del curso | **Cumplido** | `notas/course_and_local_docs.md` (tabla concepto → clase → uso). Informe §III (unidades 05-1/05-2/05-3/05-4) y diapositivas con la unidad en cada componente. La unidad 02 (difusos) queda solo como propuesta opcional [LP]. |
| 3 | Redes neuronales avanzadas, "ultra modernas y aplicadas" | **Cumplido** (con matiz) | Transformer de entidades con atención de puntero y máscaras, PPO por self-play con liga, warm-start por imitación, autoencoder auto-supervisado, PPO+RND con novedad episódica, entrenamiento en GPU AMD vía DirectML. Las piezas más recientes (world models, JEPA, SSM/xLSTM, acciones latentes) están justificadas en la arquitectura objetivo, pero no se entrenaron en este plazo. |
| 4a | **Poder poner** el agente en cualquier Pokémon (Rojo, Rojo Fuego, Esmeralda, HeartGold, Platino…) | **Cumplido** (ejecutar y explorar) | **E3** y la capa de abstracción `ares_u/gal`: el mismo agente de píxeles, sin cambios de código ni de arquitectura, corre en **8 juegos de 5 generaciones y 4 consolas**: Rojo (GB), Cristal (GBC), Rojo Fuego y Esmeralda (GBA), Platino y HeartGold (NDS, Gen 4), Negro y Negro 2 (NDS, Gen 5). Backends: mGBA compilado con los núcleos GB y GBA (`native/multi`) y DeSmuME (py-desmume). Los estados iniciales se obtienen con un conductor de intros genérico y los adaptadores de medición se descubren automáticamente (`scripts/ares_u_gal.py`). |
| 4b | Que **termine el juego** con eficiencia y eficacia | **No cumplido** (largo plazo) | Ningún sistema publicado lo logra sin scripts ni harness (`ESTADO_DEL_ARTE.md`). El agente explora, pero no persigue la historia: falta una política jerárquica con subobjetivos y memoria. Plan en `VICIOS_Y_SOLUCIONES.md`, Parte IV. |
| 5 | Que aprenda los patrones comunes a todos los juegos | **Parcial (con evidencia en 3 ejes)** | E1: transferencia de combate Gen1 → Gen2/Gen3 (especies nunca vistas) frente a una ablación con IDs. E2: transferencia FireRed → Emerald. E3: transferencia **entre consolas y generaciones** (GBA → GB/GBC/NDS) sin reentrenar, frente al aleatorio. Hallazgo crítico: ajustar la política completa pierde plasticidad (entropía) y queda por debajo de entrenar desde cero; transferir solo el encoder visual se evalúa como remedio. No cubre cambios de mecánica de combate de Gen 4+. |
| 6 | Investigar **todos** los agentes, videos, repositorios y papers de agentes que juegan Pokémon | **Parcial → en curso** | Papers y repositorios: unos 72 ítems con bibliografía verificada contra arXiv/GitHub (`referencias.bib`, `ESTADO_DEL_ARTE.md`). Videos, streams y comunidad: catálogo dedicado en `VIDEOS_Y_COMUNIDAD.md`, que se agregó tras detectar este hueco. "Todos" no es verificable: la cobertura se declara con su fecha. |
| 7 | Revisar sus limitaciones, falencias, errores y vicios | **Cumplido** | `VICIOS_Y_SOLUCIONES.md`: 17 vicios del estado del arte y 6 propios de ARES, cada uno con evidencia, fuente y severidad. |
| 8 | "Resuélvelos": dejar claro cómo ahora se resuelven y se superan | **Parcial** | Cada vicio tiene un mecanismo concreto y la forma de verificarlo. **Verificados con experimentos**: generalización (V7), memorización de IDs (V15), harness/RAM (V1/V2 en E2), recompensa por juego (V3), *panic switching* (V10), N=1 y métricas incomparables (V12/V13), redes decorativas (A2), tests como evidencia (A4). **Solo diseñados [LP]**: contexto/memoria de LLM (V17), desajuste Showdown/cartucho (V11), percepción de texto (V16), swarming de savestates (V6), entre otros. |
| 9 | Usar subagentes | **Cumplido** | Flujos multiagente de auditoría, investigación, revisión adversarial del diseño y verificación bibliográfica. Varios se cortaron por límite de uso y se relanzaron. |
| 10 | Usar la RX 6700 XT, la RAM y la CPU al máximo | **Cumplido** | Se descubrió y documentó por qué DirectML "fallaba": un assert de autograd si antes se ejecuta un backward en CPU. Ahora entrena ~6× más rápido que la CPU con pérdidas idénticas. Durante los protocolos: GPU ~100%, RAM ~17/24 GB; la CPU no es el cuello de botella. |

## 2. Lineamientos del curso (`instrucciones_proyecto.pdf`)

| Requisito | Estado | Evidencia |
|---|---|---|
| Informe IEEEtran `conference`, máximo 5 páginas con referencias | **Cumplido** (borrador; los números finales están pendientes) | `informe/main.tex` compila con MiKTeX en ~4.5 páginas |
| Nombre completo y rol USM en la primera página | **Cumplido** | Cristóbal Moraga, 202130008-2 |
| Enlace a un repositorio público | **Parcial** | El enlace está en el informe y el repo es público, pero la rama `ipd434` **no se ha subido**: falta la autorización del usuario para hacer push |
| Problema, técnica y justificación, protocolo, resultados contra línea base | **Cumplido** (resultados al terminar las corridas) | Líneas base: Greedy, prior experto de ARES, aleatorio, DQN (curso), BC y la ablación con IDs |
| Variabilidad si el método es estocástico | **Cumplido** | 5 semillas (E1) y 3 (E2); media ± d.e. entre semillas; IC pareados; Holm sobre hipótesis pre-registradas |
| Código, datos o instrucciones en un único comprimido | **Pendiente** | `scripts/ares_u_package.py` lo genera al cerrar las corridas (sin ROMs) |
| Presentación de máximo 15 minutos con diapositivas en PDF | **Cumplido** (borrador) | `informe/slides.tex`, 13 láminas |

## 3. Errores propios detectados en la auditoría y ya corregidos

1. El informe decía que la acción de E2 se "mantiene 24 cuadros". En realidad las direcciones se sostienen 24 y los botones 6. Corregido.
2. El techo de 0.62 se presentaba como si fuera alcanzable por cualquier política. Es un planificador con información completa que conoce al rival, así que es una estimación optimista y no una cota. Corregido.
3. "Gen1-H = 4096 conjuntos" era impreciso: es un banco de hasta 8192 conjuntos que el entrenamiento rechaza. Corregido.
4. La tabla de vicios prometía medir *panic switching* y cómputo, pero el informe no lo mostraba. Se agregó una tabla anti-vicio generada desde los resultados.
5. La tabla principal mostraba intervalos de combates en vez de variabilidad entre semillas, que es lo que exige la pauta. Ahora muestra media ± d.e. entre semillas.
6. La atribución del scaffold `llm_pokemon_scaffold` a J. Bradshaw era incorrecta; el autor es anónimo (cicero225). Corregido.
7. La revisión adversarial del diseño encontró varios problemas. Todos se corrigieron antes de las corridas finales:
   - El test Mann-Whitney sobre 5 semillas con Holm sobre 12 tests no podía ser significativo.
   - Había fuga de PS de los miembros ocultos del rival.
   - Hidden Power y Weather Ball se codificaban mal.
   - La ablación con IDs alimentaba vectores aleatorios.
   - Las curvas de validación usaban combates del test.

## 4. Limitaciones que se declaran y no se ocultan

- Nada de lo entregado equivale a terminar un juego.
- En E1 el cambio de dominio es de especies; los movimientos y la mecánica de Gen 3 son compartidos.
- Los hiperparámetros de E1 se fijaron en pilotos que usaron combates de evaluación. Ningún checkpoint final se eligió sobre el test.
- E2 mide cobertura, no historia, y usa pocas semillas.
- La campaña nativa de ARES-PKMN sigue usando su lógica anterior. Integrar la política de E1 en sus combates es el siguiente paso natural.
