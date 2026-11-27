# ARES-U — Plan del proyecto IPD434 (2026-2)

Proyecto individual del curso IPD434. Los plazos son: presentación el 1 o el 3 de diciembre de 2026 (15 min) y entrega del informe IEEE (5 páginas, `IEEEtran` conference) el 4 de diciembre. El informe exige un problema, la técnica y su justificación, un protocolo reproducible, la comparación con **líneas base** y la **variabilidad entre semillas**.

## Visión y alcance honesto

La visión es un agente neuronal que juegue *cualquier* Pokémon aprendiendo los patrones comunes a todas las generaciones. En una semana no se puede demostrar que termine juegos completos, y ningún trabajo publicado lo ha logrado sin scripts ni harness (ver `ESTADO_DEL_ARTE.md`). Por eso el proyecto demuestra con rigor **la pieza que habilita la visión: representaciones independientes del juego que transfieren a contenido nunca visto**. Lo hace en dos experimentos:

1. **E1 (núcleo): ARES-Battle, transferencia entre generaciones en combate.** Un Transformer de entidades, sin IDs de especie ni de movimiento, se entrena por self-play con PPO usando solo especies de Gen 1 (#1-151). Se evalúa sin reentrenar (zero-shot) contra especies de Gen 2 (#152-251) y Gen 3 (#252-386), que nunca vio.
2. **E2 (secundario): ARES-Nav, exploración desde píxeles FireRed → Emerald.** Se usa una política de píxeles con un encoder auto-supervisado (AE) y novedad intrínseca (RND), sin RAM en la observación. Se compara entrenar desde cero en Emerald con transferir el encoder y la política preentrenados en FireRed.

## E1 — Especificación técnica

- **Entorno**: `core/stadium/engine.py` (motor Gen-3 determinista, ya verificado). Los equipos son aleatorios, de 3v3 a nivel 50, con 4 movimientos legales tomados de `learnsets` (`data/stadium_dex.json`, 412 especies). Cada pool de especies define un split: train = Gen1, test = Gen1 hold-out (equipos nuevos), Gen2 y Gen3.
- **Observación agnóstica**: tokens de entidad por Pokémon (tipos one-hot, stats base normalizados, fracción de HP, estado, boosts, activo, propio o rival) y tokens de acción por movimiento (tipo, potencia, precisión, categoría, PP, efectividad contra el rival activo, STAB). **Sin IDs de especie ni de movimiento**, porque eso es lo que hace posible transferir. Las acciones ilegales se enmascaran.
- **Modelo**: un Transformer pre-norm escrito a mano (es compatible con DirectML) con cabezas actor-crítico, entrenado con PPO por self-play y una liga de oponentes (instantáneas pasadas, Greedy y Random).
- **Líneas base**: (a) Random; (b) Greedy máximo daño; (c) el prior experto heurístico existente de ARES (`MLPGenome.prior`); (d) **DQN-MLP** sobre features planas con el mismo presupuesto de pasos (el algoritmo canónico del curso, clase 05-1); (e) ablación: Transformer **con IDs de especie**, para mostrar que memorizar IDs impide transferir.
- **Protocolo**: 5 semillas por método y presupuesto igual de pasos de entorno. La evaluación usa N ≥ 500 combates por par y por split, con lados alternados. Se reporta win-rate con IC 95% (Wilson y bootstrap entre semillas), media ± desv. estándar, Mann-Whitney con Holm, tiempo de pared y hardware.
- **Métricas anti-vicio**: tasa de *panic switching* (cambios consecutivos), turnos por combate, tasa de acciones inútiles (movimiento inmune), brecha de generalización (Gen1 → Gen2/Gen3).
- **Hardware**: RX 6700 XT vía DirectML en el proceso del learner (ver la nota de DirectML más abajo). Los rollouts van en CPU con varios procesos (Ryzen 5700X3D, 16 hilos).

## E2 — Especificación técnica

- **Entorno**: `NativeLibMGBABridge` (`core/bridge/ffi_mgba.py`) en modo crudo, sin orquestador. Arranca desde un savestate fijo después de la intro (con el jugador ya en control) y repite cada acción durante K frames.
- **Espacio de acción universal**: 8 botones más NOOP (a futuro se adapta a GB, GBA y NDS). La observación son píxeles (84×84 en escala de grises con apilado de frames). La RAM se usa **solo para medir** (mapa, x, y), nunca como entrada.
- **Métodos**: Random; PPO+RND desde cero en Emerald; encoder AE preentrenado en frames de FireRed y PPO+RND en Emerald (transferencia).
- **Métrica**: celdas únicas (mapa, x, y) visitadas con un presupuesto fijo de pasos, con 3 semillas.

## Nota DirectML (verificado el 2026-11-24)

torch-directml funciona en la RX 6700 XT, unas 6× más rápido que la CPU, y coincide con la CPU hasta 1e-6, siempre que:

1. No se ejecute ningún `backward()` en CPU dentro del mismo proceso antes del primer `backward()` en DML. Si se hace, salta el assert `device_ready_queues_`.
2. No se use `nn.TransformerEncoderLayer(norm_first=True)`; un bloque escrito a mano sí funciona.
3. Se use `torch.optim.Adam(..., foreach=False)`.

## Cronograma

| Día | Trabajo |
|---|---|
| 24-25 nov | Documentos de estado del arte y del repo; implementación del entorno E1 y las líneas base; benchmark del emulador para E2 |
| 25-27 nov | Entrenamientos E1 (5 semillas × métodos) y prototipo E2 |
| 28-29 nov | Evaluación, figuras y tablas |
| 29-30 nov | Diapositivas (15 min) |
| 1-3 dic | Presentación |
| 3-4 dic | Informe IEEE de 5 páginas, zip y enlace al repositorio |
