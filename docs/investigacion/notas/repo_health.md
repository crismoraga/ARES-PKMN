# Salud del repositorio (live health) — auditoría 2026-11-24

Auditoría de solo lectura. No se modificó código ni se lanzaron campañas.

## 1. Git y dependencias

- HEAD `1031f3a WIP`; commits previos: `cbf6eff` (run_stadium + tests), `2674993` (tests de recovery/monitor).
- Working tree limpio salvo `?? docs/investigacion/` (notas de esta investigación).
- `pyproject.toml:7` version 3.0.0; `:9` `requires-python >=3.12,<3.13`; `:10` deps `numpy==2.0.2, torch==2.4.1, scipy, PyYAML, z3-solver, pygame`; `:14` extra `directml = torch-directml==0.2.5.dev240914, torchvision 0.19.1`.
- `requirements.lock:30` `torch==2.4.1+cpu`, `:31` `torch-directml==0.2.5.dev240914`.

## 2. Hardware / GPU

- `torch 2.4.1+cpu`, `torch.cuda.is_available() == False`. `nvidia-smi` no existe.
- GPU: **AMD Radeon RX 6700 XT** (Win32_VideoController; AdapterRAM reporta 4 GB por límite de 32 bits de WMI, real 12 GB). Sin CUDA; solo vía DirectML (torch-directml 0.2.5, op coverage parcial) o CPU.
- CPU Ryzen 7 5700X3D 8C/16T, RAM 23.9 GB.
- `artifacts/checkpoints/training_metrics.json`: `"backend": "cpu", "backend_reason": "CPU explicitly selected"` — el último entrenamiento corrió en CPU.
- Implicación para el curso: entrenamiento de NN debe ser pequeño (CPU o DirectML). Nada de modelos grandes en 1 semana.

## 3. Tests y tipado

- Suite: 27 entradas en `tests/`, ~285 funciones `def test_`.
- pytest `-q -x`: **451 passed, 1 skipped en 256.9 s (4:16)**. Skip: `tests/test_navigation_recovery.py:84` "Campaign has moved beyond the recorded east MtMoon B2F section" → algunos tests dependen del estado vivo de `artifacts/` (acoplamiento test↔artefactos; no reproducible en un clon limpio). Warning menor de argparse con `-p no:cacheprovider` duplicado (pytest config ya lo incluye).
- mypy `--strict core scripts tests`: **1 error en 99 archivos** — `tests/test_stadium.py:69: Argument 1 to "RomCatalog" has incompatible type "str"; expected "Literal['firered', 'emerald']"`. (Además nota: secciones unused en pyproject para `torch_directml.*`, `z3.*`). Corrió en 2 s (caché de mypy).

## 4. Estado real de las campañas (artifacts/campaigns/native)

**Las campañas NO están corriendo.** Último evento/heartbeat: epoch 1794890152 = **2026-11-17 01:35** (7 días atrás). No hay ningún proceso python de ARES vivo (solo pytest de esta auditoría, MCPs, VS Code). Sin embargo `status.json` sigue diciendo `"running": true` (firered pid 22980, emerald pid 16912) y heartbeat `phase: running` → **estado obsoleto/stale**: el apagado no limpió la bandera (posible bug de shutdown o el proceso murió abruptamente).
`artifacts/processes/latest.json`: última acción `run` (no `resume`) `game: both`, pid 9720, log `run-both-20261117-011155-600` → la sesión del 17-nov **empezó partida nueva desde cero** (supervisor `worker_started ... "checkpoint": null`).

### FireRed (`firered/status.json`)
- frames 200 000 (heartbeat 202 751). Milestones completados: OAK_ROUTE1_TRIGGER, STARTER_BULBASAUR, RIVAL_LAB, OAK_PARCEL_PICKUP, OAK_PARCEL, GYM_1_BROCK. **Medallas: 1/8** (`native_completion_evidence.badges`).
- Directiva actual: MT_MOON_FOSSIL_GUARD (Miguel, MtMoon_B2F). Party: Ivysaur(2) Lv19 + 4 Pokémon nivel 3-8 (sobre-dependencia de un solo Pokémon).
- Métricas: battles 102, wins 77, losses 0, captures 5, faints 1, `neural_decisions 194`, `proved_decisions 0`, `unproven_decisions 194`.
- Supervisor (`supervisor-events.jsonl`, histórico total): 21 worker_started, 15 worker_terminated, 13 worker_restart, 2 recovery_blocked. Causas: `CampaignStalled` repetido en MT_MOON_FOSSIL_GUARD (≥11 veces), CERULEAN_RIVAL (≥10), CELADON_TEA (7). Último restart (17-nov): `CampaignStalled: No gameplay progress after 4 recoveries and 1 rewinds: animation; MT_MOON_FOSSIL_GUARD`.
- Histórico de events.jsonl (116 561 líneas, 33.5 MB, incluye varias sesiones): stall_detected 338, stall_recovery 310, campaign_blocked 2 367 (3 902 menciones "Strongest campaign Pokemon has low HP", 357 "live collision path unavailable"), oscillation_detected 40, transition_tabu 226, memory_snapshot_retry 5 123. `agent_fault`: 0.
- Mejor progreso histórico (milestone_observed en sesiones de octubre, frames hasta ~2.7M): GYM_2_MISTY, BILL_RESCUE, SS_TICKET, SS_ANNE_CUT, GYM_3_SURGE, SURGE_SWITCHES → **máximo histórico 3 medallas** en FireRed; luego bloqueado en CELADON_TEA.

### Emerald (`emerald/status.json`)
- frames 580 000 (heartbeat 582 200). 20 milestones incluyendo GYM_1_ROXANNE, GYM_2_BRAWLY, HM_CUT, STERN_GOODS, RIVAL_ROUTE110, WALLY_MAUVILLE. **Medallas: 2/8** (recovery-progress facts `badge:0`, `badge:1`). Directiva actual GYM_3_WATTSON (puzzle `mauville_switches`).
- Métricas: battles 313, wins 284, losses 1, captures 7, faints 15, neural_decisions 495, proved 0, recoveries 1.
- Supervisor histórico: 15 started, 10 terminated, 9 restarts, 1 recovery_blocked; causa dominante `CampaignStalled ... STERN_GOODS` (≥17), también GYM_2_BRAWLY.
- events.jsonl (124 652 líneas): stall_detected 720, stall_recovery 649 (reasons: stationary 375, **repeated_state_cycle 301**), campaign_blocked 2 911 (1 985 "live collision path unavailable", 1 964 en PETALBURG_WOODS_AQUA), oscillation_detected 63. `agent_fault`: 0.
- Mejor histórico: GYM_3_WATTSON, HM_ROCK_SMASH, METEOR_FALLS_TEAMS observados una vez → **máximo histórico 3 medallas** en Emerald.

### Lecciones (`lessons.json`)
- FireRed: rival de Celeste (CERULEAN_RIVAL) con **17 derrotas**, `retry_level 30` → muro de dificultad que el agente resuelve grindeando.
- Emerald: GYM_2_BRAWLY 3 derrotas, STEVEN_LETTER 2, PETALBURG_WOODS_AQUA 1.

### ¿Loops?
Sí, hay patrones de bucle recurrentes aunque el sistema los contiene: `repeated_state_cycle` (301 en Emerald), `oscillation_detected` (40/63), y reinicios de supervisor repetidos en el mismo milestone (Mt. Moon, STERN_GOODS). El progreso existe pero es lento y con muchos "atascos" contenidos por watchdog+rewind, no por aprendizaje.

## 5. Docs operativos

- `docs/PROBLEMAS.md` (74 líneas): troubleshooting de instalación (Python 3.12, uv/cmake/ninja/gcc, ROM hash, `ares_mgba.dll`), visor congelado, estación de monitoreo, significado de `stall_detected/agent_fault/stall_rewind` (l.50-52), reinicios del supervisor y presupuesto "sin nueva historia" (`recovery_blocked`, l.54-56), "terminó antes del Campeón ≠ certificación" (l.58-60), fallback DirectML→CPU (l.62-64), stop/resume con checkpoint (l.66+).
- `docs/RECUPERACION_AUTONOMA.md` (145 líneas): catálogo de bugs de bloqueo reproducidos con savestates y corregidos a mano: evolución post-combate, política de olvidar movimientos (`core/battle/move_learning.py`), árboles de Corte fuera de cámara (`core/navigation/obstacles.py`), secciones desconectadas en cuevas (Mt. Moon), interceptar patrulla de Briney, ataques sin efecto, cambio forzado, guardias del té de Azulona, preparación ante jefes (Brawly), escalera de recuperación y límites, reanudación. **Evidencia clave**: el progreso se ha logrado parcheando lógica específica de juego/mapa (ingeniería manual), no generalización aprendida.

## 6. Conclusión honesta

- Tests (≈285) + mypy casi limpio miden **corrección del software**, no capacidad de juego.
- Progreso real: 1 medalla FR / 2 medallas EM en la sesión actual (desde cero, 17-nov); máximo histórico ~3 medallas por juego. Ningún juego completado. Campañas detenidas desde hace 7 días con `running:true` stale.
- `proved_decisions = 0` en ambos juegos; las "neural_decisions" son decisiones de combate; la navegación/historia sigue directivas con `script`, `completion_flag_id`, `map_name` extraídas de la decompilación → altamente específico a FR/EM; no "juega cualquier Pokémon".
- Hardware: sin CUDA; AMD RX 6700 XT solo vía DirectML; último entrenamiento fue CPU.
