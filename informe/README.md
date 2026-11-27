# Informe y presentación IPD434 (ARES-U)

`main.tex` es el informe (IEEEtran `conference`, máximo 5 páginas) y `slides.tex` la presentación (Beamer 16:9, 15 min).

Ningún número de resultado se escribe a mano. `generated/numbers.tex`, las tablas `generated/tab_*.tex`, la bibliografía del informe (`generated/refs.bib`) y las figuras de `figs/` se generan con `scripts/ares_u_report_assets.py` a partir de los resultados de los protocolos.

## Reproducir (Windows, desde la raíz del repo, venv del proyecto)

Requisitos: las ROMs propias del usuario en la carpeta padre del repositorio (ver `ares_u/gal/games.py`), que nunca se distribuyen, y el entorno instalado con `.\ARES.ps1 setup`.

**E1, combate entre generaciones.** 5 métodos × 5 semillas, $6\cdot10^6$ pasos cada uno, en una GPU AMD vía DirectML (~6 h).

```powershell
.\.venv\Scripts\python.exe -m scripts.ares_u_battle protocol --run final --steps 6000000 --device dml --workers 10 --battles 500 --methods bc ppo bc_ppo ppo_ids dqn --seeds 0 1 2 3 4 --demo-steps 300000
```

Reevaluación combate a combate y estadística pareada (familia pre-registrada H1-H4):

```powershell
.\.venv\Scripts\python.exe -m ares_u.battle.paired evaluate --run artifacts/ares_u/battle/final --battles 500 --workers 12
.\.venv\Scripts\python.exe -m scripts.ares_u_battle summarize --run artifacts/ares_u/battle/final
```

**E2, exploración desde píxeles FireRed → Emerald.** 3 semillas (~5 h).

```powershell
.\.venv\Scripts\python.exe -m scripts.ares_u_nav protocol --out-dir artifacts/ares_u/nav/final --seeds 0 1 2 --steps-fr 100000 --steps-em 100000 --collect-steps 1500 --ae-epochs 10 --eval-episodes 8 --fr-eval-episodes 4 --num-envs 4
```

**E3, el mismo agente en ocho juegos (GB, GBC, GBA, NDS).** Primero se compila el shim multinúcleo de mGBA (GB/GBC/GBA) y se instala py-desmume (NDS):

```powershell
cmake -S native/multi -B native/build_multi -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build native/build_multi
uv pip install --python .venv\Scripts\python.exe py-desmume==0.0.9
```

Estados iniciales: el conductor de intros genérico guarda estados y una hoja de contacto. Se elige por inspección visual uno en que el jugador ya controla al personaje.

```powershell
.\.venv\Scripts\python.exe -m scripts.ares_u_gal intro --game rojo --frames 18000 --every 600
.\.venv\Scripts\python.exe -m scripts.ares_u_gal pick --game rojo --state artifacts/ares_u/gal/intro/rojo/f010200.state
```

Adaptadores de medición (solo para medir; el agente ve píxeles):

```powershell
.\.venv\Scripts\python.exe -m scripts.ares_u_gal discover --game rojo --start 0xC000 --size 0x2000 --width 1
.\.venv\Scripts\python.exe -m scripts.ares_u_gal adopt --game rojo --x 0xD367 --y 0xD366 --map 0xD363 --width 1
.\.venv\Scripts\python.exe -m scripts.ares_u_gal verify --game rojo
```

Protocolo por juego (aleatorio, zero-shot, ajuste, desde cero y encoder; 3 semillas) y montaje de capturas:

```powershell
.\.venv\Scripts\python.exe -m scripts.ares_u_e3 protocol --games rojo cristal --seeds 0 1 2 --steps 30000
.\.venv\Scripts\python.exe -m scripts.ares_u_e3 montage
```

**Recursos del informe, compilación y entrega:**

```powershell
.\.venv\Scripts\python.exe -m scripts.ares_u_report_assets
cd informe; latexmk -pdf main.tex; latexmk -pdf slides.tex; cd ..
.\.venv\Scripts\python.exe -m scripts.ares_u_package
```

Los protocolos son reanudables: si se interrumpen, el mismo comando salta las fases terminadas.
