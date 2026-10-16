# Campaña, visor y entrenamiento en vivo

Se recomienda preparar primero el entorno y, si se quieren usar los pesos recién entrenados, completar el entrenamiento antes de arrancar la campaña. Usa ventanas separadas: el visor es gráfico y los logs de optimización aparecen en la consola del entrenamiento.

## A. Entrenar y conservar métricas

En PowerShell, desde la raíz del repo:

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
.\.venv\Scripts\python.exe -u -m scripts.train_neural_modules 2>&1 |
  Tee-Object -FilePath artifacts\training-live.log
```

La consola emite líneas JSON por iteraciones de VICReg/Trajectory Balance y por ciclo PPO. El archivo conserva lo mostrado. Al final se escribe `artifacts\checkpoints\neural.pt`, que una partida nueva carga al iniciar. `resume` restaura los pesos del checkpoint de campaña para conservar su estado cognitivo. Para ejecutar sin guardar log, usa `.\ARES.ps1 train`.

Entrenamiento de fondo opcional:

```powershell
.\ARES.ps1 train -Background
$proc = Get-Content artifacts\processes\latest.json | ConvertFrom-Json
$proc.stdout
Get-Content -LiteralPath $proc.stdout -Wait
```

Usa el log de stdout que `latest.json` señala. Para ver el código de salida y errores al terminar, lee también `$proc.stderr`; un proceso terminado no implica éxito si el log informa una excepción.

Abre `.\ARES.ps1 dashboard` y selecciona **Entrenamiento**, o visita `http://127.0.0.1:8765/#training`. Verás etapa, dispositivo, actualizaciones, estado real del proceso y curvas de VICReg, Trajectory Balance, tasa de victorias PPO y pérdida de valor. Se leen `artifacts/training/live.json` y `metrics.jsonl`, con un `run_id` por sesión; no se mezclan curvas de entrenamientos anteriores. El resumen del último checkpoint se identifica aparte. Un proceso muerto o sin publicaciones recientes se muestra como tal. Entrenar en paralelo no cambia automáticamente los pesos de los workers ya abiertos.

## B. Iniciar los juegos

En otra ventana, desde la raíz del clon:

```powershell
.\ARES.ps1 run -Game both -Background
```

`run` empieza partidas nuevas en ambos cartuchos. El iniciador asigna un proceso independiente a cada juego, y devuelve sus PID, ejecutable y rutas stdout/stderr. Estado del último lanzamiento:

```powershell
Get-Content artifacts\processes\latest.json | ConvertFrom-Json | Format-List
```

Para empezar solo un juego, cambia `both` por `firered` o `emerald`. Para dejar una prueba limitada en primer plano, agrega `-MaxFrames 20000`; al llegar al límite, el estado se guarda y el programa devuelve código 2 porque la campaña aún no terminó.

## C. Estación de monitoreo web (recomendada)

Con los workers en marcha, abre la estación completa en el navegador:

```powershell
.\ARES.ps1 dashboard
```

Se abre `http://127.0.0.1:8765/` (cambia el puerto con `-Port 8800`; en segundo plano usa `-Background` y la URL queda en `artifacts\processes\dashboard.json`). El servidor usa solo la biblioteca estándar de Python, escucha únicamente en `127.0.0.1` y **nunca** abre un emulador ni escribe estado de juego: lee los archivos que publican los workers.

Qué muestra, por juego:

- **Resumen** (ambos juegos lado a lado): pantalla en vivo, estado del proceso (en ejecución, reiniciando, sin respuesta, terminado), fotogramas, velocidad en fps, tiempo de juego, latido del worker, medallas, Pokédex, combates y dinero, objetivo actual con su motivo, equipo resumido y la salud del agente.
- **Detalle** (pestañas FireRed / Emerald, o `#firered` / `#emerald` en la URL):
  - Equipo completo: nombre, nivel, PS, estado alterado, tipos, barra de experiencia y los cuatro movimientos con PP.
  - Combate en curso: Pokémon propio contra el rival (PS visibles) y la última decisión con su riesgo estimado.
  - Salud del agente: escena, menú, fotogramas sin moverse, sin progreso local y **sin avance de historia**, la escalera de recuperación, fallos contenidos, ruta fijada y salidas vetadas. El reloj de historia no se reinicia por subir niveles, curarse ni repetir terreno después de un rewind; los nuevos máximos de nivel sí le descuentan tiempo, con un presupuesto acotado.
  - **Aprendizaje y estrategia**: la estrategia actual (por ejemplo «Restaurar PP antes de GYM_2_BRAWLY» o «Plazo estratégico: intentar con la preparación actual»), las lecciones por objetivo (derrotas, cuántas fueron sin curar, nivel objetivo) y los **pasos denegados aprendidos** con su requisito (`FLAG_GOT_TEA`, `ITEM_GO_GOGGLES`...) y si se evitan o ya pueden reintentarse porque el mundo cambió.
  - Hitos de campaña: la lista completa del DAG con los completados y el siguiente.
  - Decisiones de combate: tabla de las últimas decisiones (Pokémon propio y rival, movimiento, uso de pociones, riesgo, turnos estimados hasta el KO y estado de la prueba formal).
  - Línea de tiempo filtrable (combate, historia, equipo, recuperación, aprendizaje, sistema): evoluciones, subidas de nivel, movimientos aprendidos, curaciones, atascos, retrocesos, fallos, checkpoints, pasos denegados, requisitos de paso («antes de pasar hace falta…»), cambios de estrategia y compras en tiendas.
  - Gráficos temporales: niveles del equipo, combate acumulado, progreso de historia (hitos, medallas, Pokédex), salud del agente (recuperaciones, fallos, retrocesos), velocidad de emulación y tiempo sin progreso.
  - Checkpoints recientes y eventos del supervisor (inicios, reinicios con su espera y semilla).
- Botones **Guardar checkpoint ahora** y **Detener ordenadamente**: crean las mismas solicitudes que `--checkpoint` y `--stop`.

Para otra raíz de checkpoints: `.\.venv\Scripts\python.exe -m scripts.dashboard --checkpoint-root artifacts\mi_campana`.

## C2. Visor de escritorio

En una tercera ventana (o la misma ventana una vez lanzados los workers):

```powershell
.\ARES.ps1 watch
```

El visor abre los dos framebuffers capturados directamente desde mGBA y muestra frames, hitos, mapa, posición, menú, el equipo con nombres, niveles y barras de PS, la salud del agente (escena, intentos de recuperación, retrocesos, fallos), victorias/derrotas/desmayos/capturas, decisiones demostradas o no demostradas, checkpoints y últimos eventos. Redimensiona la ventana si hace falta. `Esc` cierra solo el visor; para parar los workers usa el comando de parada de la sección F.

Para guardar una captura de toda la ventana cada dos segundos (se reemplaza el mismo PNG):

```powershell
.\ARES.ps1 watch -SnapshotPath artifacts\live_viewer.png
```

La GPU para PyTorch no cambia la salida visual: la ventana muestra la imagen real publicada por la DLL de mGBA. Si una campaña no corre aún, verás “Esperando el emulador”. El visor normal espera campañas bajo `artifacts\campaigns\native\<juego>`.

### Usar otra raíz de checkpoints

La interfaz `ARES.ps1` usa la raíz por defecto. Si arrancas directamente con `--checkpoint-root` personalizado, usa la misma ruta para el visor:

```powershell
.\.venv\Scripts\python.exe -m scripts.watch_campaign --checkpoint-root artifacts\native_probe_campaign
```

El `--checkpoint-root` pasado al visor es la raíz que contiene `native` y/o `synthetic`; debe ser exactamente la misma raíz que recibe `run_campaign`. Si se quiere ejecutar un juego con raíz personalizada, llama también `scripts.run_campaign` directamente:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_campaign --backend native --game firered --resume --checkpoint-root artifacts\mi_campana
```

En ese caso el estado queda en `artifacts\mi_campana\native\firered`; el visor recibe `--checkpoint-root artifacts\mi_campana`.

## D. Estado, eventos y evidencia en disco

Con la raíz predeterminada:

- `artifacts\campaigns\native\firered\live.rgb`, `live.json`: último framebuffer RGB y estado (incluye `telemetry`: equipo, tarjeta de entrenador, combate, última decisión, recuperación) que consumen el visor y la estación.
- `status.json`, `events.jsonl`: estado persistido y eventos de la campaña (cada evento lleva fotograma y hora `t`). Los eventos repetitivos se agrupan con `suppressed_repeats`.
- `telemetry.jsonl`: una muestra cada 5 s (fps, combates, hitos, medallas, Pokédex, dinero, niveles y PS del equipo, recuperaciones, fallos, retrocesos). Alimenta los gráficos.
- Rotación: `events.jsonl` se rota al superar 64 MB y `telemetry.jsonl` al superar 32 MB; se conserva una generación anterior (`*.1.jsonl`).
- `frame-*\`: checkpoints. Se conservan los 24 más recientes, los de hitos/inicio/completado y cualquiera que contenga un archivo `PINNED`; los demás se borran para no llenar el disco.
- `supervisor-events.jsonl`: inicios, reinicios (con espera y semilla), pausas `recovery_cooldown` y finales de cada worker.
- `lessons.json`: lecciones duraderas del agente (pasos denegados con el mundo en que ocurrieron, inmunidades observadas y derrotas por objetivo). Se conservan entre retrocesos y reinicios; borrarlo solo hace que el agente vuelva a aprenderlas.
- `latest.png` y archivos `.state`: última imagen/savestate local.
- `artifacts\processes\latest.json`: PID y logs del proceso iniciado más recientemente.
- `artifacts\training-live.log`: log del ejemplo de entrenamiento visible.

`live.rgb` y `live.json` son archivos de transporte de lectura; no los edites. Si cambian a intervalos, el contador de frames debe avanzar. Si se detienen, revisa `status.json`, stdout y stderr. El comando `ARES.ps1 status` contrasta el PID y su fecha de creación con el sistema operativo: distingue un proceso activo, detenido o sin respuesta. Cerrar el visor no detiene la campaña.

## E. Reanudar y checkpoint solicitado

Al iniciar una máquina nueva o tras una interrupción, comprueba primero que existan checkpoints. Luego:

```powershell
.\ARES.ps1 status -Game both
.\ARES.ps1 resume -Game both -Background
```

Para guardar inmediatamente sin detener:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_campaign --backend native --game both --checkpoint
```

Para guardar y detener ordenadamente:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_campaign --backend native --game both --stop
```

Las solicitudes se procesan por los workers cada ciclo de publicación. Espera a que los procesos terminen y revisa stdout/status antes de reiniciar. `resume` carga el checkpoint más reciente de cada juego; no uses `run` si deseas preservar la partida.

## F. Proceso de entrenamiento y campaña simultáneos

Se puede ejecutar entrenamiento y emulación a la vez, pero para usar los pesos recién entrenados, la secuencia reproducible es: entrenar → verificar `neural.pt` → iniciar la campaña → abrir el visor. Si comienzas el entrenamiento después, los workers ya inicializados no recargan automáticamente pesos a mitad de campaña. Mantener los dos procesos pesados simultáneos puede reducir velocidad en CPU; en el equipo de referencia se usó CPU.

## G. Qué significa “en vivo” en esta versión

La estación web y el visor muestran frames, eventos, decisiones y métricas de campaña; las métricas de **entrenamiento neuronal** (pérdidas de VICReg/Trajectory Balance, PPO) siguen en su propia consola/archivo. Un laboratorio sintético puede verse con `.\ARES.ps1 watch -Backend synthetic` o `.\ARES.ps1 dashboard -Backend synthetic`; el encabezado lo identifica como sintético y no es una ROM real.
