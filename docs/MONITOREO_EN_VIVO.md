# Campaña, visor y entrenamiento en vivo

Se recomienda preparar primero el entorno y, si se quieren usar los pesos recién entrenados, completar el entrenamiento antes de arrancar la campaña. Usa ventanas separadas: el visor es gráfico y los logs de optimización aparecen en la consola del entrenamiento.

## A. Entrenar y conservar métricas

En PowerShell, desde la raíz del repo:

```powershell
New-Item -ItemType Directory -Force artifacts | Out-Null
.\.venv\Scripts\python.exe -u -m scripts.train_neural_modules 2>&1 |
  Tee-Object -FilePath artifacts\training-live.log
```

La consola emite líneas JSON por iteraciones de VICReg/Trajectory Balance y por ciclo PPO. El archivo conserva lo mostrado. Al final se escribe `artifacts\checkpoints\neural.pt`, que el orquestador carga al iniciar. Para ejecutar sin guardar log, usa `.ARES.ps1 train`.

Entrenamiento de fondo opcional:

```powershell
.\ARES.ps1 train -Background
$proc = Get-Content artifacts\processes\latest.json | ConvertFrom-Json
$proc.stdout
Get-Content -LiteralPath $proc.stdout -Wait
```

Usa el log de stdout que `latest.json` señala. Para ver el código de salida y errores al terminar, lee también `$proc.stderr`; un proceso terminado no implica éxito si el log informa una excepción.

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

## C. Mostrar el emulador y telemetría

En una tercera ventana (o la misma ventana una vez lanzados los workers):

```powershell
.\ARES.ps1 watch
```

El visor abre los dos framebuffers capturados directamente desde mGBA y muestra frames, hitos, mapa, posición, menú, equipo, victorias/derrotas/desmayos/capturas, decisiones demostradas o no demostradas, checkpoints y últimos eventos. Redimensiona la ventana si hace falta. `Esc` cierra solo el visor; para parar los workers usa el comando de parada de la sección F.

Para guardar una captura de toda la ventana cada dos segundos (se reemplaza el mismo PNG):

```powershell
.\ARES.ps1 watch -SnapshotPath artifacts\live_viewer.png
```

La GPU para PyTorch no cambia la salida visual: la ventana muestra la imagen real publicada por la DLL de mGBA. Si una campaña no corre aún, verás “Esperando el emulador”. El visor normal espera campañas bajo `artifacts\campaigns\native\<juego>`.

### Usar otra raíz de checkpoints

La interfaz `ARES.ps1` usa la raíz por defecto. Si arrancas directamente con `--checkpoint-root` personalizado, usa la misma ruta para el visor:

```powershell
.\.venv\Scripts\python.exe -m scripts.watch_campaign --checkpoint-root artifacts\native_probe_campaign\native
```

El `--checkpoint-root` pasado al visor es la carpeta **padre** que contiene `firered` y `emerald` (tal como lo usa el lector). Si se quiere ejecutar un juego con raíz personalizada, llama también `scripts.run_campaign` directamente:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_campaign --backend native --game firered --resume --checkpoint-root artifacts\mi_campana
```

En ese caso el estado queda en `artifacts\mi_campana\native\firered`; el visor recibe `--checkpoint-root artifacts\mi_campana\native`.

## D. Estado, eventos y evidencia en disco

Con la raíz predeterminada:

- `artifacts\campaigns\native\firered\live.rgb`, `live.json`: último framebuffer RGB y estado que consume el visor.
- `status.json`, `events.jsonl`: estado persistido y eventos de la campaña.
- `checkpoints\`: checkpoints periódicos, solicitados y de hitos.
- `latest.png` y archivos `.state`: última imagen/savestate local.
- `artifacts\processes\latest.json`: PID y logs del proceso iniciado más recientemente.
- `artifacts\training-live.log`: log del ejemplo de entrenamiento visible.

`live.rgb` y `live.json` son archivos de transporte de lectura; no los edites. Si cambian a intervalos, el contador de frames debe avanzar. Si se detienen, revisa `status.json`, stdout y stderr. `.ARES.ps1 status -Game both` muestra el estado resumido guardado; no abre el emulador.

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

La vista combina dos flujos visibles: frames y eventos/estadísticas de la campaña en la ventana, y actualizaciones de entrenamiento en una consola/archivo distinto. El visor todavía no agrega gráficas de pérdida de entrenamiento dentro de la ventana de emulación. Un laboratorio sintético puede verse con `.ARES.ps1 watch -Backend synthetic`, pero el encabezado lo identifica como sintético y no es una ROM real.
