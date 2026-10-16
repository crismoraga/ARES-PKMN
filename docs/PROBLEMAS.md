# Problemas comunes

## `Python 3.12` no aparece o el instalador rechaza la versión

Instala Python 3.12 x64 y abre una consola nueva. Comprueba `python --version` y `where.exe python`. El venv existente no se cambia de versión automáticamente: si `.venv` se creó con otra versión, muévelo/elíminalo después de comprobar la ruta y vuelve a correr `.ARES.ps1 setup`.

## No se reconoce `uv`, `git`, `cmake`, `ninja` o `gcc`

Instala la herramienta y agrega su carpeta `bin` al `PATH`. Cierra/reabre PowerShell, ejecuta `Get-Command uv,git,cmake,ninja,gcc` y vuelve a iniciar setup. La compilación nativa requiere las cinco herramientas; `-SkipNativeBuild` no es suficiente para correr ROMs reales.

## Setup no puede descargar paquetes o fuentes

Comprueba acceso a Internet/proxy y vuelve a ejecutar `.ARES.ps1 setup`. Las fuentes se fijan en `data/source_manifest.json`. Si una carpeta de `native\vendor` ya existe con un commit diferente, setup falla deliberadamente: conserva cualquier trabajo local y resuelve esa copia según el manifiesto antes de intentarlo otra vez.

## Error “ROM not found”, hash incorrecto o juego no compatible

Coloca la ROM en el directorio padre del checkout y utiliza el nombre exacto de la tabla del README. Revisa el SHA-1 admitido. El programa no parchea ni modifica ROMs. El visor no puede sustituir el requisito de ROM válida.

## `ares_mgba.dll` no existe

Desde la raíz:

```powershell
.\ARES.ps1 setup
Test-Path native\build\ares_mgba.dll
```

Si falla CMake, Ninja o GCC, resuelve la salida completa del build en `artifacts\native_build.log` o consola. Ejecuta `gcc --version`, `cmake --version` y `ninja --version` para confirmar el PATH que ve el mismo PowerShell.

## El visor muestra “Esperando el emulador”

Verifica workers y ubicación de los datos:

```powershell
Get-Content artifacts\processes\latest.json | ConvertFrom-Json | Format-List
Get-Content artifacts\campaigns\native\firered\status.json -Raw
Get-Content artifacts\campaigns\native\emerald\status.json -Raw
```

Confirma que `live.json` y `live.rgb` cambien de fecha/tamaño. Si usaste una raíz personalizada, pasa al visor su carpeta padre con `--checkpoint-root`; se explica en [monitoreo en vivo](MONITOREO_EN_VIVO.md). Si el proceso terminó, abre `stderr` y `stdout` de `latest.json`. Si solo el visor no inicia, valida Pygame desde el venv: `.\.venv\Scripts\python.exe -m scripts.watch_campaign --help`.

## El visor se congela o los frames no avanzan

La ventana gráfica es un lector; congelar su UI no envía botones ni salva la partida. Revisa el timestamp de `live.json`, el PID con `Get-CimInstance Win32_Process`, y los logs. Cierra el visor con `Esc`, deja el worker en ejecución y vuelve a abrirlo. Usa `--stop` para terminar ordenadamente y luego `resume` si el estado del worker no progresa.

## La estación de monitoreo no abre o dice "Sin conexión con el servidor"

`.\ARES.ps1 dashboard` levanta un servidor local en `http://127.0.0.1:8765/`. Si el puerto está ocupado (error de dirección en uso), elige otro: `.\ARES.ps1 dashboard -Port 8800`. "Sin conexión" en la página significa que el proceso del servidor terminó: vuelve a lanzarlo. Si la página carga pero dice "Sin datos" o "Esperando el emulador", los workers no han publicado todavía en esa raíz: confirma que la campaña corre y que usas la misma `--checkpoint-root`. La estación solo lee archivos; cerrarla nunca detiene una campaña.

## Aparecen `stall_detected`, `agent_fault` o `stall_rewind` en la línea de tiempo

El watchdog detectó falta de progreso y aplicó un escalón de recuperación, o la política contuvo una excepción y replanificó. Un `stall_rewind` cargó un checkpoint previo. Consulta también el reloj **Sin avance de historia**: seguir ganando niveles no demuestra que la campaña progrese. Si se agota el presupuesto de recuperación, conserva `events.jsonl`, `status.json` y el último `frame-*` para diagnosticar (ver [recuperación autónoma](RECUPERACION_AUTONOMA.md)). Para conservar un checkpoint concreto, crea dentro un archivo vacío llamado `PINNED`.

## El supervisor reinicia el worker una y otra vez

Revisa `supervisor-events.jsonl` (o el panel "Supervisor de procesos" de la estación): cada reinicio indica el error, la espera y la semilla. Los errores de configuración no se reintentan. Incluso con reinicios generales ilimitados existe un presupuesto consecutivo sin nueva historia: `recovery_blocked` significa que se agotó y se conservó el checkpoint. Para limitar también el total usa `.\.venv\Scripts\python.exe -m scripts.run_campaign --game both --resume --max-restarts 5`. El avance de frames o XP no renueva el presupuesto de historia.

## Campaña terminó antes de llegar al Campeón

No es certificación. Examina `status.json`, `events.jsonl`, `latest.png`, derrotas/desmayos y el motivo de salida. Reanuda desde el último checkpoint con `.ARES.ps1 resume -Game <firered|emerald|both> -Background`. El sistema no afirma supervivencia absoluta: las decisiones no demostradas se registran y pueden ocurrir derrotas.

## DirectML reporta error y usa CPU

Es el fallback diseñado. Mira el log de entrenamiento para la causa. Un dispositivo AMD detectado no implica que cada operador/backward de esa versión de Torch/DirectML esté soportado. La ruta CPU sigue siendo válida y necesaria como comparación; no cambies paquetes de `requirements.lock` si quieres reproducir resultados.

## Cómo detener y reanudar

Con workers de la raíz predeterminada, solicita parada con checkpoint:

```powershell
.\.venv\Scripts\python.exe -m scripts.run_campaign --backend native --game both --stop
```

Espera que los workers terminen, comprueba status/logs y luego ejecuta `.ARES.ps1 resume -Game both -Background`. No uses `run` sobre una partida que quieres continuar: inicia un juego nuevo.
