# Recuperación autónoma y causas de bloqueo corregidas

Los bloqueos reportados se reprodujeron con savestates nativos. Ninguna corrección modifica flags, coordenadas, Pokémon o PS de las ROMs: el agente solo observa RAM y envía botones.

## Causas corregidas

### Evolución tras un combate ("Delete a move to make room for MUD SHOT?")

Después de ganar, el juego evoluciona al Pokémon con `gMain.inBattle` todavía activo (lo limpia `ReturnFromBattleToOverworld`, que corre después de la evolución). El orquestador enviaba todos esos frames al controlador de combate, que clasificaba el Sí/No de la evolución como "animación" y no pulsaba nada: Emerald quedó detenido indefinidamente con Marshtomp en pantalla. Ahora la escena de evolución tiene prioridad sobre combate y campo en cualquier contexto: el controlador de campo responde el Sí/No, elige el movimiento a olvidar en la pantalla de resumen (que mantiene viva la tarea de evolución) y nunca pulsa B, porque B cancela la evolución. Regresión: `tests/test_never_stuck.py` y `tests/test_native_stall_regressions.py` (repite el savestate real).

### Qué movimiento olvidar

Una sola política (`core/battle/move_learning.py`) decide en combate, evolución y MT/MO: nunca olvida una MO, usa un hueco libre si existe, reemplaza el movimiento de menor valor (potencia × precisión × STAB; los de estado y los reactivos como Bide/Contraataque valen poco; Autodestrucción casi nada) y **rechaza** aprender un movimiento más débil que todos los reemplazables. En el caso real, Marshtomp aprende Disparo Lodo y olvida Venganza (Bide), no su ataque de tipo Tierra.

### Árboles de Corte fuera de cámara (Ruta 2 de FireRed)

Gen 3 solo carga los objetos cercanos a la cámara. Con el jugador en (10,77) el árbol de (15,69) desaparece de `gObjectEvents`; en (10,76) reaparece. El planificador alternaba entre un camino "libre" a través del árbol y una vuelta por Ciudad Verde que, al reentrar con datos estáticos sin objetos, también "olvidaba" el árbol: 16.800 frames oscilando entre dos casillas. Ahora un mismo modelo de obstáculos (`core/navigation/obstacles.py`) alimenta la ruta en vivo y la estática: plantillas de objetos (árboles, rocas, rocas de Fuerza, NPC quietos, objetos) con sus flags, posiciones de ejecución leídas de `SaveBlock1` cuando un script los movió (`setobjectxyperm`), y obstáculos de MO que pasan a ser "removibles con costo" solo cuando la MO es usable. Reentrar al mapa actual usa también los datos en vivo. El agente toma el Bosque Verde, como corresponde.

### Secciones desconectadas del mismo mapa (cuevas)

En el Monte Moon B2F el objetivo estaba en otra sección del mismo mapa, accesible solo subiendo a B1F y bajando por otra escalera. Antes, un objetivo en el mapa actual nunca consideraba salir y volver a entrar. `find_route_to_tile` busca rutas cuya última entrada al mapa objetivo alcanza realmente las casillas desde donde se ejecuta la directiva (junto a un NPC, frente a un mostrador). Además, al estar parado sobre una escalera recién usada, el agente da un paso fuera y vuelve a pisarla (los warps de Gen 3 se activan al pisar).

Otra causa independiente era el sentido de las paredes: al entrar en un tile, el motor comprueba su borde opuesto al sentido de marcha. A* comprobaba el mismo borde y dibujaba pasos imposibles. La regresión cubre las 64 combinaciones direccionales y la ruta real B2F → B1F → 1F → B1F → B2F. El replay corregido llegó a Miguel, obtuvo el fósil y derrotó al rival de Ciudad Celeste. Solo se consideran interacciones a dos tiles cuando entre el jugador y el objetivo existe un mostrador real.

### Briney y Peeko: interceptar una patrulla

Perseguir la posición actual de Briney a su misma velocidad deja al jugador permanentemente detrás. Esperarlo dentro de su circuito también puede bloquear a Peeko y detener ambos NPCs. El controlador reconoce la patrulla observada, se coloca fuera de ella, espera durante un plazo limitado e interactúa únicamente con el `localId` de Briney cuando pasa cerca. El diálogo y los menús conservan sus controladores normales. La repetición del savestate nativo consiguió el viaje a Dewford, sin interactuar con Peeko ni modificar RAM.

### Ataques sin efecto y combates opcionales

Una inmunidad no reduce PS: el observador anterior solo aprendía de deltas positivos y repetía Disparo Lodo contra Wingull. Ahora registra el resultado visible de la intervención propia y excluye ataques del mismo tipo contra ese rival durante el combate. No confunde fallos de precisión ni flags obsoletos con inmunidad y no atribuye a tipos una habilidad desconocida. La evidencia se guarda al reanudar un combate. Durante un viaje de historia o curación, los encuentros salvajes prescindibles admiten huida estimada y acotada a dos intentos; los entrenadores, el entrenamiento planificado y las capturas deseadas conservan su objetivo.

Las creencias de estadísticas se reconstruyen al cambiar el nivel visible del rival: un Makuhita de nivel bajo encontrado en la cueva no puede proporcionar las estadísticas estimadas del Makuhita de Brawly. El cálculo de daño incluye los cambios públicos de ataque/defensa, la habilidad y el objeto propios, estados y PS propios (por ejemplo Torrente), y la planificación considera los cambios visibles de precisión y velocidad. No se consultan estadísticas crudas, habilidad, objeto ni movimientos ocultos del rival para resolver esas estimaciones. Estas comprobaciones están en `tests/test_battle_feedback.py`.

La intercepción rápida se limita a NPCs que patrullan. Los NPCs estacionarios, incluida la enfermera, dejan pasar la actualización normal de la directiva después del diálogo. La recuperación y el supervisor exigen checkpoints que contengan todos los hitos ya observados: resolver un bloqueo local no debe borrar el fósil o una medalla obtenida. Si no existe una copia compatible, se publica un diagnóstico en vez de cargar una partida anterior incompatible.

### Cambio forzado tras un debilitamiento

Si el Pokémon activo se debilitaba, la política evaluaba el daño recibido por un Pokémon con 0 PS (`ValueError: defender_hp must be positive`) y el respaldo leía un `gBattlerPartyIndexes[2]` obsoleto en combates individuales, descartando al único reserva sano. Además la recuperación solo pulsaba B, que no cierra un cambio obligatorio. Ahora la decisión ignora la exposición del Pokémon debilitado, el respaldo usa `gBattlersCount`/`gBattlerPositions`, y la escalera prueba otra fila y la confirma.

### Otras correcciones

- `field_controls.py` llamaba a un método inexistente (`extract_tile_behaviors`) al usar Buceo: habría detenido Emerald al llegar a Buceo.
- El plan de puzle guardado en un checkpoint no se restauraba idéntico (tuplas frente a listas).
- La política de preparación (`CampaignPreparationPolicy`) nunca recibía fotogramas ni victorias, así que su presupuesto de entrenamiento jamás se agotaba ni registraba derrotas; ahora se alimenta cada ciclo, registra derrotas ante el objetivo y se guarda en los checkpoints.
- Los diálogos largos (curación, tutorial del anciano de Ciudad Verde) se confundían con atascos: el watchdog ahora incluye el progreso del texto y de los cursores de menú en la firma de movimiento.
- Un único fotograma de lectura inválida durante un warp contaba como cambio de mapa; ahora se exige estabilidad de tres fotogramas.
- `campaign_blocked` se escribía en cada frame; los eventos repetitivos se agrupan.
- Autodestrucción/Explosión ya no se consideran un plan de daño.

## Inteligencia adaptativa: requisitos, lecciones y preparación

Estas correcciones nacen de los dos atascos del 15 de octubre: FireRed detenido con `recovery_blocked` en la garita de Ruta 5 (objetivo `CELADON_TEA`) y Emerald perdiendo cinco veces contra Brawly y retrocediendo cada vez a antes del entrenamiento.

### Guardias que no dejan pasar (FireRed: el té de Azulona)

El guardia de las garitas de Azafrán dice que tiene sed y empuja al jugador un paso atrás hasta que se le entrega el Té (`FLAG_GOT_TEA`). El planificador no lo sabía: elegía la garita norte, era rechazado, vetaba esa salida 2.400 fotogramas y elegía la garita sur por el Paso Subterráneo; al expirar el veto volvía a la primera. Ahora:

- **Conocimiento de fuentes** (`core/navigation/script_gates.py`, `data/gates_firered.json`, `data/gates_emerald.json`): una ejecución simbólica de los scripts de pret detecta cada *trigger* que devuelve al jugador mientras sigue armado y las condiciones de las ramas que lo dejan pasar (flags, objetos, variables, entrenadores vencidos, movimientos). FireRed tiene 95 casillas de este tipo (las cuatro garitas de Azafrán, los guardias de medallas de la Ruta 23, el guía del gimnasio de Plateada, la bicicleta del Camino de Bicis...) y Emerald 34 (tormenta de arena de la Ruta 111 sin Gafas Aisladoras, salidas de Ruta 101 durante el rescate de Abedul...). El bloqueo es direccional: solo se prohíben los pasos que el script deshace, así que salir de Azafrán hacia el norte sigue permitido. La ruta y el A* en vivo usan el mismo modelo; el agente toma Ruta 9 → Túnel Roca → Pueblo Lavanda → Paso Subterráneo → Ruta 7 → Azulona.
- **Aprendizaje en ejecución** (`core/navigation/pushback.py`): si un script desplaza al avatar en sentido contrario justo después de un paso voluntario, el paso (y las casillas hermanas del mismo trigger) queda denegado mientras el «mundo» —hitos y objetos clave— no cambie. Un hito o un objeto clave nuevo permite un reintento, porque el requisito pudo cumplirse. Caminar de vuelta por decisión propia no cuenta: solo cuenta el desplazamiento ocurrido mientras corre un script. Evento `passage_refused` con el requisito si se conoce.
- **Requisitos como sub-objetivos**: cuando no queda ruta abierta, una búsqueda hipotética (sin puertas y con los actores de historia apartables) identifica qué la bloquea: la flag que abre una puerta, el objeto que pide un guardia o la flag que oculta a unos NPCs. Se busca el evento que la activa o entrega el objeto y se va allí primero (`gate_requirement`, modo `UNLOCK_GATE`). Validado en Emerald: la fila del Equipo Aqua bloquea el museo de Portual hasta hablar con Dock en el astillero (`FLAG_HIDE_SLATEPORT_CITY_TEAM_AQUA`); el agente fue al astillero, volvió y completó `STERN_GOODS`. Solo si nada conocido abre el paso, se intenta igualmente una puerta (nunca atravesar NPCs), y un rechazo real queda aprendido.

### Preparación antes de un jefe y análisis de derrotas (Emerald: Brawly)

Las derrotas registradas mostraban a Marshtomp entrando al combate con 53/64 PS y Disparo Lodo sin PP, atacando con Placaje. La política solo curaba con PS < 40 % o sin ningún PP. Ahora (`core/planner/readiness.py`):

- Antes de un líder, rival, entrenador obligatorio o miembro del Alto Mando se exige el miembro principal con ≥ 85 % de PS y sus dos mejores ataques (potencia × STAB) con ≥ 50 % de PP; si no, se visita un Centro Pokémon (`pre_battle`). Tres restauraciones sin llegar al jefe relajan la exigencia para no crear un bucle.
- Fuera de los jefes se cura cuando los mejores ataques bajan del 25 % de PP: entrenar con Placaje mientras Disparo Lodo está vacío desperdicia combates.
- Cada derrota se analiza: si se entró sin estar preparado, la lección es restaurar antes del reintento (no sube el nivel objetivo); si se entró preparado, sube el nivel objetivo (+2 la primera vez, +3 después).

Resultado verificado al repetir el savestate real: curación en Ciudad Férrica, viaje con Briney y victoria contra Brawly usando Disparo Lodo con PS completos.

### Reloj estratégico sin destruir el entrenamiento

El reloj de 72.000 fotogramas sin historia nueva interrumpía el entrenamiento hacia Brawly y retrocedía a un checkpoint anterior, perdiendo los niveles ganados: entrenar, retroceder, perder, entrenar... Ahora cada nuevo máximo de preparación (niveles y tamaño del equipo) descuenta tiempo del reloj, con un presupuesto acotado por hito; volver a ganar niveles perdidos por un retroceso no cuenta. Al vencer el plazo se cambia de estrategia en vez de repetir la misma (`strategy_shift`): por ejemplo, intentar el objetivo con el equipo actual o olvidar vetos de salidas, y se concede un margen para ejecutar el nuevo plan. Un retroceso por falta de historia solo puede ir a checkpoints que conserven los niveles actuales.

### Planificación estable y barata

- La visión de los entrenadores se calcula desde las plantillas del mapa y su flag de derrota, no desde los actores cargados por la cámara: en el Túnel Roca el plan alternaba arriba/abajo cada vez que un excursionista aparecía o desaparecía de pantalla.
- No se planifica en el fotograma de cruce entre mapas conectados, cuando las coordenadas aún no pertenecen al mapa nuevo: una directiva calculada desde ese punto fantasma (sin ruta al Centro Pokémon) quedaba en caché toda la visita y creaba un bucle Ciudad Férrica ↔ Bosque Petalia.
- La salida comprometida se reutiliza mientras siga alcanzable antes de lanzar otra búsqueda, y una búsqueda fallida se memoriza 900 fotogramas (se invalida con cualquier flag nueva). Una ruta regional imposible costaba varios segundos por fotograma.
- Un veto temporal nunca puede ser la causa de quedarse sin ruta: se valida con costes reales y se levanta si deja el objetivo inalcanzable.
- El barco de Dewford a Portual dejaba al jugador, según los datos, en el agua de la Ruta 109; ahora aterriza en la playa (21, 23), como el script.

### Compras, pociones y correr

- El agente compra en las tiendas (`core/shopping.py`): cuando faltan objetos curativos (unas dos curaciones completas del miembro principal) o Poké Balls y hay dinero, visita la tienda cercana y usa la interfaz real de compra leyendo la lista, el cursor y la cantidad desde RAM. Gasta como máximo el 70 % del dinero. Verificado: Emerald compró 2 Superpociones y 1 Poké Ball en Portual; FireRed 8 Superpociones y 10 Poké Balls en Carmín.
- En combate puede usar una poción cuando atacar ahora probablemente pierde y la curación supera el daño esperado; se compara con el mismo modelo de riesgo que ataques y cambios. Verificado en ambos juegos (PS 61 → 81 en Emerald, 152 → 161 en FireRed).
- Con las Deportivas (`FLAG_SYS_B_DASH`) siempre corre: el modelo del mundo prefería caminar y duplicaba el tiempo de viaje.
- Las inmunidades observadas (Disparo Lodo contra Taillow) se recuerdan entre combates; un daño posterior las desmiente.

### Lecciones duraderas

`lessons.json`, junto a los checkpoints de cada juego, guarda los pasos denegados, las inmunidades y el historial de derrotas por objetivo. Se fusiona al cargar cualquier checkpoint y al reiniciar el worker, de modo que un retroceso o un reinicio del supervisor no olvidan lo aprendido en la línea temporal abandonada.

## Contención de fallos

En campañas nativas, una excepción de la política (decisión de combate, menú, ruta, obstáculo) ya no termina el worker: se registra como `agent_fault` (agrupada, con origen del error), se liberan las intenciones, se envía una entrada segura (A solo en pantallas no listas) y se replanifica al frame siguiente. Si el fallo persiste, deja de haber progreso y actúa la escalera de recuperación. Una decisión de combate imposible se reemplaza por el ataque legal más fuerte; un cambio rechazado por el motor, por el primer reserva sano.

## Escalera de recuperación y límites

Avanzar frames no cuenta por sí solo como jugar. El watchdog observa coordenadas nuevas, experiencia, inventario, hitos, flags persistentes vistos alguna vez (monotónico) y curaciones; y, fuera del mapa abierto, el progreso del texto y de los menús. Cuando no hay progreso:

Además hay un reloj estratégico independiente de 72.000 frames por defecto: solo hechos nuevos de historia lo reinician. Ni XP, victorias salvajes, curaciones, nuevas casillas, cambios temporales de objetivo ni la carga de un checkpoint lo borran. Es tiempo de juego acumulado, no tiempo de pared. Su contador y sus intentos se ven en el dashboard. Las restauraciones conservan historia y presupuesto del intento abandonado.

1. **Replanificar**: libera teclas, cancela intenciones y descarta menús inesperados con B (solo una vez).
2. **Sondear**: en pantallas no reconocidas, pulsaciones ordinarias (A; en un cambio obligatorio, otra fila + A). En el mapa, un paso adyacente validado por colisiones.
3. **Desviar / vetar**: veta temporalmente la salida que se estaba usando (solo si existe otra ruta) y camina a una casilla alcanzable al azar para romper un ciclo determinista. Nunca se envía B a una evolución.
4. **Escalar**: más variedad de entradas y desvíos.
5. **Retroceder**: carga el checkpoint más reciente guardado antes del último progreso observado (cada retroceso adicional va más atrás), conservando lo aprendido del intento fallido —casillas visitadas, flags vistos, salidas vetadas y contadores de diagnóstico— y con otra semilla, para no repetir el mismo camino. Hasta 6 retrocesos por estancamiento.

Si se agotan los retrocesos el worker termina con `CampaignStalled`. El supervisor reintenta desde un checkpoint verificado con otra semilla y espera creciente (hasta 60 s); si falla sin nuevos hitos, prueba uno anterior. Existe un límite independiente para reinicios consecutivos sin historia nueva, también cuando el presupuesto general es `--max-restarts -1`: avanzar miles de frames o ganar XP no lo reinicia. Al agotarlo, la línea de comandos ya no detiene la campaña: publica `recovery_cooldown`, espera `--stagnation-cooldown` segundos (300 por defecto) y vuelve a intentarlo desde el checkpoint verificado más reciente con las lecciones acumuladas. Un valor negativo restaura el comportamiento anterior, que publica `recovery_blocked` y conserva el diagnóstico. Los errores de configuración no se reintentan. También se detectan cuelgues sin latido durante `--heartbeat-timeout` (120 s por defecto). Los eventos aparecen en la estación y en los registros.

Esto maximiza la autonomía, pero no es una garantía matemática de terminar cualquier partida: una causa desconocida se diagnostica con esos eventos y su savestate.

## Reanudar normalmente

Desde la raíz del repositorio:

```powershell
.\ARES.ps1 resume -Game both -Background
.\ARES.ps1 dashboard
```

Para estado sin abrir otro emulador:

```powershell
.\ARES.ps1 status -Game both
```

No inicies `run` para continuar una partida existente: ese comando empieza una nueva partida.

## Reproducir un checkpoint en otra carpeta

La herramienta de diagnóstico conserva la campaña original y escribe una rama de prueba en otro directorio:

```powershell
.\.venv\Scripts\python.exe -m scripts.verify_native_recovery --game firered --source-root artifacts\campaigns --output-root artifacts\recovery_verification --max-frames 15000
.\.venv\Scripts\python.exe -m scripts.verify_native_recovery --game emerald --source-root artifacts\campaigns --output-root artifacts\recovery_verification --max-frames 15000
.\.venv\Scripts\python.exe -m scripts.dashboard --checkpoint-root artifacts\recovery_verification
```

Cada juego produce `recovery_result.json`, imagen, eventos y un checkpoint íntegro. La salida verifica experiencia/resultados de combate, hitos y cambio de mapa observados; **no certifica al Campeón**. `--reset-watchdog` rearma expresamente los intentos de un checkpoint cuyo watchdog ya estaba agotado.

Los checkpoints usados por las pruebas nativas de regresión contienen un archivo `PINNED`, de modo que la retención de checkpoints no los borra. Son artefactos locales, excluidos de Git; los tests correspondientes sí forman parte del código fuente y se omiten si el savestate no existe.
