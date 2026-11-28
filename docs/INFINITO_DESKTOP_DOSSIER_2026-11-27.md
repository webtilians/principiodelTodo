# INFINITO Desktop para Windows: dossier de implementación

27/11/2026 · Hilo (Claude Code), a petición de Enrique · Estado: **propuesta, nada implementado**.

**Revisión 1 (27/11/2026).** Aplica las cinco propuestas de revisión guardadas en INFINITO el mismo día (`8b443130`,
`d99e9389`, `ebcf779d`, `5ce28448` y `dd825978`), contrastadas con master `7bd314b`. Qué cambia y por qué está en la
§2.2.

Este dossier es para el agente que construya la app de escritorio de INFINITO. Parte de dos cosas del 27/11/2026: la
idea de Enrique en Talent Land (`7bef0c68`: «INFINITO tiene que estar conectado al PC de alguna forma; también podría
sacar información del PC») y la valoración de Hilo del mismo día (`7ea7c944`). Va en paralelo al dossier del iPhone
(`docs/INFINITO_APP_IPHONE_DOSSIER_2026-11-27.md`, PR #197, aún sin fusionar), con el que comparte el contrato de
dispositivos de la §5. Lo marcado **[verificar]** no se ha comprobado y hay que mirarlo antes de dar por buena esa
parte. Lo marcado **(propuesta)** es una cifra o un nombre para decidir en la puerta, no un dato.

## 0. Antes de empezar (para el agente)

1. Lee [README.md](../README.md), [AGENTS.md](../AGENTS.md), este dossier entero,
   [cognition/infinito.py](../cognition/infinito.py), [cognition/notify.py](../cognition/notify.py),
   [src/infinito3/remote_apps.py](../src/infinito3/remote_apps.py),
   [src/infinito3/remote_accounts.py](../src/infinito3/remote_accounts.py) y, del dossier del iPhone, las §3 y §5
   (fase 0).
2. Una rama desde `master` y un PR contra `master` por fase. Enrique fusiona. No se empieza una fase sin la puerta de
   la anterior, y cada fase tiene su propia puerta: que una pase no da por buena la siguiente.
3. **Solo Windows.** Enrique trabaja en un portátil con Windows 11 y no tiene Mac. Los ejecutables se compilan en
   GitHub Actions; las pruebas en el PC real las hace Enrique, y cada fase termina con una lista de comprobación para
   él en la descripción del PR.
4. Antes de la fase 1, que Enrique responda a las decisiones abiertas de la §10.
5. Las fases 0 a 4, 6 y 7 son producto: no hace falta protocolo experimental, pero sí pruebas. **La fase 5 es
   investigación:** según AGENTS.md, antes de ejecutarla hace falta un protocolo escrito en `docs/` con la pregunta, la
   base de referencia, la puerta de aprobado y suspenso y qué resultado cambiaría el plan. Cualquier mejora que se
   afirme se compara con la base RAG al mismo presupuesto de tokens y sobre datos que no se usaron para ajustarla. No
   se tocan los protocolos congelados (M1, M2, E13-A) ni se reutilizan particiones selladas o consumidas.
6. `src/infinito3` sigue siendo solo biblioteca estándar y `python -m pytest tests/test_*v3.py` sigue en verde con cada
   cambio. La app vive fuera del núcleo, como `cognition/` y `mcpb/`.
7. No leas ni muestres el contenido de `~/.infinito/cognition/session.json`, `~/.infinito/remote-session.json` ni de
   la bóveda: son sesiones y secretos de Enrique.

## 1. Para qué, y para qué no

**Qué es.** INFINITO Desktop no es solo un panel de mantenimiento: es el **sistema nervioso de INFINITO en el PC**.
Recoge, con permiso y procesado en local, lo que pasa en el ordenador y lo convierte en episodios con sentido, y en el
futuro es también un **nodo de procesamiento cognitivo local** que pone los recursos libres del PC al servicio del motor
cognitivo ya existente. Va en la línea de la decisión del 26/11/2026 de conectar todos los dispositivos a la memoria
(`69497532`), y de las del 27/11 sobre la arquitectura híbrida del motor (`c5e22562`) y la búsqueda piramidal
(`3e18ffc8`).

**El problema de hoy.** INFINITO ya está en el PC de Enrique, pero en piezas sueltas, y cuando una falla nadie se
entera:

| Pieza en el PC | Cómo avisa hoy |
|---|---|
| Hooks de Claude Code (`infinito-remote context` y `checkpoint`) | No avisa: sin sesión o sin respuesta del servidor no añade nada, a propósito |
| Bóveda local (`infinito-mcp` con `INFINITO_LOCAL_MEMORY=off`) | Solo cuando una IA la usa |
| Motor cognitivo: tareas «INFINITO Motor cognitivo» y «INFINITO Motor cognitivo (noche)» | Una notificación de Windows cuando llega una revisión (`cognition/notify.py`) |
| Copias: tarea «INFINITO copias de seguridad», cada día a las 21:00 | Una línea en `~/.infinito/copias-servidor/pull.log` |

Dos casos de este mes:

- **Copias.** La última buena es del 7/11/2026 (`d4782ddf`). Desde entonces cada intento falla, y el único aviso es una
  línea en `pull.log` que nadie abre. La del 27/11 a las 21:00 dice
  `FAILED: … is NOT usable: unexpected entry 'accounts/enrikelr/cognition.sealed'`, mientras la tarea marca 0 como
  último resultado.
- **Motor.** Del 25 al 27/11 el supervisor (qwen4-max) respondió `403 insufficient_quota`, y la notificación decía
  «ha terminado la revisión del día» (`07c2bbfe`, `3cdd32da`). Lo arregló el PR #196, pero se descubrió tres días
  tarde y mirando a mano.

Y lo que pasa en el PC solo llega a la memoria si una IA decide guardarlo.

**Objetivos, por fases (§6).**

1. Decir de un vistazo si INFINITO funciona en este PC y avisar cuando algo deja de funcionar (F1).
2. Mandar algo a la memoria en dos segundos, desde cualquier sitio, sin abrir una IA (F2).
3. Hablar con el servidor con el mismo contrato que el resto de dispositivos (F3, compartido con el iPhone).
4. Observar en pasivo, solo lo que Enrique encienda, y convertirlo en episodios (F4).
5. Prestar los recursos libres del PC a modelos ligeros locales que organicen la memoria, sin estorbar a Enrique (F5,
   investigación).
6. Integrar documentos de carpetas y repositorios Git autorizados (F6).
7. Mandar capturas de pantalla a mano, con vista previa y confirmación (F7).

**Fuentes que se pueden observar, siempre con permiso** (`8b443130`): aplicaciones usadas, proyectos activos, tiempo de
actividad e inactividad, cambios en repositorios Git autorizados, documentos de carpetas autorizadas, errores de
servicios y procesos autorizados, CPU, GPU y RAM disponibles, y encendido, suspensión y apagado. Todo se procesa en
local y sale como episodio, nunca como lectura continua, título, ruta ni contenido en crudo.

**Fuera, y por qué.**

- **Teclado, portapapeles, capturas de pantalla automáticas o periódicas, historial del navegador, micrófono abierto,
  correo y mensajería del PC, y cualquier vigilancia indiscriminada.** Arrastran contraseñas, claves y datos de
  terceros, y la regla de AGENTS.md es no guardar secretos nunca. Microsoft tuvo que retrasar Recall en 2024 por las
  críticas de privacidad y acabó haciéndolo opcional.
- **Indexar el disco.** Solo carpetas y repositorios que Enrique elija.
- **Una segunda memoria o un segundo motor.** Desktop se integra con la memoria del servidor y con el motor de
  `cognition/`, con sus contratos y su estado (§6, F5).
- **Un chat.** Enrique lo descartó el 11/11/2026 (`1d1f907b`).
- **macOS y Linux.** No hay Mac; Linux, si alguien lo pide.
- **Distribución pública firmada.** De momento es para uso propio; firmar el ejecutable se decide después.
- **Teams.** Un agente de escritorio para empresas es otro producto, y según AGENTS.md en Teams solo entran funciones
  estables y probadas.

## 2. Estado de partida

### 2.1 Lo que ya existe (no rehacerlo)

| Pieza | Dónde | Estado |
|---|---|---|
| OAuth con registro dinámico, PKCE y vuelta a `127.0.0.1`, con nombre de app propio | `cognition/infinito.py` (`Infinito.login`, `_token`); `src/infinito3/remote_client.py` (`login`, con el nombre fijo «INFINITO on <equipo>») | En uso: el motor y los hooks de este PC |
| Ámbitos (*scopes*) pedidos por la app | `src/infinito3/remote_accounts.py`: acepta cualquier ámbito que pida la app y lo guarda en el código y en los tokens (`data["scopes"]`); hoy solo hay uno, `SCOPE` | En producción. Nadie los usa para limitar |
| Llamar a una herramienta MCP con el token | `cognition/infinito.py` (`Infinito.call`) | En uso por el motor |
| Niveles por app: `all`, `add` y `read`; **una app nueva empieza en `all`** | `src/infinito3/remote_apps.py` (`LEVELS`, `DEFAULT = "all"`, `allowed`). `add` permite además la cola de tareas (`task_*`), `guideline_propose`, `reflection_add`, `cognition_report` y `photo_save` | En producción. Ningún nivel es el mínimo de un dispositivo |
| Página Apps del portal | `remote_apps.apps_body`: nombre, desde dónde entra, fecha de conexión, nivel y Desconectar | En producción. **Sin última actividad ni capacidades** |
| Filtro de secretos | `src/infinito3/safety.py` (`SensitiveInformationFilter`, solo biblioteca estándar); lo aplica `engine.py` en `remember` | En producción. Se puede usar también en local |
| Olvido en cascada | `forget` en `remote.py` borra las reflexiones y los elementos del motor que citan lo olvidado (`drop_citing`) | En producción |
| Notificación de Windows sin dependencias, cuyo clic abre una página | `cognition/notify.py` (`toast`, por PowerShell y con su identidad) | Comprobada por Enrique el 11/11/2026 (`3482fe19`) |
| Estado del motor | `python -m cognition status` (última revisión, gasto del día, propuestas esperando, en pausa); `~/.infinito/cognition/`: `PAUSED` (contiene la fecha), `diary.jsonl` (`review_end` con `outcome` y `judged`), `ledger.json`, `reports/` | En el PC. En pausa desde el 27/11 a las 19:26 (`4329fdc3`) |
| Planificación del motor | `scripts/install_cognition_task.ps1`: `python -m cognition tick` cada hora y una tarea de noche que despierta el PC | En el PC |
| Estado de las copias | Tarea «INFINITO copias de seguridad» y `~/.infinito/copias-servidor/pull.log` | Fallan desde el 7/11 |
| Lectores, frases, ritmo (`Throttle`) y trayecto sellado | `src/infinito3/remote_devices.py` | En master |
| API de dispositivos con token, `POST /api/devices/<acción>` | Fase 0 del dossier del iPhone (PR #197) | Propuesta, sin construir |
| `photo_save` | `src/infinito3/remote.py`. Recibe un `download_url` público HTTPS que descarga el servidor, y pide la invitación a la bóveda (`vault_allowed`) | En producción. **No sirve para un archivo local** |
| Subida de archivos (`file_save`, `file_search`, `file_read`) | Tarea aprobada `6987ed99` (20/11/2026) | Sin construir |
| Motor continuo mientras el PC está encendido | Tarea aprobada `bacf4244` (27/11/2026) y decisiones `c5e22562` y `3e18ffc8` | Propuesta, sin construir |
| Modelos locales | Ollama con `qwen3.8-motor` en el PC (`1f63d4b1`, 16/11/2026) y una RTX 4060 de portátil | **[verificar]** que siguen así y cuánta VRAM hay libre |
| Evidencia previa sobre organizar la memoria | M1 (PR #192, abierto): «no pasa nada y la recuperación plana se queda». E13-A (PR #79, abierto): «ventaja adaptativa insuficiente — no integrar». M2 (PR #191): protocolo, sin resultados | Ver la F5 |
| Ejecutable de Windows con PyInstaller en Actions | `webtilians/lain`, `.github/workflows/beta01-windows.yml` (`windows-2022`, `pyinstaller==6.16.0`, `--onedir` y `--onefile --windowed`) | En uso en LAIN |
| Extensión de Claude Desktop (`.mcpb`) | `mcpb/` y `.github/workflows/desktop-extension.yml`, adjunta a la release v0.7.0 (6/11/2026) | Publicada pero desfasada: ver la F0 |

### 2.2 Qué cambia en esta revisión

| Propuesta | Qué cambia en el dossier | Dónde |
|---|---|---|
| `8b443130`: ampliar el propósito y adelantar la observación pasiva | Desktop pasa a ser el sistema nervioso de INFINITO en el PC y un futuro nodo cognitivo local. La observación pasiva sube de la antigua F5 a la F4, con las fuentes de la lista. Se distingue lo observado, lo declarado y lo inferido | §1, §3, F4 |
| `d99e9389`: integración cognitiva local | Fase nueva, F5: Desktop planifica los recursos para que modelos ligeros locales organicen la memoria en incremental, integrado con `cognition/`, sin segunda memoria; marcada como investigación con protocolo | F5 |
| `ebcf779d`: API universal de dispositivos | Contrato común INFINITO Device API v1: identidad, capacidades, permisos por capacidad, eventos versionados con ID e idempotencia, procedencia y registro. Lo comparten el PC, el iPhone, el reloj y la domótica | §5, F3 |
| `5ce28448`: correcciones técnicas | Ámbito mínimo de dispositivo desde el registro y aplicado en el servidor; cola local cifrada con DPAPI; ID de evento e idempotencia; procedencia; registro de dispositivos con última actividad; presupuestos de CPU, RAM y GPU; actualizaciones verificadas; panel de qué se recoge y qué se envía. Se quita la regla de las cinco capturas de la §10 y cada función tiene su puerta | §3, §5, F1–F5, §11 |
| `dd825978`: orden de desarrollo | Fases F0 a F7 en ese orden. La captura de pantalla a mano pasa de F3 a F7, y las carpetas de F4 a F6, con los repositorios | §6 |

Lo que **se conserva** del dossier anterior, como piden las propuestas: Python, la app de bandeja de Windows, OAuth con
PKCE y un único refresco, DPAPI, la arquitectura por módulos, la captura rápida, el estado y los avisos, el estado «no
lo sé» cuando algo falla, la privacidad y las pruebas por fase.

## 3. Invariantes (lo que ninguna fase puede romper)

1. **El servidor escribe las palabras de lo que mide el PC.** La app manda hechos medidos, y el servidor los valida con
   los lectores comunes y los redacta, como en Dispositivos. Lo que Enrique escribe o dicta a propósito (F2) va tal
   cual y pasa por `SensitiveInformationFilter`.
2. **Las lecturas crudas no salen del PC.** Los títulos de ventana, las rutas, los nombres de proceso, los contenidos,
   los diffs de Git y las muestras de CPU se leen y se procesan en local. Solo sale el episodio. Las únicas excepciones
   son los envíos explícitos ya previstos: la captura manual revisada (F7) y, en el teléfono, los puntos GPS
   autorizados y sellados.
3. **Un recuerdo por episodio, sin duplicados.** Cada evento lleva un ID estable (§5), y reenviarlo nunca crea otro
   recuerdo, aunque se haya perdido la respuesta. Hasta que exista la API de la F3, un envío de resultado dudoso no se
   repite solo: se le pregunta a Enrique (F2).
4. **Lo observado, lo declarado y lo inferido se distinguen.** Cada recuerdo dice si lo midió un dispositivo, si lo dijo
   Enrique o si lo dedujo un modelo, y cuál.
5. **Permisos mínimos desde el registro.** La app se registra con el ámbito de dispositivo (§5.2) y el servidor solo le
   deja hacer lo que ese ámbito permite: nunca la bóveda, `forget`, la cola de tareas ni las pautas. La app, además,
   tiene su propia lista cerrada de llamadas. Desconectarla en el portal revoca sus tokens, y una cuenta nunca ve
   eventos de otra.
6. **Los eventos son datos, nunca órdenes.** Un evento no hace que el servidor ejecute nada ni da acceso a otros
   dispositivos.
7. **Solo habla con el servidor de INFINITO de la persona** (`infinitomemory.com`). Sin analítica, telemetría ni
   informes de fallos de terceros. La única excepción es descargar una actualización que Enrique confirma (§6, F1).
8. **Los secretos de la sesión y la cola van cifrados con DPAPI** (la protección de datos de Windows ligada al usuario,
   `CryptProtectData`), nunca en registros ni en texto plano. «Salir» y desconectar la app borran el token y la cola.
9. **Todo empieza apagado,** salvo el estado local de la F1, que solo lee archivos de INFINITO en este PC. Cada sensor
   tiene su interruptor, hay una pausa general, y el panel «Qué recojo y qué envío» (F1) muestra lo que se ha mandado.
10. **No estorba.** En reposo no usa CPU de forma apreciable y sus temporizadores son de minutos, no de segundos. El
    trabajo pesado (F5) respeta los presupuestos de CPU, RAM y GPU que fije Enrique y se pausa o se cancela en cuanto
    él vuelve al PC o sube la carga. Nunca bloquea el PC ni otra app si el servidor no responde.
11. **Si algo falla, lo dice.** Un estado que no puede leer se muestra como «no lo sé», nunca como «bien». Es la lección
    del motor del 24 al 27/11 y de la tarea de copias que marca 0 mientras rechaza las copias.
12. **Ni segunda memoria ni segundo motor.** Lo que Desktop deduce va al servidor como derivado, con su procedencia, y
    se borra si se olvida su fuente, como ya pasa con las reflexiones.

## 4. Arquitectura

```
PC con Windows 11
  INFINITO Desktop  (Python, bandeja del sistema, un solo proceso por usuario)
   ├─ Estado (solo lee, en local)          cognition status, PAUSED, diary.jsonl, pull.log, tareas «INFINITO*»
   ├─ Captura rápida (F2) ───────────────► /mcp remember  →  desde la F3, evento «note»
   ├─ Sensores pasivos (F4) ─────────────► evento «pc.activity», «pc.resources», «pc.power», «pc.errors»
   ├─ Nodo cognitivo local (F5) ─────────► ventana de recursos para el motor de cognition/ (Ollama en local)
   ├─ Documentos y repositorios (F6) ────► file_save (tarea 6987ed99) y evento «pc.repo»
   ├─ Captura visual a mano (F7) ────────► evento «photo»
   └─ Cola cifrada (DPAPI) y panel «Qué recojo y qué envío»
          HTTPS · Authorization: Bearer <token con el ámbito de dispositivo>
                       ▼
infinitomemory.com   /mcp (ya existe)   ·   POST /api/devices/v1/events (F3)   ·   registro de dispositivos (F3)
```

**Módulos** del paquete `infinito_desktop`: `tray` (icono y menú), `session` (inicio de sesión y DPAPI), `status`
(comprobaciones), `queue` (cola cifrada), `capture`, `sensors/` (un módulo por sensor), `planner` (recursos para la F5),
`docs` (F6), `screen` (F7) y `panel` (qué se recoge y qué se envía). Cada sensor es un módulo independiente con su
interruptor, sus pruebas y su puerta.

**Tecnología recomendada: Python con solo la biblioteca estándar.**

- `ctypes` para el icono de la bandeja (`Shell_NotifyIconW`), el atajo global (`RegisterHotKey`), DPAPI, una sola
  instancia (`CreateMutexW`), el tiempo sin usar el PC (`GetLastInputInfo`, que solo da la hora de la última
  pulsación, nunca qué se pulsó) y la memoria y la CPU (`GlobalMemoryStatusEx`, `GetSystemTimes`).
- `tkinter` para la ventana de captura, los ajustes y el panel.
- `winreg` para arrancar al iniciar sesión, si Enrique lo enciende.
- La notificación de `cognition/notify.py` y el filtro de `src/infinito3/safety.py`.
- La GPU, si hace falta, con `nvidia-smi` cuando exista **[verificar]**. Los modelos de la F5 corren en Ollama, no
  dentro de la app.

Motivos: es como `remote_client.py` y `cognition/`, el código de inicio de sesión ya existe, y un producto con doble
licencia (AGPL y comercial, `1fcd299`) no suma dependencias cuyas licencias haya que revisar. Las alternativas están
en la decisión P-3: `pystray` con Pillow (menos código, pero dependencias y licencia **[verificar]**), o Tauri o
Electron si algún día hace falta una interfaz rica.

**Ejecutable:** PyInstaller `--onedir --windowed` en `windows-2022`, como LAIN. Cada build publica el SHA-256 de lo que
genera. Sin firmar, Windows SmartScreen avisará al abrirlo **[verificar]**; para uso propio basta con «Ejecutar de
todas formas».

**Dónde:** `desktop/` en este repositorio (decisión P-2). Las pruebas de la lógica pura van en
`tests/test_desktop_v3.py`, como `tests/test_cognition_v3.py`, para que corran en la CI del núcleo.

**Inicio de sesión:** no copiar el flujo una tercera vez. Sacar a una función común el login con nombre de app, ámbito
y ruta de sesión, en `src/infinito3/remote_client.py`, que ya es biblioteca estándar. La usa la app y, si se quiere,
el motor. Cambiar `remote_client.py` exige sus pruebas en verde.

## 5. INFINITO Device API v1 (contrato común de dispositivos)

En vez de una integración por aparato, un solo contrato que reutilizan el PC, el teléfono, el reloj y la domótica
(Home Assistant, decisión `25977129`), y después el coche, las gafas, las cámaras o los robots, sin rediseñar el
núcleo. El dossier del iPhone (PR #197) lo adopta en su fase 0.

**Estado (28/11/2026).** El servidor ya está construido en la rama `feat/device-api-v1-20261127` (fase 0 del iPhone,
decisión P-8 del iPhone: la API va primero). Tiene la ruta, el ámbito, las capacidades `phone.motion`, `phone.place`,
`phone.track` y `note`, la idempotencia, la procedencia, el ritmo compartido y el registro en la página Apps
(`src/infinito3/remote_device_api.py`, documentado en `remote/README.md`). Falta que Enrique lo fusione y se despliegue.
Las capacidades del PC (`pc.*`) y el pulso del reloj siguen en propuesta. Lo que cambió al construirlo está en §5.2.

### 5.1 Identidad y registro

- Cada dispositivo es un cliente OAuth propio, registrado por registro dinámico con PKCE, ligado a la cuenta de la
  persona que inicia sesión. El nombre lo elige el dispositivo («INFINITO Desktop en <equipo>», «INFINITO para
  iPhone»).
- Además del `client_id`, el dispositivo genera un `device_id` estable (UUID) y lo manda en cada evento. Si se reinstala,
  es otro dispositivo.
- **Registro de dispositivos** en el portal (hoy, la página Apps): nombre, tipo, capacidades declaradas, cuáles están
  permitidas, última actividad (hora del último evento aceptado) y Desconectar. La última actividad se guarda sellada
  con la llave de la cuenta, como el resto.

### 5.2 Ámbito de dispositivo y permisos por capacidad

- El dispositivo pide en el registro y en `/authorize` un ámbito propio, `infinito:device` **(propuesta)**. Hoy
  `remote_accounts.py` ya acepta cualquier ámbito y lo guarda en el token, así que pedir menos no rompe nada.
- Un token con ese ámbito **solo** puede llamar a `POST /api/devices/v1/events`. En `/mcp` recibe `403` con el texto de
  `remote_apps.device_refusal`. La excepción temporal de `remember` por `/mcp` ya no hace falta, porque la API existe:
  la captura rápida de la F2 es un evento `note`.
  Nada de lectura de la memoria, bóveda, `forget`, cola de tareas, pautas, reflexiones ni `photo_save`. El servidor lo
  comprueba en cada llamada, sea cual sea el nivel de la app en Apps.
- **Capacidades:** el dispositivo declara las suyas (por ejemplo `note`, `photo`, `pc.activity`, `pc.resources`,
  `pc.power`, `pc.errors`, `pc.repo`, `phone.motion`, `phone.place`, `phone.track`, `watch.pulse`). La persona puede
  apagar cada una en el registro. Un evento de una capacidad no declarada o apagada se rechaza con `403`.
- **Cómo se declaran (fijado al construirla):** como ámbitos OAuth, uno por capacidad, en el registro del cliente y en
  `/authorize`: `infinito:device infinito:device:phone.motion infinito:device:note`… Así:
  - la página de inicio de sesión dice qué podrá añadir el dispositivo;
  - un refresco no puede añadir ninguna (el SDK de MCP rechaza un ámbito que no dio el primer inicio);
  - un dispositivo que no pide ámbito en `/authorize` recibe los de su registro, nunca `memory`.

  Añadir una capacidad en una versión nueva de una app obliga a registrarse e iniciar sesión otra vez, y eso es un
  consentimiento nuevo. Los metadatos del servidor no anuncian estos ámbitos, para que una app que pide todos los que
  ve no se convierta en dispositivo.
- Ningún dispositivo ve los eventos de otro ni tiene acceso a él por defecto.

### 5.3 Formato del evento (v1)

```json
{
  "v": 1,
  "event_id": "8f0c…",
  "device_id": "3b1e…",
  "capability": "pc.activity",
  "type": "episode",
  "origin": "observed",
  "observed_from": "2026-11-27T10:05:00+01:00",
  "observed_to": "2026-11-27T12:15:00+01:00",
  "payload": {"label": "INFINITO", "minutes": 130}
}
```

- `event_id`: UUID que genera el dispositivo una vez por evento y conserva en la cola hasta tener respuesta.
- `origin`: `observed` (lo midió el dispositivo), `declared` (lo dijo la persona) o `inferred` (lo dedujo un modelo,
  con `payload.model`).
- `observed_from` y `observed_to`: cuándo pasó. El servidor añade `received_at`: cuándo llegó.
- `payload`: campos tipados por capacidad. Los que sobran se ignoran. Solo admiten texto libre los eventos con origen
  `declared` (una nota), y ese texto pasa por `SensitiveInformationFilter`.

### 5.4 Lo que hace el servidor

- Valida con los lectores comunes (`remote_devices.py` y los nuevos por capacidad), escribe la frase, aplica un ritmo
  compartido por cuenta y capacidad (`Throttle`) y guarda **un recuerdo por episodio**.
- **Idempotencia:** guarda el `event_id` de cada evento aceptado durante 30 días **(propuesta)**, sellado. Si llega otra
  vez, responde `200 {"saved": true, "duplicate": true, "memory_id": …}` sin guardar nada más.
- **Procedencia:** el autor es el nombre del dispositivo. La fuente dice «observado por <dispositivo>», «dicho por
  Enrique en <dispositivo>» o «inferido por <modelo> en <dispositivo>». El origen va también como campo de la
  metadata del recuerdo, como ya hace `photo` **[verificar que el núcleo guarda un campo nuevo sin tocar el esquema]**.
- Respuestas: `200`, `400` si los datos no valen, `401` con `WWW-Authenticate: Bearer error="invalid_token"`, `403`
  por ámbito o capacidad, y `429` por ritmo. Sin CORS.
- Ruta versionada: `/api/devices/v1/events`. Un cambio incompatible es `v2`, y `v1` sigue funcionando mientras haya
  dispositivos que la usen.

## 6. Fases

Orden de `dd825978`. Una rama y un PR por fase, CI y prueba real en Windows, y una puerta independiente para cada
fase. El orden es una recomendación: no sustituye las decisiones de la §10 ni autoriza a implementar ni a fusionar.

### Fase 0: poner al día la extensión de Claude Desktop

**Estado (28/11/2026): hecha en la rama `feat/desktop-f0-vault-extension-20261128`, opción A (decisión de Enrique del
28/11/2026).** Se queda con el mismo `name` (`infinito-memory`) y el mismo archivo (`infinito-memory.mcpb`), para que
Claude Desktop la actualice en su sitio y el enlace del README siga valiendo. Ahora se muestra como «INFINITO Vault»,
con la versión 0.8.0 en el manifiesto, `mcpb/pyproject.toml` y `pyproject.toml` (`build.py` exige que coincidan). La
carpeta configurable pasa a ser `vault_folder`. Lo que guardaba la 0.7.0 se queda en `~/.infinito/memory.db`; no hay
una forma automática de pasarlo a la cuenta, porque `copy-local` solo sirve para las direcciones privadas. Falta
publicar la release v0.8.0 después de fusionar: hasta entonces, el enlace del README sigue dando la 0.7.0.

Es independiente del resto y pequeña. Se puede hacer antes que las demás o a la vez.

**Problema.** La sección «Claude Desktop, one click» del README enlaza el `infinito-memory.mcpb` de la release v0.7.0
(6/11/2026). Esa extensión guarda la memoria **en local**, en `~/.infinito/memory.db`, separada de la cuenta de
infinitomemory.com: quien la instale hoy crea sin querer una segunda memoria, distinta de la que usa con ChatGPT o con
Claude web. Además, el manifiesto dice 0.7.0 y su `homepage` es el demo antiguo (`demo.178-105-103-4.sslip.io`).

**Alcance (opción A, la recomendada en P-1): una extensión solo de bóveda.** La bóveda local sigue separada, en el PC.

- `mcpb/manifest.json`: `INFINITO_LOCAL_MEMORY=off` en `env`, solo las herramientas `vault_*` en `tools`, textos
  nuevos (la memoria llega por el conector y la bóveda se queda en el PC) y `homepage` = `https://infinitomemory.com`.
- `mcpb/check_bundle.py`: espera solo las herramientas de la bóveda y comprueba que `remember` no existe.
- README: primero, conectar Claude Desktop a `https://infinitomemory.com/mcp` como conector personalizado
  (**[verificar]** la ruta del menú y qué planes de Claude lo permiten); después, la extensión para la bóveda. Lo mismo
  en el apartado «Keep the vault on your computer» de `remote/README.md`.
- El paquete ya lleva `pykeepass` (`mcpb/pyproject.toml`), que el modo solo bóveda necesita: sin él, `mcp_server.py`
  sale con error.

**Opción B:** retirar la extensión del README y de las releases, y dejar el modo local documentado para quien quiera
usarlo sin servidor.

**Puerta:** la CI de `desktop-extension.yml` en verde y una release nueva. Enrique instala la `.mcpb` en Claude Desktop
y comprueba que `vault_status` responde por la extensión y `recall` por el conector.

### Fase 1: app base, estado y avisos

**Estado (28/11/2026): construida en la rama `feat/desktop-f1-tray-status-20261128`, con P-2 (`desktop/`) y P-3
(Python solo con la biblioteca estándar) decididas por Enrique.**

- **Archivos.** El paquete `desktop/infinito_desktop`, el lanzador `desktop/infinito-desktop.pyw`, `desktop/README.md`
  (con la tabla de qué lee y qué envía), 15 pruebas en `tests/test_desktop_v3.py` y una de integración en
  `tests/test_remote_devices_api_v3.py`, y `.github/workflows/desktop.yml`.
- **Sesión.** El inicio de sesión común ya está en `src/infinito3/remote_client.py`: `sign_in` y `refresh`, que usan los
  hooks y la app. El motor da su estado como datos en `cognition/status.py` y en `python -m cognition status --json`.
- **Probado en el PC de Enrique** el 28/11/2026, con `--check`: «Copias» en rojo (la última buena es del 07/11/2026),
  «Motor» en ámbar (en pausa desde el 27/11 a las 19:26), servidor, hooks y tareas bien, y «Sesión de la app» en ámbar
  hasta que entre.
- **Cambia respecto a lo escrito abajo:**
  - la parte del servidor ya la hizo la Device API (#199): un token de dispositivo recibe 403 en todo `/mcp`;
  - la app pide solo `infinito:device`, sin capacidades, porque en la F1 no envía nada; la F2 pedirá `note`, con un
    inicio de sesión nuevo;
  - el menú separa «Salir de INFINITO Desktop», que cierra la app, de «Cerrar sesión» en Ajustes, que borra el token y
    lo revoca en el servidor.
- **Queda para la F2:** la cola cifrada, porque nadie la usa antes de la captura rápida.

**Servidor (pequeño): el ámbito de dispositivo desde el registro** (§5.2, invariante 5)

- **Ya hecho en la rama `feat/device-api-v1-20261127`, con la Device API (§5): con el ámbito, `/mcp` responde `403` a
  todo, también a `remember`.** El plan original era este: `infinito:device` en el registro y en los tokens. Con ese
  ámbito, `/mcp` solo deja `remember` y responde con el texto
  de `remote_apps.refusal` a todo lo demás.
- Pruebas en `tests/test_remote_apps_v3.py` o en un archivo nuevo: un token con el ámbito no puede llamar a la bóveda,
  `forget`, `recall`, `task_*` ni `guideline_propose`, sea cual sea su nivel; un token sin el ámbito se comporta como
  hoy.
- El cambio general de `DEFAULT = "all"` para todas las apps es otra decisión (P-8), fuera de este dossier.

**App**

- **Icono en la bandeja,** verde, ámbar o rojo, con un menú: Estado, Captura rápida (F2), Qué recojo y qué envío,
  Pausa, Abrir mi memoria, Ajustes y Salir. Una sola instancia.
- **Inicio de sesión** con el flujo común de la §4, `client_name` «INFINITO Desktop en <equipo>», el ámbito
  `infinito:device` y el token cifrado con DPAPI. Un único refresco a la vez: el refresco rota, y dos a la vez dejan la
  app fuera.
- **Comprobaciones.** Cada una es una función pura que recibe rutas y devuelve `bien`, `aviso`, `fallo` o `no lo sé`,
  con una frase y un enlace:
  1. **Sesión de la app:** el refresco funciona, al arrancar y cada pocas horas.
  2. **Servidor:** `GET /.well-known/oauth-authorization-server` responde.
  3. **Hooks de Claude Code:** existe `~/.infinito/remote-session.json` (sin leer el token) y `~/.claude/settings.json`
     tiene los hooks `infinito-remote context` y `checkpoint`.
  4. **Motor cognitivo:** reutilizar `python -m cognition status` en vez de leer sus archivos por otra vía (añadirle
     salida JSON si no la tiene **[verificar]**). Debe dar: en pausa y desde cuándo (`PAUSED`); si la última
     revisión se juzgó (`review_end` con `judged`); el último error del proveedor, si consta (`insufficient_quota`); y
     el gasto frente al tope.
  5. **Copias:** la última línea de `pull.log`. `FAILED` o «más de 3 días» es un fallo, con la fecha de la última copia
     buena.
  6. **Tareas programadas:** `schtasks /query /v /fo csv` de las tareas «INFINITO*». Una tarea deshabilitada o con un
     último resultado distinto de 0 es un aviso. Solo lee.
- **Avisos:** una notificación (`cognition/notify.py`) solo cuando una comprobación empeora, y como mucho una al día por
  comprobación. El clic abre la página o la carpeta que toca.
- **Cola cifrada** (la usa la F2): archivo cifrado con DPAPI, con un máximo de 200 elementos y 7 días **(propuesta,
  P-11)**, reintentos con espera creciente, y borrado al «Salir», al desconectar la app o al caducar. Nunca guarda
  secretos: antes de encolar, lo que lleva texto pasa en local por `SensitiveInformationFilter`, y lo que el filtro
  rechazaría no se encola. Nunca guarda lecturas crudas.
- **Panel «Qué recojo y qué envío»:** cada sensor con su interruptor (en la F1 todavía no hay ninguno), la pausa
  general, lo que espera en la cola y lo último que se mandó, con la respuesta del servidor. Se guarda cifrado y se
  borra con «Salir».
- **Presupuestos de recursos** en Ajustes: CPU, RAM y GPU máximas para el trabajo de fondo, y «nunca con batería».
  Los usa la F5.
- **Actualizaciones:** manuales en el MVP. Si más adelante hay actualizador (P-10), solo avisa. Enrique confirma, y la
  app descarga únicamente la release oficial y comprueba su SHA-256 contra el que publica `infinitomemory.com` antes de
  reemplazarse. Nada de actualizaciones de otros sitios ni sin verificar.
- **Arrancar con Windows:** un interruptor en Ajustes (`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`), apagado
  por defecto.
- `python -m infinito_desktop --check` imprime el estado en JSON y sale, para la CI y para depurar.
- **CI:** `.github/workflows/desktop.yml`. Las pruebas de Win32 (DPAPI, `schtasks`) en `windows-2022`, PyInstaller, el
  ejecutable como artefacto y su SHA-256.

**Pruebas** (`tests/test_desktop_v3.py`), con datos inventados que reproducen los casos reales:

- un `pull.log` con la línea `FAILED … unexpected entry` → fallo;
- una revisión con `judged: false` → aviso;
- `PAUSED` presente → aviso con la fecha;
- archivos ausentes o corruptos → «no lo sé», nunca «bien»;
- la cola: respeta sus límites, se borra al salir y nunca guarda un texto que el filtro rechaza;
- la app no tiene ningún camino a `forget` ni a `vault_*`: una sola función llama al servidor, con una lista cerrada;
- el token no aparece nunca en los registros, en el panel ni en `--check`.

**Puerta (Enrique, en su PC)**

1. Descargar el artefacto de Actions, comprobar su SHA-256 y abrirlo.
2. Iniciar sesión y ver la app en Apps del portal.
3. Si las copias y el motor siguen como el 27/11, ver «Copias» en rojo y «Motor en pausa» en ámbar, y recibir **una**
   notificación por cada uno.
4. Arreglar uno (por ejemplo, reanudar el motor) y ver que se pone en verde sin reiniciar la app.
5. Anotar la CPU y la memoria en reposo que muestra el Administrador de tareas.

### Fase 2: captura rápida

**Estado (28/11/2026): construida en la rama `feat/desktop-f2-quick-capture-20261128`.**

- **Atajo:** `Ctrl+Alt+I`, la propuesta de la P-4. En el PC de Enrique está libre. Se puede cambiar la letra en
  Ajustes, y si otro programa tiene el atajo la app lo dice.
- **Ventana:** texto, tipo y proyecto. La lista de proyectos la guarda la app.
- **Filtro de secretos** (`safety.py`) en el PC antes de encolar.
- **Cola cifrada con DPAPI:** 200 notas, 7 días y reintentos de 1 minuto a 1 hora.
- **Envío** como evento `note` de la Device API (#199), con `event_id`: un reenvío nunca duplica.
- **Servidor:** la nota admite ahora `project`.
- **Sesión:** la app pide `infinito:device:note` al entrar. Una sesión de la F1 pide volver a entrar, y la nueva
  revoca la anterior.
- **Icono:** las ventanas llevan el de INFINITO en vez de la pluma de Tk.
- **La cola cifrada**, que el dossier ponía en la F1, entra aquí.
- **Pendiente** de la auditoría del #199 (`faa9947f`): entre guardar el recuerdo y apuntar el `event_id` en el
  servidor no hay una transacción. Si el servidor cae justo entre las dos, un reintento podría duplicar la nota. Es
  raro, pero es la garantía de esta fase y hay que arreglarlo en el servidor.

**Alcance**

- **Atajo global** (P-4), que abre una ventana pequeña. Con `Win+H`, el dictado de Windows debería funcionar en el campo
  de texto **[verificar en `tkinter`]**.
- **Campos:** el texto; el tipo (dato, decisión, pendiente, propuesta…, los `kind` que ya acepta `remember`); y el
  proyecto, opcional, elegido de una lista que Enrique guarda en la app (el ámbito de dispositivo no lee la memoria).
- **Guardar** manda un evento `note` (origen `declared`) a la Device API (§5), que ya existe. El servidor pone la
  fuente «dicho por la persona en INFINITO Desktop en <equipo> (nota)» y la hora del evento. Antes de construir la API
  el plan era llamar a `remember` por `/mcp`, y eso ya no vale: un token de dispositivo recibe `403` en `/mcp`. La ventana muestra la respuesta del servidor tal cual: «Remembered as …» o el rechazo
  del filtro.
- **Sin red:** la nota va a la cola cifrada con su `event_id` y se reenvía sola cuando vuelve la red. El servidor
  responde `duplicate` a lo que ya guardó, así que no hay casos dudosos.

**Puerta**

- Guardar tres capturas, una de ellas dictada, y una sin red que se envía al volver.
- Una con un número de tarjeta de prueba (`4111 1111 1111 1111`) se rechaza en local y no entra en la cola.
- Desde ChatGPT, `recall` las encuentra, con «INFINITO Desktop» como autor y sin duplicados.

### Fase 3: INFINITO Device API v1, compartida con el iPhone

**Servidor** (`src/infinito3`): el contrato de la §5. **Ya construido, con las capacidades del teléfono y `note`, en la
rama `feat/device-api-v1-20261127`** (fase 0 del iPhone). A esta fase le queda añadir las capacidades del PC
(`pc.*`) con sus lectores y frases, y sus pruebas. Esto era lo previsto:

- `POST /api/devices/v1/events`, con el ámbito, las capacidades, la validación, la idempotencia, la procedencia y el
  ritmo.
- El registro de dispositivos con capacidades y última actividad, en el portal.
- Los lectores y las frases de `remote_devices.py` se reutilizan, no se copian; los `Throttle` salen de
  `remote_memory.routes` a un sitio común, como pide el dossier del iPhone.
- Pruebas en `tests/test_remote_devices_api_v3.py`:
  - token válido, caducado, de otra cuenta y sin el ámbito;
  - una capacidad no declarada o apagada da `403`;
  - el mismo `event_id` dos veces da un solo recuerdo, también si la primera respuesta se perdió;
  - `received_at` frente a `observed_*`;
  - el origen y la fuente de cada recuerdo;
  - el texto libre solo en `declared`, y filtrado;
  - el ritmo compartido con la página Dispositivos;
  - ningún evento lanza una acción.

**App:** la captura rápida pasa a ser un evento `note` con `event_id`, y la cola lo reenvía sin miedo a duplicar.

**iPhone:** su fase 0 ya adopta este contrato en vez de las rutas por acción (revisión 1 de su dossier). Quien llegue
antes lo construye y el otro lo reutiliza (P-6).

**Puerta:** la CI en verde, Enrique fusiona y se despliega. Con la red cortada a mitad de un envío, la nota acaba
guardada una sola vez. El portal muestra el PC con sus capacidades y su última actividad.

### Fase 4: sensores pasivos por consentimiento

Cada sensor es un módulo con su interruptor (apagado), su capacidad en el contrato y su puerta propia. Se empieza por
los tres primeros.

1. **Actividad e inactividad** (`pc.activity`). `GetLastInputInfo` cada minuto **(propuesta)**: solo cuánto tiempo
   lleva el PC sin usar, nunca qué se pulsa. Episodios de 15 minutos o más, uniendo huecos de menos de 5
   **(propuesta)**: «2 h 10 min usando el PC, de 10:05 a 12:15».
2. **Proyectos autorizados.** El programa en primer plano y su título se leen **en local** y se comparan con reglas que
   escribe Enrique, por ejemplo: «título que contiene principiodelTodo → INFINITO». Solo sale la etiqueta; lo que no
   casa con ninguna regla no se registra.
3. **Recursos** (`pc.resources`). CPU, RAM y GPU disponibles, resumidas por episodio, por ejemplo «el PC estuvo
   ocupado al 90 % de 22:00 a 23:30», nunca muestras sueltas. También alimentan el planificador de la F5.

Después, cada uno con su decisión:

4. **Aplicaciones usadas:** solo las de una lista que Enrique aprueba, por su nombre legible y sin rutas, agregadas
   por episodio.
5. **Encendido, suspensión y apagado** (`pc.power`): de los avisos de energía de Windows (`WM_POWERBROADCAST`) o de su
   registro de eventos **[verificar la fuente]**.
6. **Errores de servicios y procesos autorizados** (`pc.errors`): primero los de INFINITO (tareas programadas, motor,
   copias), y solo después los servicios que Enrique añada. Del registro de eventos de Windows solo las fuentes
   elegidas, resumidas sin rutas ni nombres de usuario.

- Un botón «No registrar durante una hora», y la pausa general.
- Todo pasa por el panel «Qué recojo y qué envío», y Enrique puede borrar un episodio desde el portal.

**Puerta, por sensor:**

- **Precisión:** contra un registro a mano de Enrique durante 3 días **(propuesta)**.
- **Consumo:** CPU y memoria medidas en el Administrador de tareas.
- **Duplicados:** cero en una semana.
- **Utilidad:** Enrique marca en el panel si cada episodio le sirve.

Un sensor que no pasa se apaga sin parar a los demás.

### Fase 5 (investigación): nodo cognitivo local

**Qué propone `d99e9389`.** Modelos ligeros locales que clasifiquen en incremental los recuerdos nuevos: etiquetas,
relaciones, agrupaciones y ramas jerárquicas, sin reetiquetar todo cada día, para la búsqueda piramidal (`3e18ffc8`).
Los modelos potentes, en la nube, siguen con la investigación, la reflexión, el razonamiento y las propuestas, que es
lo que ya hace el motor. Desktop es el nodo local.

**Cómo, sin duplicar el motor.**

- El trabajo lo hace `cognition/`, con sus contratos y su estado, como una variante separada. Desktop no crea una
  segunda memoria ni un segundo motor: es el **planificador de recursos**.
- El planificador comprueba si Enrique está usando el PC (F4), la CPU, la GPU y la RAM libres, la batería y si hay
  trabajo pendiente. Con eso publica una ventana de recursos (por ejemplo, `~/.infinito/desktop/resources.json` con
  «libre hasta», los presupuestos y la carga, **(propuesta)**) que el `tick` del motor consulta antes de lanzar trabajo
  local. Cuando Enrique vuelve o sube la carga, la ventana se cierra y el motor pausa o cancela lo que tenga en marcha.
  No se usa `PAUSED`, que es la pausa de Enrique.
- Los modelos locales corren en Ollama, que ya está en el PC.
- Los derivados (etiquetas, relaciones, ramas) se guardan versionados, con origen `inferred`, el modelo y las fuentes.
  Nunca cambian los originales, respetan los permisos y se borran si se olvida su fuente, como `drop_citing`. Hoy el
  servidor no tiene dónde guardarlos **[verificar]**: dónde y cómo es parte del protocolo.
- Si la clasificación falla o no ayuda, la búsqueda vuelve a la recuperación plana de siempre.

**Antes de ejecutar** (AGENTS.md, §0.5):

1. **Un protocolo en `docs/`:** la pregunta, la base de referencia (RAG que guarda todo, al mismo presupuesto de
   tokens), la puerta y qué resultado cambiaría el plan. Datos no usados para ajustar. Es una variante separada de M1,
   M2 y E13-A, sin tocar sus protocolos congelados.
2. **La evidencia previa:**
   - M1 (PR #192): con estructuras descubiertas, «no pasa nada y la recuperación plana se queda».
   - E13-A (PR #79): la consolidación adaptativa no tuvo ventaja suficiente.
   - M2 (PR #191): aún sin resultados.

   El protocolo tiene que decir qué hace distinto esta variante para esperar otro resultado.
3. **Medir calidad, consumo y utilidad** antes de desplegar:
   - **calidad:** frente a la base RAG;
   - **consumo:** CPU, GPU, energía y temperatura;
   - **interferencia:** cero casos en que el trabajo pesado siga más de 10 segundos después de que Enrique vuelva
     **(propuesta)**;
   - **utilidad:** si la búsqueda encuentra mejor lo que se pedía.

Se relaciona con el motor continuo (`bacf4244`) y con las decisiones `c5e22562` y `3e18ffc8`, y se coordina con quien
haga esa tarea para no hacer dos veces el mismo planificador.

### Fase 6: documentos y repositorios

**Documentos.** Requisito: la subida de archivos de la tarea `6987ed99` (`file_save`, almacenamiento cifrado por
cuenta, borrado y copias). Los documentos no se guardan como recuerdos de 2.000 caracteres.

- Enrique añade carpetas en Ajustes. Solo `.md`, `.txt`, `.pdf` y `.docx`, con un tamaño máximo.
- Exclusiones fijas: `.env`, `*.kdbx`, archivos de claves, `.git` y `node_modules`, más las que añada Enrique.
- Un documento nuevo o cambiado se manda como una versión, no como un recuerdo por cada guardado.
- Lo que se ha mandado se ve en el panel y se puede borrar desde el portal.

**Repositorios Git autorizados** (`pc.repo`). No dependen de `6987ed99` y pueden ir antes.

- La app lee `git log` en local, solo en los repositorios que Enrique añade, y manda episodios: cuántos commits, en
  qué repositorio (con su etiqueta) y entre qué horas.
- Los asuntos de los commits, solo si Enrique lo enciende, y pasando por el filtro.
- Nunca diffs ni contenido de archivos.

**Puerta:** un documento de prueba se recupera entero desde otra IA, y una semana de commits da los episodios
esperados, sin duplicados.

### Fase 7: capturas visuales autorizadas, a mano

**Servidor:** la capacidad `photo` del contrato (§5).

- La imagen y la descripción van en el evento.
- Reutiliza `remote_photos.refusal` y `remote_photos.keep`: 20 MB por imagen y 500 MB por persona (`MAX_BYTES`,
  `MAX_TOTAL`), y la misma invitación (`vault_allowed`).
- Pruebas en `tests/test_remote_devices_api_v3.py`.

**App:** un atajo o el menú captura la pantalla entera o una zona (GDI por `ctypes` y PNG con `zlib`).

- Enseña una vista previa con el aviso «revisa que no salga ninguna contraseña», porque el filtro no lee imágenes.
- Pide la descripción en palabras de Enrique y su confirmación, y solo entonces la manda.
- Nunca captura sin que Enrique lo pida.

**Puerta:** mandar una captura y verla desde Claude con `photo_show`.

## 7. Pruebas comunes

- **Núcleo:** `tests/test_*v3.py` en verde con cada cambio en `src/infinito3` (login común, ámbito de dispositivo,
  Device API).
- **App:** la lógica pura (comprobaciones, cola, episodios y cuerpos de los eventos) se prueba en Linux, en la CI del
  núcleo. Lo que toca Win32 se prueba en `windows-2022`.
- **Privacidad:** cada PR de la app incluye una tabla de qué sale del PC, a dónde, cuándo y con qué origen.
- **Límite de llamadas:** una prueba falla si la app puede llamar a algo fuera de su lista cerrada, y otra si el
  servidor deja a un token de dispositivo hacer algo fuera de su ámbito.
- **Idempotencia:** desde la F3, cada tipo de evento tiene su prueba de reenvío.

## 8. Riesgos

- **Deriva hacia la vigilancia.** Con la observación pasiva en la F4 es el riesgo mayor. Por eso: §3, todo apagado,
  la lista de fuera, el panel y una puerta por sensor.
- **Un token de dispositivo robado** podría inyectar observaciones falsas. Lo limitan el ámbito, las capacidades, el
  ritmo y la revocación desde el portal, y la procedencia dice de qué dispositivo vino cada recuerdo.
- **Calor y batería.** El portátil ya tuvo el ventilador de la gráfica al máximo (`b1aab805`). La F5 nunca va con
  batería y se mide su temperatura.
- **F5 con evidencia previa negativa** (M1, E13-A): puede no mejorar nada. Por eso es investigación, con puerta.
- **Ejecutable sin firmar:** SmartScreen avisa, y algunos antivirus marcan los ejecutables de PyInstaller, sobre todo
  los de `--onefile` **[verificar]**. Por eso, `--onedir` y el SHA-256 publicado.
- **Atajo ocupado:** si otro programa tiene el mismo atajo, `RegisterHotKey` falla. La app tiene que decirlo y dejar
  elegir otro.
- **Notificaciones silenciadas:** el modo «No molestar» de Windows oculta los avisos. El color del icono sigue a la
  vista.
- **Formatos que cambian:** `pull.log`, la salida de `cognition status` y `~/.claude/settings.json` pueden cambiar.
  Cada comprobación falla con «no lo sé» y nunca hace caer la app.
- **Otras sesiones en texto plano:** el `session.json` del motor y el `remote-session.json` de los hooks guardan su
  refresco sin cifrar. Esta app no lo empeora; cifrarlos es un cambio aparte.
- **Prioridad:** es una ampliación del producto, y el hito del núcleo sigue siendo cerrar la distancia con la base de
  referencia RAG (AGENTS.md).

## 9. Estimación (sin medir)

- **Fase 0:** una sesión corta del agente.
- **Fase 1:** dos o tres sesiones. Lo que más cuesta es el ámbito en el servidor, las comprobaciones y el icono de la
  bandeja con `ctypes`.
- **Fase 2:** una sesión.
- **Fase 3:** dos sesiones, la mayor parte en el servidor, compartidas con el iPhone.
- **Fase 4:** una sesión por sensor, más los días de la puerta.
- **Fase 5:** primero el protocolo; el resto depende de él.
- **Fases 6 y 7:** una o dos sesiones cada una. Los documentos dependen de `6987ed99`.

A diferencia del iPhone, cada ciclo de prueba es descargar un artefacto y abrirlo en el mismo PC: sin TestFlight, sin
Mac y sin la revisión de una tienda.

## 10. Decisiones abiertas (para Enrique, antes de la fase 1)

- **P-1. Extensión de Claude Desktop.** Recomendado: A, solo bóveda, con la memoria por el conector. Alternativa: B,
  retirarla.
- **P-2. Dónde vive la app.** Recomendado: `desktop/`, dentro de `principiodelTodo`, para que un PR pueda tocar
  servidor y app a la vez. Alternativa: un repositorio aparte.
- **P-3. Tecnología.** Recomendado: Python con solo la biblioteca estándar. Alternativas: `pystray` con Pillow, o
  Tauri o Electron.
- **P-4. Atajo de la captura rápida.** Propuesta: `Ctrl+Alt+I` **[verificar que no choca con nada en su PC]**.
- **P-5. Capturas visuales a mano (F7):** si se hacen, y cuándo.
- **P-6. Contrato común con el iPhone.** El dossier del iPhone (PR #197) ya adopta la Device API v1 en su fase 0, a
  petición de Enrique el 27/11/2026. Queda quién la construye primero (es también la P-8 de ese dossier).
- **P-7. Prioridad** frente al hito RAG, la app del iPhone (PR #197) y las acciones de Teams (PR #190).
- **P-8. Ámbito de dispositivo.** El nombre (`infinito:device`) y qué permite. Aparte: si el `DEFAULT = "all"` de las
  apps nuevas debe cambiar para todas.
- **P-9. Nodo cognitivo local (F5).** Si se escribe su protocolo, con qué modelo local, y si puede correr solo con el
  PC enchufado (recomendado) o nunca de noche.
- **P-10. Actualizaciones.** Manuales (MVP) o un actualizador que solo avisa y verifica el SHA-256.
- **P-11. Cola.** Sus límites (propuesta: 200 elementos y 7 días) y qué se hace con lo pendiente al salir (propuesta:
  borrarlo).

## 11. Qué cambiaría el plan

Cada función tiene su puerta. Que una no sirva no para las demás.

- **Avisos (F1):** si en dos semanas son más ruido que ayuda (más de uno al día que no pide hacer nada), se ajustan
  antes de seguir.
- **Captura rápida (F2):** se mide por duplicados (cero) y por rechazos correctos del filtro. Cuántas capturas se
  guardan no decide nada sobre los sensores ni sobre la F5.
- **Sensores (F4):** cada uno por su precisión, su consumo, sus duplicados y la utilidad que marque Enrique. El que no
  pasa se apaga solo.
- **Nodo cognitivo (F5):** decide su protocolo. Si no supera la base RAG, se queda la recuperación plana.
- **Contrato (F3):** si el iPhone necesita algo que el contrato no cubre, se versiona (`v2`) en vez de abrir una ruta
  aparte.
- **Inicio de sesión:** si el registro dinámico rechaza al cliente o la vuelta a `127.0.0.1` falla desde el ejecutable
  empaquetado, primero se arregla eso.
- **El entorno:** si Claude Desktop, ChatGPT o Windows sacan su propia forma de mandar cosas a una memoria por MCP, se
  compara antes de seguir con la F2.
