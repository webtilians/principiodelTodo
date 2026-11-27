# INFINITO Desktop para Windows: dossier de implementación del MVP

27/11/2026 · Hilo (Claude Code), a petición de Enrique · Estado: **propuesta, nada implementado**.

Este dossier es para el agente que construya la primera app de escritorio de INFINITO. Parte de la idea que Enrique
tuvo en Talent Land y guardó el 27/11/2026 (recuerdo `7bef0c68`: «INFINITO tiene que estar conectado al PC de alguna
forma; también podría sacar información del PC») y de la valoración de Hilo del mismo día (`7ea7c944`). Va en paralelo
al dossier del iPhone (`docs/INFINITO_APP_IPHONE_DOSSIER_2026-11-27.md`, PR #197, aún sin fusionar) y comparte sus
invariantes. Lo marcado **[verificar]** no se ha comprobado y hay que mirarlo antes de dar por buena esa parte.

## 0. Antes de empezar (para el agente)

1. Lee [README.md](../README.md), [AGENTS.md](../AGENTS.md), este dossier entero,
   [cognition/infinito.py](../cognition/infinito.py), [cognition/notify.py](../cognition/notify.py),
   [src/infinito3/remote_apps.py](../src/infinito3/remote_apps.py) y, del dossier del iPhone, las §3 y §5 (fase 0).
2. Una rama desde `master` y un PR contra `master` por fase. Enrique fusiona. No se empieza una fase sin la puerta de
   la anterior.
3. **Solo Windows.** Enrique trabaja en un portátil con Windows 11 y no tiene Mac. Los ejecutables se compilan en
   GitHub Actions; las pruebas en el PC real las hace Enrique, y cada fase termina con una lista de comprobación para
   él en la descripción del PR.
4. Antes de la fase 1, que Enrique responda a las decisiones abiertas de la §9.
5. Esto es producto, no investigación: no hace falta protocolo experimental, pero sí pruebas. `src/infinito3` sigue
   siendo solo biblioteca estándar y `python -m pytest tests/test_*v3.py` sigue en verde con cada cambio. La app vive
   fuera del núcleo, como `cognition/` y `mcpb/`.
6. No leas ni muestres el contenido de `~/.infinito/cognition/session.json`, `~/.infinito/remote-session.json` ni de
   la bóveda: son sesiones y secretos de Enrique.

## 1. Para qué, y para qué no

**Problema.** INFINITO ya está en el PC de Enrique, pero en piezas sueltas, y cuando una falla nadie se entera:

| Pieza en el PC | Cómo avisa hoy |
|---|---|
| Hooks de Claude Code (`infinito-remote context` y `checkpoint`) | No avisa: sin sesión o sin respuesta del servidor no añade nada, a propósito |
| Bóveda local (`infinito-mcp` con `INFINITO_LOCAL_MEMORY=off`) | Solo cuando una IA la usa |
| Motor cognitivo: tareas «INFINITO Motor cognitivo» y «INFINITO Motor cognitivo (noche)» | Una notificación de Windows cuando llega una revisión (`cognition/notify.py`) |
| Copias: tarea «INFINITO copias de seguridad», cada día a las 21:00 | Una línea en `~/.infinito/copias-servidor/pull.log` |

Dos casos de este mes:

- **Copias.** La última buena es del 7/11/2026 (`d4782ddf`). Desde entonces cada intento falla, y el único aviso es una
  línea en `pull.log` que nadie abre. La del 26/11 a las 22:17 dice
  `FAILED: … is NOT usable: unexpected entry 'accounts/enrikelr/cognition.sealed'`.
- **Motor.** Del 25 al 27/11 el supervisor (qwen4-max) respondió `403 insufficient_quota`, y la notificación decía
  «ha terminado la revisión del día» (`07c2bbfe`, `3cdd32da`). Lo arregló el PR #196, pero se descubrió tres días
  tarde y mirando a mano.

Además, lo que pasa en el PC solo llega a la memoria si una IA decide guardarlo.

**Objetivo.** Una app en la bandeja del sistema que:

1. diga de un vistazo si INFINITO funciona en este PC, y avise cuando algo deja de funcionar;
2. permita mandar algo a la memoria en dos segundos, desde cualquier sitio, sin abrir una IA;
3. más adelante, y con permiso, convierta en recuerdos lo que Enrique elija del PC.

Todo con las mismas garantías que la página Dispositivos y la app del iPhone. Va en la línea de la decisión del
26/11/2026 de conectar todos los dispositivos a la memoria (`69497532`).

**Fuera del MVP, y por qué.**

- **Capturas de pantalla automáticas o periódicas, portapapeles, historial del navegador, teclado, micrófono abierto,
  correo y mensajería del PC.** Arrastran contraseñas, claves y datos de terceros, y la regla de AGENTS.md es no
  guardar secretos nunca. Microsoft tuvo que retrasar Recall en 2024 por las críticas de privacidad y acabó haciéndolo
  opcional.
- **Indexar el disco.** Solo carpetas que Enrique elija, y solo cuando exista la subida de archivos (fase 4).
- **Un chat.** Enrique lo descartó el 11/11/2026 (`1d1f907b`).
- **macOS y Linux.** No hay Mac; Linux, si alguien lo pide.
- **Distribución pública firmada.** El MVP es para uso propio; firmar el ejecutable se decide después.
- **Teams.** Un agente de escritorio para empresas es otro producto, y según AGENTS.md en Teams solo entran funciones
  estables y probadas.

## 2. Lo que ya existe (no rehacerlo)

| Pieza | Dónde | Estado |
|---|---|---|
| OAuth con registro dinámico, PKCE y vuelta a `127.0.0.1`, con nombre de app propio | `cognition/infinito.py` (`Infinito.login`, `_token`); `src/infinito3/remote_client.py` (`login`, con el nombre fijo «INFINITO on <equipo>») | En uso: el motor y los hooks de este PC |
| Llamar a una herramienta MCP con el token | `cognition/infinito.py` (`Infinito.call`) | En uso por el motor |
| Niveles por app (`all`, `add`, `read`); una app nueva empieza en `all` | `src/infinito3/remote_apps.py` (`LEVELS`, `DEFAULT`, `allowed`) | En producción |
| Filtro de secretos al guardar | `src/infinito3/engine.py` (`SensitiveInformationFilter`), por donde pasa `remember` | En producción |
| Notificación de Windows sin dependencias, cuyo clic abre una página | `cognition/notify.py` (`toast`, por PowerShell y con su identidad) | Comprobada por Enrique el 11/11/2026 (`3482fe19`) |
| Estado del motor | `~/.infinito/cognition/`: `PAUSED`, `diary.jsonl` (entradas `review_end` con `outcome` y `judged`), `ledger.json` y `reports/`; se lee en `cognition/cycle.py` (`last_review`) | En el PC. En pausa desde el 27/11 a las 19:26 (`4329fdc3`) |
| Estado de las copias | Tarea «INFINITO copias de seguridad» y `~/.infinito/copias-servidor/pull.log` | Fallan desde el 7/11 |
| Lectores, frases, ritmo (`Throttle`) y trayecto sellado | `src/infinito3/remote_devices.py` | En master |
| API de dispositivos con token, `POST /api/devices/<acción>` | Fase 0 del dossier del iPhone (PR #197) | Propuesta, sin construir |
| `photo_save` | `src/infinito3/remote.py`. Recibe un `download_url` público HTTPS que descarga el servidor, y pide la invitación a la bóveda (`vault_allowed`) | En producción. **No sirve para un archivo local** |
| Subida de archivos (`file_save`, `file_search`, `file_read`) | Tarea aprobada `6987ed99` (20/11/2026) | Sin construir |
| Ejecutable de Windows con PyInstaller en Actions | `webtilians/lain`, `.github/workflows/beta01-windows.yml` (`windows-2022`, `pyinstaller==6.16.0`, `--onedir` y `--onefile --windowed`) | En uso en LAIN |
| Extensión de Claude Desktop (`.mcpb`) | `mcpb/` y `.github/workflows/desktop-extension.yml`, adjunta a la release v0.7.0 (6/11/2026) | Publicada pero desfasada: ver la fase 0 |

## 3. Invariantes (lo que ninguna fase puede romper)

1. **El servidor escribe las palabras de lo que mide el PC.** La app manda hechos medidos, y el servidor los valida y
   los redacta, como en Dispositivos. La excepción es lo que Enrique escribe o dicta a propósito (fase 2), que va como
   `remember` y pasa por `SensitiveInformationFilter` en el servidor.
2. **Las lecturas crudas no salen del PC.** Los títulos de ventana, las rutas, los nombres de programa y los contenidos
   se leen en local; solo sale lo que Enrique ha elegido mandar.
3. **Un recuerdo por episodio,** con límites de ritmo.
4. **Solo habla con el servidor de INFINITO de la persona** (`infinitomemory.com`). Sin analítica, telemetría, informes
   de fallos de terceros ni actualizaciones automáticas desde otro sitio.
5. **El token se guarda cifrado con DPAPI** (la protección de datos de Windows ligada al usuario, `CryptProtectData`),
   nunca en registros ni en texto plano. «Salir» lo borra; desconectar la app en Apps del portal revoca sus tokens.
6. **Todo empieza apagado,** salvo el estado local de la fase 1, que solo lee archivos de INFINITO en este PC. Cada
   sentido tiene su interruptor y explica qué manda.
7. **Una línea propia en Apps:** «INFINITO Desktop en <equipo>», separada del motor y de los hooks. El nivel
   recomendado es «Leer y añadir». La app no llama nunca a `forget` ni a la bóveda, y sus recuerdos llevan su nombre
   como autor, para que una IA distinga lo que dijo Enrique de lo que vio el PC.
8. **No molesta.** En reposo no usa CPU de forma apreciable, sus temporizadores son de minutos y no de segundos, y
   nunca bloquea el PC ni otra app si el servidor no responde.
9. **Si algo falla, lo dice.** Un estado que no puede leer se muestra como «no lo sé», nunca como «bien». Es la lección
   del motor del 24 al 27/11 y de la tarea de copias que terminaba con código 0 mientras rechazaba las copias.

## 4. Arquitectura

```
PC con Windows 11
  INFINITO Desktop  (Python, bandeja del sistema, un solo proceso por usuario)
   ├─ Estado (solo lee, en local): ~/.infinito/cognition/{PAUSED, diary.jsonl, ledger.json}
   │                                ~/.infinito/copias-servidor/pull.log, tareas programadas «INFINITO*»
   ├─ Captura rápida (atajo global) ──────────► /mcp  remember            (ya existe)
   ├─ Captura de pantalla a mano (opcional) ──► /api/photos               (fase 3, nuevo)
   ├─ Carpetas elegidas (fuera del MVP) ──────► file_save                 (tarea 6987ed99)
   └─ Actividad resumida (fuera del MVP) ─────► /api/devices/activity     (fase 5, nuevo)
          HTTPS · Authorization: Bearer <acceso de «INFINITO Desktop en <equipo>»>
```

Las fases 1 y 2 **no necesitan cambios en el servidor**: el estado es local y la captura rápida usa `remember` por
`/mcp`, como ya hace el motor.

**Tecnología recomendada: Python con solo la biblioteca estándar.**

- `ctypes` para el icono de la bandeja (`Shell_NotifyIconW`), el atajo global (`RegisterHotKey`), DPAPI y una sola
  instancia (`CreateMutexW`).
- `tkinter` para la ventana de captura y los ajustes.
- `winreg` para arrancar al iniciar sesión, si Enrique lo enciende.
- La notificación de `cognition/notify.py`.

Motivos: es como `remote_client.py` y `cognition/`, el código de inicio de sesión ya existe, y un producto con doble
licencia (AGPL y comercial, `1fcd299`) no suma dependencias cuyas licencias haya que revisar. Las alternativas están
en la decisión P-3: `pystray` con Pillow (menos código, pero dependencias y licencia **[verificar]**), o Tauri o
Electron si algún día hace falta una interfaz rica.

**Ejecutable:** PyInstaller `--onedir --windowed` en `windows-2022`, como LAIN, subido como artefacto de Actions. Sin
firmar, Windows SmartScreen avisará al abrirlo **[verificar]**; para uso propio basta con «Ejecutar de todas formas».

**Dónde:** `desktop/` en este repositorio (decisión P-2), con el paquete `infinito_desktop`. Las pruebas de la lógica
pura van en `tests/test_desktop_v3.py`, como `tests/test_cognition_v3.py`, para que corran en la CI del núcleo.

**Inicio de sesión:** no copiar el flujo una tercera vez. Sacar a una función común el login con nombre de app y ruta
de sesión, en `src/infinito3/remote_client.py`, que ya es biblioteca estándar. La usa la app y, si se quiere, el
motor. Cambiar `remote_client.py` exige sus pruebas en verde.

## 5. Fases

### Fase 0: poner al día la extensión de Claude Desktop

Es independiente del resto y pequeña. Se puede hacer antes que las demás o a la vez.

**Problema.** La sección «Claude Desktop, one click» del README enlaza el `infinito-memory.mcpb` de la release v0.7.0
(6/11/2026). Esa extensión guarda la memoria **en local**, en `~/.infinito/memory.db`, separada de la cuenta de
infinitomemory.com: quien la instale hoy tiene una memoria distinta de la que usa con ChatGPT o con Claude web. Además,
el manifiesto dice 0.7.0 y su `homepage` es el demo antiguo (`demo.178-105-103-4.sslip.io`).

**Alcance (opción A, la recomendada en P-1): una extensión solo de bóveda.**

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

**Alcance**

- **Icono en la bandeja,** verde, ámbar o rojo, con un menú: Estado, Captura rápida (fase 2), Abrir mi memoria,
  Ajustes y Salir. Una sola instancia.
- **Inicio de sesión** con el flujo común de la §4, `client_name` «INFINITO Desktop en <equipo>» y el token cifrado con
  DPAPI (§3.5). Un único refresco a la vez: el refresco rota, y dos a la vez dejan la app fuera.
- **Comprobaciones.** Cada una es una función pura que recibe rutas y devuelve `bien`, `aviso`, `fallo` o `no lo sé`,
  con una frase y un enlace:
  1. **Sesión de la app:** el refresco funciona, al arrancar y cada pocas horas.
  2. **Servidor:** `GET /.well-known/oauth-authorization-server` responde.
  3. **Hooks de Claude Code:** existe `~/.infinito/remote-session.json` (sin leer el token) y `~/.claude/settings.json`
     tiene los hooks `infinito-remote context` y `checkpoint`.
  4. **Motor cognitivo:** si existe `PAUSED`, «en pausa desde <fecha del archivo>». Si la última `review_end` de
     `diary.jsonl` tiene `judged` falso, aviso. El último error del proveedor, si consta (`insufficient_quota`). El
     gasto frente al tope de `ledger.json` **[verificar los campos]**.
  5. **Copias:** la última línea de `pull.log`. `FAILED` o «más de 3 días» es un fallo, con la fecha de la última copia
     buena.
  6. **Tareas programadas:** `schtasks /query /v /fo csv` de las tareas «INFINITO*». Una tarea deshabilitada o con un
     último resultado distinto de 0 es un aviso. Solo lee.
- **Avisos:** una notificación (`cognition/notify.py`) solo cuando una comprobación empeora, y como mucho una al día por
  comprobación. El clic abre la página o la carpeta que toca.
- **Arrancar con Windows:** un interruptor en Ajustes (`HKCU\Software\Microsoft\Windows\CurrentVersion\Run`), apagado
  por defecto.
- `python -m infinito_desktop --check` imprime el estado en JSON y sale, para la CI y para depurar.
- **CI:** `.github/workflows/desktop.yml`. Las pruebas de Win32 (DPAPI, `schtasks`) en `windows-2022`, PyInstaller y el
  ejecutable como artefacto.

**Pruebas** (`tests/test_desktop_v3.py`), con datos inventados que reproducen los casos reales:

- un `pull.log` con la línea `FAILED … unexpected entry` → fallo;
- un `diary.jsonl` con una `review_end` con `judged: false` → aviso;
- `PAUSED` presente → aviso con la fecha;
- archivos ausentes o corruptos → «no lo sé», nunca «bien»;
- la app no tiene ningún camino a `forget` ni a `vault_*`: una sola función llama a herramientas, con una lista
  cerrada;
- el token no aparece nunca en los registros ni en `--check`.

**Puerta (Enrique, en su PC)**

1. Descargar el artefacto de Actions y abrirlo.
2. Iniciar sesión y poner la app en «Leer y añadir» en Apps.
3. Si las copias y el motor siguen como el 27/11, ver «Copias» en rojo y «Motor en pausa» en ámbar, y recibir **una**
   notificación por cada uno.
4. Arreglar uno (por ejemplo, reanudar el motor) y ver que se pone en verde sin reiniciar la app.
5. Anotar la CPU y la memoria en reposo que muestra el Administrador de tareas.

### Fase 2: captura rápida

**Alcance**

- **Atajo global** (P-4), que abre una ventana pequeña. Con `Win+H`, el dictado de Windows debería funcionar en el campo
  de texto **[verificar en `tkinter`]**.
- **Campos:** el texto; el tipo (dato, decisión, pendiente, propuesta…, los `kind` que ya acepta `remember`); y el
  proyecto, opcional, con sugerencias de `overview`.
- **Guardar** llama a `remember` por `/mcp`, con `source` = «INFINITO Desktop, captura rápida» y la fecha de hoy. La
  ventana muestra la respuesta del servidor tal cual: «Remembered as …» o el rechazo del filtro.
- **Sin red:** el texto se queda en la ventana y la app lo dice. En el MVP no hay cola en disco; si algún día la hay,
  va cifrada con DPAPI.

**Puerta**

- Guardar tres capturas, una de ellas dictada.
- Una con un número de tarjeta de prueba (`4111 1111 1111 1111`) se rechaza.
- Desde ChatGPT, `recall` las encuentra, con «INFINITO Desktop» como autor.

### Fase 3 (opcional en el MVP, P-5): captura de pantalla a mano

**Servidor:** `POST /api/photos` con `Authorization: Bearer`, nivel `add` o `all`, y en el cuerpo la imagen y la
descripción.

- Reutiliza `remote_photos.refusal` y `remote_photos.keep`: 20 MB por imagen y 500 MB por persona (`MAX_BYTES`,
  `MAX_TOTAL`), y la misma invitación (`vault_allowed`).
- Si la fase 0 del iPhone ya está hecha, va junto a `/api/devices`, con el mismo manejo del token y de los errores
  (`401` con `WWW-Authenticate`, `403` por nivel, `429` por ritmo).
- Pruebas en `tests/test_remote_photos_api_v3.py`.

**App:** un atajo o el menú captura la pantalla entera o una zona (GDI por `ctypes` y PNG con `zlib`).

- Enseña una vista previa con el aviso «revisa que no salga ninguna contraseña», porque el filtro no lee imágenes.
- Pide la descripción en palabras de Enrique, y solo entonces la manda.
- Nunca captura sin que Enrique lo pida.

**Puerta:** mandar una captura y verla desde Claude con `photo_show`.

### Fase 4 (fuera del MVP): carpetas elegidas

**Requisito:** que exista la subida de archivos de la tarea `6987ed99` (`file_save`, almacenamiento cifrado por cuenta,
borrado y copias). Los documentos no se guardan como recuerdos de 2.000 caracteres.

- Enrique añade carpetas en Ajustes. Solo `.md`, `.txt`, `.pdf` y `.docx`, con un tamaño máximo.
- Exclusiones fijas: `.env`, `*.kdbx`, archivos de claves, `.git` y `node_modules`.
- Un documento nuevo o cambiado se manda como una versión, no como un recuerdo por cada guardado.
- Lo que se ha mandado se ve en la app y se puede borrar desde el portal.

### Fase 5 (fuera del MVP, con decisión aparte): actividad resumida

- **En local,** cada pocos segundos **[cuantificar en la puerta]**, el programa en primer plano y su título. Se comparan
  **en el PC** con reglas que escribe Enrique, por ejemplo: «título que contiene principiodelTodo → INFINITO». Lo que
  no casa con ninguna regla no se registra.
- **Episodios** de 15 minutos o más, uniendo los huecos de menos de 5. Al cerrar uno, `POST /api/devices/activity` con
  la etiqueta, el inicio y los minutos. El servidor escribe la frase («2 h 10 min en INFINITO en el PC, de 10:05 a
  12:15») y aplica el ritmo.
- Un botón «No registrar durante una hora», y la pausa total en el menú.
- **Riesgo:** hasta las etiquetas dicen mucho de una persona. Se decide después de usar las fases 1 a 3.

## 6. Pruebas comunes

- **Núcleo:** `tests/test_*v3.py` en verde con cada cambio en `src/infinito3` (login común, `/api/photos`,
  `/api/devices/activity`).
- **App:** la lógica pura (comprobaciones, episodios y cuerpos de los envíos) se prueba en Linux, en la CI del núcleo.
  Lo que toca Win32 se prueba en `windows-2022`.
- **Privacidad:** cada PR de la app incluye una tabla de qué sale del PC, a dónde y cuándo.
- **Límite de herramientas:** una prueba falla si la app puede llamar a algo fuera de su lista cerrada (`remember`,
  `overview` y, en la fase 4, `file_save`).

## 7. Riesgos

- **Ejecutable sin firmar:** SmartScreen avisa, y algunos antivirus marcan los ejecutables de PyInstaller, sobre todo
  los de `--onefile` **[verificar]**. Por eso, `--onedir`.
- **Atajo ocupado:** si otro programa tiene el mismo atajo, `RegisterHotKey` falla. La app tiene que decirlo y dejar
  elegir otro.
- **Notificaciones silenciadas:** el modo «No molestar» de Windows oculta los avisos. El color del icono sigue a la
  vista.
- **Formatos que cambian:** `diary.jsonl`, `pull.log` y `~/.claude/settings.json` pueden cambiar. Cada comprobación
  falla con «no lo sé» y nunca hace caer la app.
- **Deriva hacia la vigilancia:** cada fase nueva que lea algo del PC pasa por la §3 y por la lista «Fuera del MVP».
- **Otras sesiones en texto plano:** el `session.json` del motor y el `remote-session.json` de los hooks guardan su
  refresco sin cifrar. Esta app no lo empeora; cifrarlos es un cambio aparte.
- **Prioridad:** es una ampliación del producto, y el hito del núcleo sigue siendo cerrar la distancia con la base de
  referencia RAG (AGENTS.md).

## 8. Estimación (sin medir)

- **Fase 0:** una sesión corta del agente.
- **Fase 1:** dos o tres sesiones; lo que más cuesta son las comprobaciones y el icono de la bandeja con `ctypes`.
- **Fases 2 y 3:** una sesión cada una.
- **Fases 4 y 5:** dependen de la tarea `6987ed99` y de una decisión aparte.

A diferencia del iPhone, cada ciclo de prueba es descargar un artefacto y abrirlo en el mismo PC: sin TestFlight, sin
Mac y sin la revisión de una tienda.

## 9. Decisiones abiertas (para Enrique, antes de la fase 1)

- **P-1. Extensión de Claude Desktop.** Recomendado: A, solo bóveda, con la memoria por el conector. Alternativa: B,
  retirarla.
- **P-2. Dónde vive la app.** Recomendado: `desktop/`, dentro de `principiodelTodo`, para que un PR pueda tocar
  servidor y app a la vez. Alternativa: un repositorio aparte.
- **P-3. Tecnología.** Recomendado: Python con solo la biblioteca estándar. Alternativas: `pystray` con Pillow, o
  Tauri o Electron.
- **P-4. Atajo de la captura rápida.** Propuesta: `Ctrl+Alt+I` **[verificar que no choca con nada en su PC]**.
- **P-5. Captura de pantalla a mano (fase 3):** dentro o fuera del MVP.
- **P-6. Orden con el iPhone.** Si el escritorio va antes, la API con token (fase 0 del iPhone) se construye cuando la
  necesite la fase 3 o la 5, y sirve para los dos.
- **P-7. Prioridad** frente al hito RAG, la app del iPhone (PR #197) y las acciones de Teams (PR #190).

## 10. Qué cambiaría el plan

- Si en dos semanas con la fase 1 los avisos son más ruido que ayuda (más de uno al día que no pide hacer nada), se
  ajustan antes de seguir.
- Si en dos semanas Enrique guarda menos de cinco capturas rápidas (recuerdos con «INFINITO Desktop» como autor), no se
  hacen las fases 4 y 5, y se revisa qué aporta de verdad el PC.
- Si Claude Desktop, ChatGPT o Windows sacan su propia forma de mandar cosas a una memoria por MCP, se compara antes
  de seguir con la fase 2.
- Si el registro dinámico rechaza al cliente, o la vuelta a `127.0.0.1` falla desde el ejecutable empaquetado, primero
  se arregla el inicio de sesión.
