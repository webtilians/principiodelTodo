# INFINITO para iPhone: dossier de implementación del MVP

27/11/2026 · Hilo (Claude Code), a petición de Enrique · Estado: **propuesta. La fase 0 (servidor) está construida en
la rama `feat/device-api-v1-20261127` (28/11/2026), sin fusionar; la app, sin empezar.**

**Revisión 1 (27/11/2026).** Enrique pidió que este dossier adopte el contrato común de dispositivos, **INFINITO Device
API v1**, definido en la §5 del dossier de INFINITO Desktop (`docs/INFINITO_DESKTOP_DOSSIER_2026-11-27.md`, PR #198,
aún sin fusionar; mientras tanto está en la rama `docs/desktop-app-dossier-20261127`). La fase 0 deja las rutas por
acción (`/api/devices/<acción>`) y pasa a ser esa API. Cambian también las fases 1 a 5 (capacidades del teléfono), los
invariantes, las pruebas, los riesgos y las decisiones. Esa §5 es la fuente del contrato; aquí solo se dice cómo lo usa
el teléfono.

Este dossier es para el agente que construya la primera app de INFINITO para iPhone. Parte de la propuesta del
27/11/2026 (recuerdo `4f025901`) y de la valoración de Hilo del mismo día (`e06d1761`). Lo marcado **[verificar]** no
se ha comprobado y hay que mirarlo antes de dar por buena esa parte. Lo marcado **(propuesta)** es una cifra o un
nombre para decidir en la fase, no un dato.

## 0. Antes de empezar (para el agente)

1. Lee [README.md](../README.md), [AGENTS.md](../AGENTS.md), este dossier entero,
   [src/infinito3/remote_devices.py](../src/infinito3/remote_devices.py) y la §5 del dossier de INFINITO Desktop (el
   contrato).
2. Una rama desde `master` y un PR contra `master` por fase. Enrique fusiona. No se empieza una fase sin la puerta de
   la anterior.
3. **No hay Mac.** Todo build de iOS se hace en GitHub Actions. Las pruebas en el iPhone 16 las hace Enrique por
   TestFlight, y cada fase termina con una lista de comprobación para él.
4. Antes de la fase 1, que Enrique responda a las decisiones abiertas de la §9.
5. Esto es producto, no investigación: no hace falta protocolo experimental, pero sí pruebas y que
   `python -m pytest tests/test_*v3.py` siga en verde con cada cambio en `src/infinito3`.
6. La fase 0 es la misma API que la fase 3 de Desktop. Antes de empezarla, mira si ya está en `master`: si lo está,
   esta fase se reduce a añadir las capacidades del teléfono.

## 1. Para qué, y para qué no

**Problema.** La página Dispositivos del portal (`/memory?view=devices`) ya convierte el movimiento y el GPS en
recuerdos, y el pulso de un reloj está a medio hacer. En el iPhone tiene tres límites que una página web no puede
salvar:

- Web Bluetooth no existe en iOS: todos los navegadores de iPhone van sobre WebKit. El pulso del reloj no funciona.
- Con Safari en segundo plano o la pantalla bloqueada, se paran los sensores y el GPS: no hay trayecto en el bolsillo.
- La sesión del portal dura 15 minutos.

**Objetivo.** Que el iPhone sea un sentido de INFINITO con las mismas garantías que la página: el teléfono mide, el
servidor escribe el recuerdo y cada episodio es un recuerdo. El teléfono es un dispositivo más del contrato común, como
el PC, el reloj o la domótica, así que no necesita una integración propia en el servidor. Va en la línea de las
decisiones del 20/11/2026 (apps nativas con Paszoom, `150ced08`) y del 26/11/2026 (conectar todos los dispositivos a la
memoria, `69497532`).

**Fuera del MVP.**

- **Un chat.** Enrique lo descartó el 11/11/2026 (`1d1f907b`): nadie instala una app para chatear.
- **Leer los datos de otras apps** (WhatsApp, Mensajes, correo, notificaciones, historial). iOS no lo permite.
- **Apple Salud** (HealthKit). Es la fase 5, con decisión aparte.
- **Publicar en el App Store.** El MVP es para uso propio por TestFlight.
- **Android.** Va después, con el mismo proyecto Capacitor y el mismo contrato.

Nota: el 7/11/2026 Enrique decidió empezar por Home Assistant antes de las apps nativas (`25977129`), y no consta en
la memoria que se hiciera. Las decisiones del 20/11 y del 26/11 van después; si la del 7/11 sigue en pie, es la
decisión P-7. Home Assistant también entraría por el mismo contrato.

## 2. Lo que ya existe (no rehacerlo)

| Pieza | Dónde | Estado |
|---|---|---|
| OAuth con registro dinámico de clientes y PKCE. Acceso de 1 h y refresco de 90 días que rota con cada uso | `src/infinito3/remote_accounts.py` (`ACCESS_SECONDS`, `REFRESH_SECONDS`, `exchange_refresh_token`) | En producción |
| Ámbitos (*scopes*) pedidos por la app | `remote_accounts.py`: acepta cualquier ámbito que pida la app y lo guarda en el código y en los tokens (`data["scopes"]`); hoy solo hay uno, `SCOPE` | En producción. Nadie los usa para limitar |
| Token → memoria: `bearer(token)` devuelve cuenta, llave, nombre de la app y client id | `remote_accounts.py` y el router de `remote.py` (`__call__`, rama `/mcp`) | En producción |
| Niveles por app (`all`, `add` y `read`; una app nueva empieza en `all`) | `src/infinito3/remote_apps.py` (`allowed`, `refusal`) y `accounts.app_level(cuenta, client_id)`. `add` permite además la cola de tareas, las pautas y las reflexiones: más de lo que necesita un teléfono | En producción |
| Lectores y frases de la página: `read_event`, `read_where`, `read_place`, `sentence`, `place_sentence`, `trip_sentence` | `remote_devices.py` | En master (`4de4a65`) |
| Ritmo por cuenta (`Throttle`) y trayecto sellado con la llave, 24 h, fuera de las copias (`track_*`) | `remote_devices.py`, y su uso en `remote_memory.py` (`device_event`, `device_track`) | En master |
| Pulso del reloj: `read_pulse`, `read_pulse_session` y sus frases (acciones `device_pulse` y `device_pulse_session`) | Solo en la rama local `feat/devices-watch-hr-20261126`, **sin commit** | Requisito de la fase 3 |
| **Contrato común INFINITO Device API v1:** identidad del dispositivo, ámbito `infinito:device`, capacidades con permiso propio, evento versionado con `event_id` e idempotencia, origen observado, declarado o inferido, y registro de dispositivos con su última actividad | §5 del dossier de INFINITO Desktop (PR #198) | Propuesta, sin construir |
| Compilar y firmar iOS sin Mac, con Capacitor 7 y `@capacitor-community/background-geolocation` | `webtilians/dhtrailslocal`: `.github/workflows/ios.yml`, `movil/` | DH Trails está en el App Store desde el 5/11/2026, con `UIBackgroundModes` = location |
| Compilar iOS con Expo SDK 58 (`expo prebuild` en `macos-26`) | `webtilians/recipAI`: `.github/workflows/ios.yml` | Compila sin firmar; TestFlight pendiente de secretos |

`webtilians/principiodelTodo` es **público**, así que los Mac de GitHub Actions no deberían gastar minutos de pago
**[verificar la política vigente de GitHub]**.

## 3. Invariantes (lo que ninguna fase puede romper)

1. **El servidor escribe las palabras.** La app manda números y hechos medidos, en eventos v1, y los lectores comunes
   de `remote_devices.py` los validan. La app no manda texto libre, salvo lo que la persona escribe, comparte o dicta a
   propósito (fase 4): eso va como origen `declared` y pasa por `SensitiveInformationFilter`.
2. **Las lecturas crudas no salen del teléfono.** Se envían episodios. El GPS solo se envía mientras la persona lo
   tiene encendido, y sus puntos van al trayecto sellado de 24 h, no a recuerdos. Es una de las excepciones explícitas
   del contrato.
3. **Un recuerdo por episodio, sin duplicados.** Cada evento lleva su `event_id`; reenviarlo nunca crea otro recuerdo,
   aunque se haya perdido la respuesta. Los límites de ritmo son los de la página y se comparten con ella.
4. **Solo habla con infinitomemory.com.** Sin SDK de analítica, informes de fallos de terceros, mapas ni publicidad.
5. **Permisos mínimos desde el registro.** La app pide el ámbito `infinito:device`, y el servidor solo le deja mandar
   eventos de sus capacidades: nada de leer la memoria, bóveda, `forget`, cola de tareas ni pautas, sea cual sea su
   nivel en Apps.
6. **El token vive en el Llavero (Keychain)** y nunca en registros ni en `UserDefaults`. Al desconectar la app desde
   el portal se revocan sus tokens; en la app, «Salir» borra el Llavero y la cola.
7. **Todo empieza apagado.** Cada sentido se enciende con un interruptor de la persona, y la app explica qué envía.
8. **Lo observado, lo declarado y lo inferido se distinguen.** El autor del recuerdo es el nombre del dispositivo
   («INFINITO para iPhone»), y la fuente y el origen siguen la §5.4 del contrato, así que una IA distingue lo que dijo
   la persona de lo que vio el teléfono. La página Dispositivos sigue con su autor de siempre
   (`remote_devices.AUTHOR`).
9. **Los eventos son datos, nunca órdenes.** Un evento no hace que el servidor ejecute nada ni da acceso a otros
   dispositivos.

## 4. Arquitectura

```
iPhone (Capacitor 7: interfaz web dentro de la app + plugins nativos)
   │  inicio de sesión: navegador del sistema → /authorize (OAuth + PKCE, ámbito infinito:device)
   │                    → com.infinitomemory.app:/oauth
   │  envíos: HTTPS, Authorization: Bearer <acceso>, un evento v1 con su event_id
   ▼
infinitomemory.com  POST /api/devices/v1/events            (contrato común, fase 0)
   │  bearer() → cuenta + llave; ámbito infinito:device; capacidad declarada y permitida
   │  event_id ya visto → la misma respuesta, sin guardar otra vez
   ▼
remote_devices.py: lectores → frase → Throttle → memory.remember(texto, "fact", fuente, autor), con su origen
```

**Capacidades del teléfono en el contrato.** Los nombres son **(propuesta)** y se fijan en la fase 0, en la §5 del
dossier de Desktop:

| Capacidad | Tipos | Hoy en la página (`remote_devices.ACTIONS`) | Fase |
|---|---|---|---|
| `phone.motion` | `episode` | `device_event` | 1 |
| `phone.place` | `place` | `device_place` | 2 |
| `phone.track` | `start`, `point`, `end` | `device_track_start`, `device_track_point`, `device_track_end` | 2 |
| `watch.pulse` | `reading`, `session` | `device_pulse`, `device_pulse_session` (rama local sin commit) | 3 |
| `note` | `text`, con origen `declared` | — (nuevo) | 4 |
| `phone.health` | `daily` | — (nuevo) | 5 |

`device_track_get` y `device_track_clear` son lecturas y borrados de la página, no eventos: se quedan en la página,
porque el ámbito de dispositivo no lee.

Recomendación: **Capacitor 7 como DH Trails**, que corrige la recomendación anterior de Hilo de usar Expo. Motivos:

- Es lo que ya está en el App Store con GPS en segundo plano.
- La carpeta `ios/` se versiona y no se regenera. Expo regenera el proyecto con `expo prebuild --clean`, lo que
  complica añadir targets nativos (extensión Compartir, App Intents) en la fase 4.
- La interfaz puede reutilizar el JavaScript de la página Dispositivos, por ejemplo el conteo de sacudidas.
- El mismo proyecto sirve después para Android.

## 5. Fases

### Fase 0: INFINITO Device API v1 (servidor, `src/infinito3`)

**Estado (28/11/2026): construida en la rama `feat/device-api-v1-20261127`, sin fusionar.** Enrique eligió el
27/11/2026 empezar por aquí (P-8). Archivos:

- `src/infinito3/remote_device_api.py`, nuevo;
- los cambios pequeños en `remote_accounts.py`, `remote.py`, `remote_memory.py`, `remote_apps.py`,
  `remote_devices.py` y `mcp_server.py`;
- 15 pruebas en `tests/test_remote_devices_api_v3.py`;
- la documentación en `remote/README.md`.

Lo que cambió respecto a lo escrito abajo:

- **La ruta** no pasa por la rama `/mcp` del router. Es una ruta más de la app de inicio de sesión, que comprueba el
  token ella misma.
- **Las capacidades** se declaran como ámbitos OAuth, `infinito:device:<capacidad>`, en el registro y en `/authorize`
  (§5.2 del dossier de Desktop). La página de inicio de sesión las enumera. La app debe registrarse con
  `scope = "infinito:device infinito:device:phone.motion …"`.
- **El texto libre** de otras capacidades se ignora, como cualquier campo de más, y nunca llega a un recuerdo.
- **`received_at`** es la hora de creación del recuerdo. `observed_to` queda como `said_at` del recuerdo y es la hora
  que dicen sus palabras.
- **P-6:** el servidor acepta puntos con la hora del teléfono hasta 24 h después (`remote_devices.TRACK_SECONDS`, lo
  que guarda el trayecto), y los demás eventos hasta 7 días (`MAX_AGE_SECONDS`). Si Enrique decide descartar los
  puntos sin cobertura, la app simplemente no los manda. El ritmo de los puntos sigue siendo uno cada 2 s por llegada:
  una cola atrasada se reenvía a ese paso.
- **`[verificar]` resuelto:** el registro acepta `com.infinitomemory.app:/oauth` y el flujo completo funciona con él
  (lo prueba `device_sign_in` en las pruebas). La página de inicio de sesión muestra «INFINITO para iPhone
  (com.infinitomemory.app:/oauth)», porque `_asking` solo acorta las direcciones `esquema://host`.

**Alcance.** Construir el contrato de la §5 del dossier de Desktop, con las capacidades del teléfono. Si la fase 3 de
Desktop ya lo construyó, solo se añaden las capacidades (decisión P-8).

- **Ruta:** `POST /api/devices/v1/events`, con cuerpo JSON (el evento v1) y `Authorization: Bearer`. El router de
  `remote.py` la abre como `/mcp`: `self.bearer(...)` da la cuenta y la llave.
- **Ámbito:** `infinito:device` en el registro y en los tokens. Con él, solo esta ruta. Un token con el ámbito que llame
  a `/mcp` recibe el texto de `remote_apps.refusal`. La única excepción del contrato es temporal: Desktop usa
  `remember` por `/mcp` para su captura rápida hasta que exista esta API, y en cuanto existe deja de poder.
- **Capacidades.** Las de la tabla de la §4, que la persona puede apagar en el registro de dispositivos del portal. Una
  capacidad no declarada o apagada da `403`. Cómo se declaran (en el registro del cliente o en un primer evento) se fija
  en esta fase y se escribe en la §5 del dossier de Desktop, para que los dos dossiers digan lo mismo.
- **Idempotencia:** el `event_id` de cada evento aceptado se guarda 30 días **(propuesta)**, sellado. Un repetido
  responde `200 {"saved": true, "duplicate": true, …}` sin guardar nada.
- **Horas:** `observed_from` y `observed_to` vienen del teléfono; el servidor añade `received_at`.
- **Procedencia:** autor, fuente y origen como en la §5.4 del contrato.
- **Registro de dispositivos** en el portal: nombre, capacidades, cuáles están permitidas, última actividad y
  Desconectar.
- **Reutilizar, no copiar,** los lectores, las frases y el trayecto de `remote_devices.py`. Sacar los `Throttle` de
  `remote_memory.routes` a un sitio común, para que la página y la app compartan el límite por cuenta.
- **Respuestas:** `200 {"saved": true, "text": …}`; `400` si los datos no valen; `401` con
  `WWW-Authenticate: Bearer error="invalid_token"`, para que la app refresque; `403` por ámbito o capacidad; `429`
  si va demasiado rápido.
- Sin CORS: la llama una app nativa, no una página de otro origen.
- **[verificar]** Que el registro dinámico acepte `com.infinitomemory.app:/oauth` como `redirect_uri`, y que la página
  de inicio de sesión muestre bien el nombre de la app. `_asking` en `remote_accounts.py` espera
  `esquema://host`.

**Pruebas:** `tests/test_remote_devices_api_v3.py`, el mismo archivo que la fase 3 de Desktop. Debe cubrir:

- un token válido, caducado, de otra cuenta y sin el ámbito;
- que un token con el ámbito no puede usar `/mcp`;
- una capacidad no declarada o apagada → 403;
- cada capacidad del teléfono;
- que los campos de más se ignoran y el texto libre se rechaza, salvo en `note`;
- que el mismo `event_id` dos veces da un solo recuerdo, también si la primera respuesta se perdió;
- que el límite de ritmo se comparte con la página;
- que el trayecto queda sellado;
- que ningún evento lanza una acción.

**Puerta:** la CI en verde, Enrique fusiona y se despliega. Una prueba con un token real de una app de prueba (como
`python -m cognition login`, que usa una vuelta a `127.0.0.1`, pidiendo el ámbito) guarda un episodio, y el portal
muestra ese dispositivo con su última actividad.

### Fase 1: app base (`movil/` en este repositorio, si se acepta P-1)

**Alcance**

- Proyecto Capacitor 7 con `ios/` versionado, el bundle id y el nombre de la decisión P-2, y el icono de INFINITO.
- **Inicio de sesión**
  - OAuth con PKCE en el navegador del sistema (ASWebAuthenticationSession o equivalente **[verificar plugin]**).
  - Registro dinámico con `client_name` «INFINITO para iPhone» y el ámbito `infinito:device`.
  - Un `device_id` estable (UUID) generado en la primera instalación.
  - Tokens en el Llavero **[verificar plugin]**.
  - Un único refresco a la vez: el refresco rota, y dos a la vez dejan la app fuera.
- **Pantalla:** estado de la conexión, los últimos envíos con la respuesta del servidor, interruptores por sentido
  (todos apagados) y «Salir».
- **Movimiento en primer plano:** el mismo algoritmo que la página (portar el JavaScript de `remote_devices.SCRIPT`),
  enviado como evento `phone.motion`.
- **Sin red:** el evento se guarda en una cola pequeña, con la protección de datos de iOS **[verificar]**, y se reenvía
  con el mismo `event_id`, así que no duplica. Los límites son como los de la cola de Desktop (su decisión P-11).
- **CI:** `.github/workflows/ios-movil.yml`, copiando el de DH Trails.
  - En cada push a `movil/**`, compila sin firmar.
  - A mano (`workflow_dispatch`), firma con `codemagic-cli-tools` y sube a TestFlight.
  - Secretos: los mismos `ASC_KEY_ID`, `ASC_ISSUER_ID`, `ASC_KEY_P8`, `APPLE_TEAM_ID` y `CERTIFICATE_PRIVATE_KEY`
    que DH Trails, que Enrique añade a este repositorio.
- **[verificar]** La ficha de la app en App Store Connect la crea Enrique; `fetch-signing-files --create` hace los
  perfiles.

**Puerta (Enrique, en el iPhone 16)**

1. Instalar desde TestFlight.
2. Iniciar sesión.
3. Ver el iPhone en el registro de dispositivos del portal, con sus capacidades.
4. Encender «Movimiento», sacudir el teléfono y dejarlo quieto.
5. Comprobar que aparece el recuerdo en el portal, con el iPhone como autor y origen observado.
6. Con el modo avión puesto, sacudirlo, quitar el modo avión y ver **un** recuerdo.

### Fase 2: trayectos en segundo plano

**Alcance**

- `@capacitor-community/background-geolocation`, como DH Trails.
- `UIBackgroundModes` = `location` y textos de permiso en español
  (`NSLocationWhenInUseUsageDescription`, `NSLocationAlwaysAndWhenInUseUsageDescription`).
- Los mismos envíos que la página, como eventos de `phone.track`: `start`, un `point` como mucho cada
  `POINT_GAP_SECONDS` (2 s en el servidor) y `end`. «Guardar dónde estoy» es `phone.place`. Las precisiones peores que
  `POINT_ACCURACY_M` (100 m, en el JavaScript de la página) no se envían.
- **Sin cobertura.** Cada punto lleva en `observed_from` la hora del GPS, y el servidor añade la de llegada. Aceptar
  puntos sin cobertura ya no exige otro formato: solo decidir cuánta antigüedad admite el servidor (P-6). En el MVP se
  siguen descartando los que no se pueden mandar enseguida, y la app lo dice.

**Puerta:** andar 20 minutos con la pantalla bloqueada. La ruta aparece en Dispositivos y, al apagar el GPS, un
recuerdo del trayecto (distancia, duración, de dónde a dónde). Anotar el consumo de batería que muestre iOS.

### Fase 3: pulso del reloj por Bluetooth

**Requisito:** fusionar antes en `master` el trabajo de `feat/devices-watch-hr-20261126` (servidor del pulso), y que sus
lectores atiendan la capacidad `watch.pulse` del contrato.

**Alcance**

- `@capacitor-community/bluetooth-le`: servicio Heart Rate `0x180D`, característica `0x2A37`.
- `UIBackgroundModes` = `bluetooth-central` **[verificar qué deja hacer iOS en segundo plano]**.
- Las mismas reglas que la página: el pulso se ve en vivo. Solo se guarda si la persona lo pide (evento `reading`) o,
  al desconectar el reloj, como un resumen con media, mínimo y máximo (evento `session`). Nunca se nombra el reloj.
- El reloj no es un dispositivo aparte del contrato: es una capacidad que declara el teléfono, que es quien lo lee.

**Puerta:** con el reloj Huawei y «HR Data Broadcasts» activado, el pulso se ve en vivo en la app, y al desconectar
queda un recuerdo con el resumen.

### Fase 4: Compartir con INFINITO y Siri

**Alcance**

- **Extensión Compartir** (target Swift en `movil/ios`): texto o enlace compartido desde cualquier app, con un límite
  de longitud, como evento de la capacidad `note`, con origen `declared`.
  - Son palabras de la persona: el servidor las guarda tal cual, con `SensitiveInformationFilter`.
  - La fuente es «Compartido desde el iPhone».
- **App Intent «Recuerda en INFINITO»** para Siri y Atajos, con el texto dictado, por la misma capacidad.
- El token se comparte entre la app y las extensiones con un grupo de acceso del Llavero **[verificar]**. Las
  extensiones generan su propio `event_id`.

**Puerta:**

- Compartir una página desde Safari crea un recuerdo.
- «Oye Siri, recuerda en INFINITO que…» crea un recuerdo.
- Una clave o un número de tarjeta compartidos se rechazan.

### Fase 5 (fuera del MVP): Apple Salud

- Lectura de HealthKit: un resumen diario (pasos, pulso en reposo, sueño) como un recuerdo al día, por una capacidad
  nueva, `phone.health`, con origen observado.
- Requiere el entitlement de HealthKit y una política de privacidad. Si se publica, se aplican las reglas de Apple
  sobre datos de salud **[verificar]**.
- Se decide después de usar las fases 1 a 4.

## 6. Pruebas comunes

- **Servidor:** las pruebas de cada fase en `tests/test_remote_devices_api_v3.py`, compartido con Desktop, y
  `tests/test_*v3.py` en verde.
- **Idempotencia:** cada capacidad nueva lleva su prueba de reenvío con el mismo `event_id`.
- **App:** compilación sin firmar en cada push. La CI no puede probar en un dispositivo: cada fase lleva su lista de
  comprobación para Enrique en la descripción del PR.
- **Privacidad:** cada PR de la app incluye una tabla de qué sale del teléfono, a dónde, cuándo y con qué origen.

## 7. Riesgos

- **Segundo plano:** iOS puede suspender o matar la app. GPS y Bluetooth en segundo plano gastan batería.
- **Rotación del refresco:** dos refrescos a la vez dejan la app fuera (de ahí el refresco único de la fase 1).
- **Esquema propio interceptado:** otra app podría registrar el mismo esquema de retorno. PKCE lo mitiga; más adelante,
  enlaces universales (`https://infinitomemory.com/...`).
- **Pérdida del teléfono:** el token permite mandar eventos, no leer la memoria, gracias al ámbito. Aun así, Enrique
  desconecta el iPhone desde el portal y eso revoca sus tokens.
- **Un token robado** podría inyectar observaciones falsas. Lo limitan el ámbito, las capacidades, el ritmo y la
  revocación, y la procedencia dice de qué dispositivo vino cada recuerdo.
- **El contrato lo comparten dos dossiers.** Un cambio que necesite el teléfono y rompa al PC se versiona (`v2`), y la
  `v1` sigue funcionando.
- **Sin Mac,** cada ciclo de prueba es un build en Actions y TestFlight: depurar el segundo plano y el Bluetooth es
  lento.
- **Publicar** en el App Store pide justificar el GPS en segundo plano (DH Trails ya lo pasó) y el Bluetooth.

## 8. Estimación (sin medir)

- **Fase 0:** una o dos sesiones del agente; menos si la fase 3 de Desktop ya construyó la API.
- **Fases 1 a 3:** una o dos sesiones cada una, más los ciclos de TestFlight de Enrique.
- **Fase 4:** la más incierta, porque tiene targets Swift sin Mac.

No son 20 minutos por función: lo que cuesta es probar cada permiso y cada comportamiento en segundo plano en el
teléfono real.

## 9. Decisiones abiertas (para Enrique, antes de la fase 1)

- **P-1. Dónde vive la app.** Recomendado: `movil/`, dentro de `principiodelTodo` (público: un PR que toca servidor y
  app junto, Mac de Actions sin coste). La alternativa es un repositorio aparte, como DH Trails.
- **P-2. Bundle id y nombre.** Propuesta: `com.infinitomemory.app` y «INFINITO».
- **P-3. Tecnología.** Recomendado: Capacitor 7, como DH Trails. Alternativas: Expo, como RecipAI, o Swift nativo.
- **P-4. Alcance.** El MVP solo por TestFlight; el App Store se decide después.
- **P-5. Orden.** Fases 1 → 2 → 3 → 4, o adelantar el pulso del reloj (fase 3) si es lo que más interesa.
- **P-6. Puntos sin cobertura.** Descartarlos (MVP) o aceptar la hora de observación del teléfono, que ya viaja en el
  evento, hasta una antigüedad máxima.
- **P-7. Prioridad.** Esto es ampliación del producto, y el hito del núcleo sigue siendo cerrar la distancia con la
  base de referencia RAG (AGENTS.md). Hay que decidir también si sigue en pie la decisión del 7/11 de empezar por
  Home Assistant.
- **P-8. Quién construye primero la Device API v1:** esta fase 0 o la fase 3 de Desktop. **Decidida el 27/11/2026:**
  esta fase 0, ya construida en la rama `feat/device-api-v1-20261127`.
- **P-9. Ámbito de dispositivo.** Es la decisión P-8 del dossier de Desktop: el nombre (`infinito:device`) y qué
  permite. Se decide una vez para los dos. Construido como propuesta: `infinito:device` más un ámbito por capacidad, y
  con él solo la ruta de eventos. Se puede cambiar antes de fusionar.

## 10. Qué cambiaría el plan

- Si tras dos semanas con la fase 2 los recuerdos de trayecto no le sirven a Enrique, o la batería no lo aguanta,
  se para antes de las fases 3 y 4 y se revisa qué sentido aporta de verdad.
- Si la fase 0 muestra que el registro dinámico no acepta un esquema propio, primero el servidor, con enlaces
  universales o una página de retorno.
- Si el teléfono necesita algo que la v1 no cubre, se versiona el contrato (`v2`) en vez de abrir una ruta aparte.
- Si Apple cambia lo que una app puede hacer en segundo plano, se rehace la §5 antes de seguir.
