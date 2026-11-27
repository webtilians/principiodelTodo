# INFINITO para iPhone: dossier de implementación del MVP

27/11/2026 · Hilo (Claude Code), a petición de Enrique · Estado: **propuesta, nada implementado**.

Este dossier es para el agente que construya la primera app de INFINITO para iPhone. Parte de la propuesta del
27/11/2026 (recuerdo `4f025901`) y de la valoración de Hilo del mismo día (`e06d1761`). Lo marcado **[verificar]** no
se ha comprobado y hay que mirarlo antes de dar por buena esa parte.

## 0. Antes de empezar (para el agente)

1. Lee [README.md](../README.md), [AGENTS.md](../AGENTS.md), este dossier entero y
   [src/infinito3/remote_devices.py](../src/infinito3/remote_devices.py).
2. Una rama desde `master` y un PR contra `master` por fase. Enrique fusiona. No se empieza una fase sin la puerta de
   la anterior.
3. **No hay Mac.** Todo build de iOS se hace en GitHub Actions. Las pruebas en el iPhone 16 las hace Enrique por
   TestFlight, y cada fase termina con una lista de comprobación para él.
4. Antes de la fase 1, que Enrique responda a las decisiones abiertas de la §9.
5. Esto es producto, no investigación: no hace falta protocolo experimental, pero sí pruebas y que
   `python -m pytest tests/test_*v3.py` siga en verde con cada cambio en `src/infinito3`.

## 1. Para qué, y para qué no

**Problema.** La página Dispositivos del portal (`/memory?view=devices`) ya convierte el movimiento, el GPS y el pulso
de un reloj en recuerdos. En el iPhone tiene tres límites que una página web no puede salvar:

- Web Bluetooth no existe en iOS: todos los navegadores de iPhone van sobre WebKit. El pulso del reloj no funciona.
- Con Safari en segundo plano o la pantalla bloqueada, se paran los sensores y el GPS: no hay trayecto en el bolsillo.
- La sesión del portal dura 15 minutos.

**Objetivo.** Que el iPhone sea un sentido de INFINITO con las mismas garantías que la página: el teléfono mide, el
servidor escribe el recuerdo y cada episodio es un recuerdo. Va en la línea de las decisiones del 20/11/2026 (apps
nativas con Paszoom, `150ced08`) y del 26/11/2026 (conectar todos los dispositivos a la memoria, `69497532`).

**Fuera del MVP.**

- **Un chat.** Enrique lo descartó el 11/11/2026 (`1d1f907b`): nadie instala una app para chatear.
- **Leer los datos de otras apps** (WhatsApp, Mensajes, correo, notificaciones, historial). iOS no lo permite.
- **Apple Salud** (HealthKit). Es la fase 5, con decisión aparte.
- **Publicar en el App Store.** El MVP es para uso propio por TestFlight.
- **Android.** Va después, con el mismo proyecto Capacitor.

Nota: el 7/11/2026 Enrique decidió empezar por Home Assistant antes de las apps nativas (`25977129`), y no consta en
la memoria que se hiciera. Las decisiones del 20/11 y del 26/11 van después; si la del 7/11 sigue en pie, es la
decisión P-7.

## 2. Lo que ya existe (no rehacerlo)

| Pieza | Dónde | Estado |
|---|---|---|
| OAuth con registro dinámico de clientes y PKCE. Acceso de 1 h y refresco de 90 días que rota con cada uso | `src/infinito3/remote_accounts.py` (`ACCESS_SECONDS`, `REFRESH_SECONDS`, `exchange_refresh_token`) | En producción |
| Token → memoria: `bearer(token)` devuelve cuenta, llave, nombre de la app y client id | `remote_accounts.py` y el router de `remote.py` (`__call__`, rama `/mcp`) | En producción |
| Niveles por app (`all`, `add` y `read`; una app nueva empieza en `all`) | `src/infinito3/remote_apps.py` (`allowed`, `refusal`) y `accounts.app_level(cuenta, client_id)` | En producción |
| Lectores y frases de la página: `read_event`, `read_where`, `read_place`, `sentence`, `place_sentence`, `trip_sentence` | `remote_devices.py` | En master (`4de4a65`) |
| Ritmo por cuenta (`Throttle`) y trayecto sellado con la llave, 24 h, fuera de las copias (`track_*`) | `remote_devices.py`, y su uso en `remote_memory.py` (`device_event`, `device_track`) | En master |
| Pulso del reloj: `read_pulse`, `read_pulse_session` y sus frases | Solo en la rama local `feat/devices-watch-hr-20261126`, **sin commit** | Requisito de la fase 3 |
| Compilar y firmar iOS sin Mac, con Capacitor 7 y `@capacitor-community/background-geolocation` | `webtilians/dhtrailslocal`: `.github/workflows/ios.yml`, `movil/` | DH Trails está en el App Store desde el 5/11/2026, con `UIBackgroundModes` = location |
| Compilar iOS con Expo SDK 58 (`expo prebuild` en `macos-26`) | `webtilians/recipAI`: `.github/workflows/ios.yml` | Compila sin firmar; TestFlight pendiente de secretos |

`webtilians/principiodelTodo` es **público**, así que los Mac de GitHub Actions no deberían gastar minutos de pago
**[verificar la política vigente de GitHub]**.

## 3. Invariantes (lo que ninguna fase puede romper)

1. **El servidor escribe las palabras.** La app manda números y hechos medidos, con los mismos campos que la página, y
   los lectores de `remote_devices.py` los validan. La app no manda texto libre, salvo lo que la persona escribe,
   comparte o dicta a propósito (fase 4), y eso pasa por `SensitiveInformationFilter`.
2. **Las lecturas crudas no salen del teléfono.** Se envían episodios. El GPS solo se envía mientras la persona lo
   tiene encendido, y sus puntos van al trayecto sellado de 24 h, no a recuerdos.
3. **Un recuerdo por episodio**, con los mismos límites de ritmo que la página y compartidos con ella.
4. **Solo habla con infinitomemory.com.** Sin SDK de analítica, informes de fallos de terceros, mapas ni publicidad.
5. **El token vive en el Llavero (Keychain)** y nunca en registros ni en `UserDefaults`. Al desconectar la app desde
   Apps del portal se revocan sus tokens; en la app, «Salir» borra el Llavero.
6. **Todo empieza apagado.** Cada sentido se enciende con un interruptor de la persona y la app explica qué envía.
7. El recuerdo dice que viene del teléfono (autor de `remote_devices.AUTHOR`), así que una IA distingue lo que dijo la
   persona de lo que vio un dispositivo.

## 4. Arquitectura

```
iPhone (Capacitor 7: interfaz web dentro de la app + plugins nativos)
   │  inicio de sesión: navegador del sistema → /authorize (OAuth + PKCE) → com.infinitomemory.app:/oauth
   │  envíos: HTTPS, Authorization: Bearer <acceso>
   ▼
infinitomemory.com  /api/devices/<acción>            (nuevo, fase 0)
   │  bearer() → cuenta + llave; nivel de la app (add/all; read → 403)
   ▼
remote_devices.py: lectores → frase → Throttle → memory.remember(texto, "fact", fuente, autor)
```

Recomendación: **Capacitor 7 como DH Trails**, que corrige la recomendación anterior de Hilo de usar Expo. Motivos:

- Es lo que ya está en el App Store con GPS en segundo plano.
- La carpeta `ios/` se versiona y no se regenera. Expo regenera el proyecto con `expo prebuild --clean`, lo que
  complica añadir targets nativos (extensión Compartir, App Intents) en la fase 4.
- La interfaz puede reutilizar el JavaScript de la página Dispositivos, por ejemplo el conteo de sacudidas.
- El mismo proyecto sirve después para Android.

## 5. Fases

### Fase 0: API de dispositivos con token (servidor, `src/infinito3`)

**Alcance**

- Rutas `POST /api/devices/<acción>` con cuerpo JSON y `Authorization: Bearer`. Acciones: `event`, `place`,
  `track_start`, `track_point`, `track_end` y, después de la fase 3, `pulse` y `pulse_session`.
- El router de `remote.py` las abre como `/mcp`: `self.bearer(...)` da la cuenta y la llave.
- Si no hay token o está caducado: `401` con `WWW-Authenticate: Bearer error="invalid_token"`, para que la app refresque.
- **Nivel:** `remote_apps.allowed(level, "remember")`; si no, `403` con el texto de `refusal`.
- **Reutilizar, no copiar,** los lectores, las frases y el trayecto de `remote_devices.py`. Sacar los `Throttle` de
  `remote_memory.routes` a un sitio común, para que la página y la app compartan el límite por cuenta.
- Respuestas: `200 {"saved": true, "text": …}`, `400` si los datos no valen, `429` si va demasiado rápido.
- Sin CORS: la llama una app nativa, no una página de otro origen.
- **[verificar]** Que el registro dinámico acepte `com.infinitomemory.app:/oauth` como `redirect_uri`, y que la página
  de inicio de sesión muestre bien el nombre de la app. `_asking` en `remote_accounts.py` espera
  `esquema://host`.

**Pruebas:** un archivo nuevo, `tests/test_remote_devices_api_v3.py`, que cubra:

- token válido, caducado y de otra cuenta;
- nivel `read` → 403;
- cada acción;
- que los campos de más se ignoran y el texto libre se rechaza;
- que el límite se comparte con la página;
- que el trayecto queda sellado.

**Puerta:** la CI en verde, Enrique fusiona, se despliega, y una prueba con un token real de una app de prueba
(como `python -m cognition login`, que usa una vuelta a `127.0.0.1`) guarda un episodio.

### Fase 1: app base (`movil/` en este repositorio, si se acepta P-1)

**Alcance**

- Proyecto Capacitor 7 con `ios/` versionado, el bundle id y el nombre de la decisión P-2, y el icono de INFINITO.
- **Inicio de sesión**
  - OAuth con PKCE en el navegador del sistema (ASWebAuthenticationSession o equivalente **[verificar plugin]**).
  - Registro dinámico con `client_name` «INFINITO para iPhone».
  - Tokens en el Llavero **[verificar plugin]**.
  - Un único refresco a la vez: el refresco rota, y dos a la vez dejan la app fuera.
- **Pantalla:** estado de la conexión, los últimos recuerdos guardados desde el teléfono, interruptores por sentido
  (todos apagados) y «Salir».
- **Movimiento en primer plano:** el mismo algoritmo que la página (portar el JavaScript de `remote_devices.SCRIPT`),
  enviado a `/api/devices/event`.
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
3. Ver la app en Apps del portal y ponerla en «Leer y añadir».
4. Encender «Movimiento», sacudir el teléfono y dejarlo quieto.
5. Comprobar que aparece el recuerdo en el portal, con el teléfono como autor.

### Fase 2: trayectos en segundo plano

**Alcance**

- `@capacitor-community/background-geolocation`, como DH Trails.
- `UIBackgroundModes` = `location` y textos de permiso en español
  (`NSLocationWhenInUseUsageDescription`, `NSLocationAlwaysAndWhenInUseUsageDescription`).
- Los mismos envíos que la página: `track_start`, un `track_point` como mucho cada `POINT_GAP_SECONDS` y `track_end`.
  Las precisiones peores que `POINT_ACCURACY_M` no se envían.
- **Sin cobertura:** el servidor pone la hora del punto al recibirlo, así que en el MVP los puntos sin red se descartan
  y la app lo dice. Aceptar la hora del teléfono, dentro de unos límites, es la decisión P-6.

**Puerta:** andar 20 minutos con la pantalla bloqueada. La ruta aparece en Dispositivos y, al apagar el GPS, un
recuerdo del trayecto (distancia, duración, de dónde a dónde). Anotar el consumo de batería que muestre iOS.

### Fase 3: pulso del reloj por Bluetooth

**Requisito:** fusionar antes en `master` el trabajo de `feat/devices-watch-hr-20261126` (servidor del pulso).

**Alcance**

- `@capacitor-community/bluetooth-le`: servicio Heart Rate `0x180D`, característica `0x2A37`.
- `UIBackgroundModes` = `bluetooth-central` **[verificar qué deja hacer iOS en segundo plano]**.
- Las mismas reglas que la página: el pulso se ve en vivo. Solo se guarda si la persona lo pide o, al desconectar el
  reloj, como un resumen (media, mínimo y máximo). Nunca se nombra el reloj.

**Puerta:** con el reloj Huawei y «HR Data Broadcasts» activado, el pulso se ve en vivo en la app, y al desconectar
queda un recuerdo con el resumen.

### Fase 4: Compartir con INFINITO y Siri

**Alcance**

- **Extensión Compartir** (target Swift en `movil/ios`): texto o enlace compartido desde cualquier app, con un límite
  de longitud, a una acción nueva `/api/devices/note`.
  - Son palabras de la persona: el servidor las guarda tal cual, con `SensitiveInformationFilter`.
  - La fuente es «Compartido desde el iPhone».
- **App Intent «Recuerda en INFINITO»** para Siri y Atajos, con el texto dictado.
- El token se comparte entre la app y las extensiones con un grupo de acceso del Llavero **[verificar]**.

**Puerta:**

- Compartir una página desde Safari crea un recuerdo.
- «Oye Siri, recuerda en INFINITO que…» crea un recuerdo.
- Una clave o un número de tarjeta compartidos se rechazan.

### Fase 5 (fuera del MVP): Apple Salud

- Lectura de HealthKit: un resumen diario (pasos, pulso en reposo, sueño) como un recuerdo al día.
- Requiere el entitlement de HealthKit y una política de privacidad. Si se publica, se aplican las reglas de Apple
  sobre datos de salud **[verificar]**.
- Se decide después de usar las fases 1 a 4.

## 6. Pruebas comunes

- **Servidor:** las pruebas de cada fase, y `tests/test_*v3.py` en verde.
- **App:** compilación sin firmar en cada push. La CI no puede probar en un dispositivo: cada fase lleva su lista de
  comprobación para Enrique en la descripción del PR.
- **Privacidad:** cada PR de la app incluye una tabla de qué sale del teléfono, a dónde y cuándo.

## 7. Riesgos

- **Segundo plano:** iOS puede suspender o matar la app. GPS y Bluetooth en segundo plano gastan batería.
- **Rotación del refresco:** dos refrescos a la vez dejan la app fuera (de ahí el refresco único de la fase 1).
- **Esquema propio interceptado:** otra app podría registrar el mismo esquema de retorno. PKCE lo mitiga; más adelante,
  enlaces universales (`https://infinitomemory.com/...`).
- **Pérdida del teléfono:** el token abre la memoria. Enrique desconecta la app desde Apps del portal y eso revoca sus
  tokens.
- **Sin Mac,** cada ciclo de prueba es un build en Actions y TestFlight: depurar el segundo plano y el Bluetooth es
  lento.
- **Publicar** en el App Store pide justificar el GPS en segundo plano (DH Trails ya lo pasó) y el Bluetooth.

## 8. Estimación (sin medir)

- **Fase 0:** una sesión del agente.
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
- **P-6. Puntos sin cobertura.** Descartarlos (MVP) o aceptar la hora del teléfono dentro de unos límites.
- **P-7. Prioridad.** Esto es ampliación del producto, y el hito del núcleo sigue siendo cerrar la distancia con la
  base de referencia RAG (AGENTS.md). Hay que decidir también si sigue en pie la decisión del 7/11 de empezar por
  Home Assistant.

## 10. Qué cambiaría el plan

- Si tras dos semanas con la fase 2 los recuerdos de trayecto no le sirven a Enrique, o la batería no lo aguanta,
  se para antes de las fases 3 y 4 y se revisa qué sentido aporta de verdad.
- Si la fase 0 muestra que el registro dinámico no acepta un esquema propio, primero el servidor, con enlaces
  universales o una página de retorno.
- Si Apple cambia lo que una app puede hacer en segundo plano, se rehace la §5 antes de seguir.
