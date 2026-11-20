# INFINITO Teams — ADR P0: espacios, claves y roles

**Estado:** PROPUESTO. Pendiente de firma del propietario (sección 8). No autoriza implementación, migración,
despliegue, datos reales ni gasto.
**Fecha:** 20/11/2026.
**Base de código:** `master` en `411c540` (comprobado con `git fetch` el 20/11/2026). Las referencias `archivo:línea`
son a ese commit.
**Entrada:** dossier «INFINITO Empresarial — Paper de arquitectura v0.1» (20/11/2026, no versionado en el repositorio),
y las correcciones posteriores del propietario del mismo día. Este documento es el entregable de su fase **P0**.
**Alcance:** producto empresarial, separado del programa de investigación (decisión del propietario del 20/11/2026).
No toca la memoria personal, el motor cognitivo ni los experimentos.

---

## 0. Resumen

| # | Decisión propuesta | Por qué |
|---|---|---|
| D1 | La unidad de seguridad, cifrado e índice es el **espacio**; cada espacio es un archivo sellado propio, con el mismo formato que una memoria personal. | Reutiliza `SealedMemories`. Cada índice queda aislado sin trabajo extra. |
| D2 | **Una clave de datos (DEK) por espacio**, entregada a cada miembro en un sobre. Se recomienda que los sobres usen un **par X25519 por cuenta** (opción B2). La decisión la toma el propietario a la vista de la tabla del §2.2. | Es la única opción evaluada que da altas, sobre automático del CEO y rotación sin que el destinatario esté conectado, y sin romper la promesa actual sobre el disco. |
| D3 | Se separa **quién autoriza** (base de políticas) de **quién entrega la clave** (alguien que ya la tiene abierta). `org_admin` administra sin recibir sobres. Nadie se concede acceso a sí mismo. | El dossier quiere un administrador que no lee, y el cifrado por espacio solo lo permite así. |
| D4 | El `owner` recibe siempre un sobre de cada espacio corporativo, y la interfaz lo muestra. No existe ninguna ruta de claves hacia la memoria personal. | Así se cumple «el CEO lee lo corporativo, nunca lo personal», sin puertas traseras. |
| D5 | Hay un PDP central en cada llamada. La identidad sale siempre del token, nunca de los argumentos. Los metadatos de autorización van en claro y documentados; el contenido va sellado. | Es el orden de evaluación del dossier (§5) aplicado al código actual. |
| D6 | Las apps tienen un nivel empresarial **aparte** del personal y, por defecto, **ninguno**. El `DEFAULT="all"` personal no cambia. | Mínimo privilegio en la empresa (A16) sin romper a los usuarios actuales. |
| D7 | Cada recuerdo lleva su versión. Las escrituras llevan `idempotency_key`. Una `decision` que guarda un agente queda en `proposal` hasta que una persona la confirme. | Hoy no existen ni versiones ni idempotencia (§1.2). |
| D8 | Al dar de baja a alguien se le deniega de inmediato en el PDP. Después se rota la DEK de forma diferida, con sobres nuevos para los miembros que siguen. | La revocación es real en el servidor, y su límite fuera de él se dice con honestidad. |
| D9 | Se recupera solo de los espacios autorizados y luego se fusionan los resultados. La caché del ranker semántico se separa por espacio. | Hoy esa caché es común a todo el proceso (§1.2). |
| D10 | Todo queda detrás de `INFINITO_ENTERPRISE_ENABLED=0`. Las copias de seguridad incluyen `orgs/`. | Con el flag apagado, el comportamiento actual es idéntico. |

---

## 1. Contexto verificado

### 1.1 Flujo actual de una llamada MCP autenticada

1. El cliente llama a `POST /mcp` con `Bearer`. `_Router` (`remote.py:235`) llama a `bearer()` (`remote_accounts.py:581`).
   Este usa `Grants.find("access", token)`, que desenvuelve la **clave de memoria** guardada bajo el propio token, y
   devuelve `(cuenta, clave, nombre de app, client_id)`.
2. `_Router` escribe en `scope["state"]` los valores `infinito_person`, `infinito_key`, `infinito_memories`,
   `infinito_app` e `infinito_client` (`remote.py:284`).
3. Cada tool pasa por `with_memory()` (`remote.py:365`), que hace dos cosas:
   - llama a `refused()` (`remote.py:336`), que compara `accounts.app_level(persona, client_id)` con
     `remote_apps.refusal()`;
   - si se permite, llama a `SealedMemories.use()` (`remote.py:207`). Este toma un bloqueo por nombre, descifra
     `memory.sealed` (AES-256-GCM), carga la SQLite con `SQLiteCognitiveMemoryStore.from_bytes`, ejecuta la acción y,
     si escribe, vuelve a cifrar el **archivo entero**.
4. El autor de lo que se guarda lo pone `app_of(ctx)` (`remote.py:385`): es el nombre de app del registro OAuth, nunca
   un argumento de la tool.
5. La página web `/memory` pide la contraseña. La clave se guarda 15 minutos envuelta bajo un token de sesión, en una
   cookie `httponly` (`remote_memory.py:32-34`, `:541`, `:613`). Los formularios llevan CSRF.

**Lo que esto implica para Teams:** la identidad ya se resuelve en el servidor y llega en `scope["state"]`; el PDP se
apoya en ella. La clave de una cuenta solo existe mientras dura una petición de esa cuenta (por el token o por la
sesión web). Esa restricción es la que obliga a elegir bien el modelo de claves.

### 1.2 Lo que existe y lo que no

| Elemento | Estado en `411c540` |
|---|---|
| Cifrado por memoria, con la clave envuelta bajo contraseña, código de recuperación y tokens | Sí (`remote_accounts.py:8-13`) |
| Promesa: leer el disco y las copias no basta; un servidor modificado en ejecución sí podría capturar datos | Sí, escrita (`remote.py:11-17`) |
| Claves asimétricas por cuenta | **No.** No hay ninguna coincidencia de `X25519`, `Ed25519` ni `public_key` en `src/infinito3/`. |
| Compartir a mano en cajas con clave por pareja, entregada por enlace de un uso (`Grants.issue`) | Sí (`remote_shares.py:1-19`) |
| Nivel de permisos por app, con valor por defecto `"all"` (también al reconectar) | Sí (`remote_apps.py:18`) |
| Cola de tareas entre IAs que **solo la persona aprueba** | Sí (`remote_tasks.py:1-12`) |
| Historial por sustitución (`supersedes_id`, estado `SUPERSEDED`) | Sí (`persistent_memory.py:222`, `:289-292`) |
| `expected_version` por recuerdo e `idempotency_key` | **No** |
| Organización, espacios, roles y concesiones por recurso | **No** |
| Caché de vectores del ranker semántico | Común a todo el proceso, con el texto en claro como clave (`semantic.py:56`, `:103-117`) |
| Servidor | Un único proceso uvicorn (`remote.py:856`); los bloqueos son `threading.Lock` de ese proceso |
| Copias de seguridad | Copian lo que ya está sellado. Los nombres de usuario y los tamaños se pueden leer en ellas (`remote_backup.py:10`). Lo olvidado sigue en las copias antiguas hasta que caducan (`remote_backup.py:15`). |

### 1.3 Trabajo previo relacionado (no fusionado o solo propuesto)

- **Rama `experiment/security-s0h-tenant-isolation-20261109`**, PR #151 en borrador, sin fusionar. Tiene 12 pruebas E2E
  (W00–W11) de aislamiento entre **cuentas personales** OAuth y `remote_provenance.py` (firma Ed25519 opcional). Es un
  **arnés reutilizable** para A04, A06, A08 y A16. **No certifica** el aislamiento empresarial: no hay organizaciones
  ni espacios, y su A07 (rotación de la clave de firma) sigue abierto.
- **Diseño del 8/11/2026 para proyectos compartidos personales** (solo propuesto): cada proyecto es una memoria sellada
  aparte, se entra por enlace de un uso, un recuerdo pasa al proyecto solo si la persona lo pide y el autor lo pone el
  servidor. Teams generaliza esa idea. Su mecanismo de entrega por enlace se evalúa como opción B1 en el §2.2.

### 1.4 Coste medido del modelo «archivo entero por espacio»

**Método:** el script del apéndice A, ejecutado en el portátil del propietario con Python 3.11.9. Se mide la mediana
de 7 ejecuciones, con recuerdos sintéticos de unos 100 caracteres que llevan `kind`, `source`, `project` y `date`, sin
modelo semántico.
**No incluye** el AES-GCM, porque `cryptography` no está instalado en ese equipo; habrá que medirlo en CI dentro de P1.

| Recuerdos en el espacio | Tamaño de la imagen SQLite | Abrir + `recall` | Abrir + `remember` + serializar |
|---:|---:|---:|---:|
| 500 | 1,14 MB | 116 ms | 88 ms |
| 2.000 | 4,48 MB | 423 ms | 529 ms |
| 5.000 | 11,15 MB | 947 ms | 1.034 ms |

**Lectura:** el coste crece de forma aproximadamente lineal, unos 0,2 ms por recuerdo y llamada. Mientras dura una
escritura, el bloqueo del espacio frena a los demás. Un `company_recall` que recorra varios espacios suma sus costes.

---

## 2. Decisiones

### D1 — El espacio como unidad de seguridad y almacenamiento

**Decisión.** Cada `MemorySpace` es un archivo `orgs/<org_id>/spaces/<space_id>/memory.sealed` con el mismo formato
(`INF1` + nonce + AES-GCM) que una memoria personal. Se abre con `SealedMemories` mediante un tercer tipo de
propietario, `Spaces`, igual que hoy existen `People` y `Accounts`. El índice léxico vive dentro del archivo, de modo
que el aislamiento de índices por espacio que pide el dossier (§10.2) sale por construcción.

**Alternativas rechazadas:**
- **Una SQLite común con columna `org_id`:** pierde el cifrado actual, y un fallo de filtro sería una fuga.
- **Un archivo por organización:** no aísla departamentos ni proyectos, ni en el cifrado ni en el índice.

**Consecuencias:**
- `company_recall` abre cada espacio autorizado por separado y fusiona resultados que ya están filtrados.
- **Límite operativo propuesto para el piloto:** unos 2.000 recuerdos por espacio (alrededor de 0,5 s por llamada,
  según el §1.4). Por encima, habrá que decidir entre partir el espacio, mantener en memoria la imagen descifrada con un
  TTL o pasar a un almacén por registros. Queda fuera del MVP.
- Se mantiene un solo proceso. Varios workers necesitan otro tipo de bloqueo y quedan fuera del MVP.

### D2 — Claves: DEK por espacio y entrega por sobres

Cada espacio tiene una DEK aleatoria de 256 bits con versión `k`. La pregunta abierta es **cómo llega la DEK a cada
miembro**, teniendo en cuenta que la clave de su cuenta solo existe durante sus propias peticiones (§1.1).

#### 2.1 Escenarios que la entrega debe cubrir

- **E1:** alta de alguien que no está conectado en ese momento.
- **E2:** sobre automático para el `owner` cuando otra persona crea un espacio.
- **E3:** rotación tras una baja: nueva DEK para los que siguen, sin que estén conectados.
- **E4:** alta de muchas personas de golpe (por ejemplo, 15 en un proyecto).
- **E5:** mantener la promesa actual de que leer el disco y las copias no basta.
- **E6:** cuánto trabajo nuevo supone.

#### 2.2 Opciones evaluadas

| Opción | E1 | E2 | E3 | E4 | E5 | E6 |
|---|---|---|---|---|---|---|
| **A.** Clave única por organización | Tiene el mismo problema de entrega | Igual | Rotar afecta a toda la organización | Igual | Sí | Baja, **pero no aísla espacios** |
| **B1.** DEK por espacio entregada por enlace de un uso (patrón de `remote_shares`) | Sí, pero el destinatario tiene que abrir el enlace y aceptar | **No:** el `owner` tendría que aceptar un enlace por cada espacio | **No:** cada rotación obliga a todos a aceptar de nuevo, y mientras tanto pierden el acceso | 15 enlaces | Sí | Baja: el mecanismo ya existe |
| **B2.** DEK por espacio en sobres X25519 por cuenta | Sí, sin que el destinatario haga nada | Sí | Sí | Sí | Sí: la clave privada va sellada con la clave de la cuenta | Media: par de claves, formato de sobre y pruebas |
| **C.** DEK envuelta bajo una clave maestra del servidor (KEK) | Sí | Sí | Sí | Sí | **No** si la KEK está en el mismo disco. Solo parcialmente con un KMS o HSM externo, que añade infraestructura. | Baja |

Una entrega «diferida» con claves simétricas (esperar a que el destinatario entre) solo funciona si el servidor guarda
la DEK sin proteger hasta ese momento. Eso equivale a la opción C.

**Recomendación: B2.** Es la única opción que cubre E1–E4 sin romper E5. B1 **sí se reutiliza**, pero para otra cosa:
invitar a una persona a la organización (`Grants.issue`, enlace de un uso, sin clave dentro). No sirve para entregar
DEKs. C es defendible solo si el propietario decide que el servidor es de confianza y lo dice en el producto.

#### 2.3 Detalle de B2 (para implementar en P1 si se firma)

- **Par de claves por cuenta.** La clave privada X25519 se guarda sellada con la clave de memoria de la cuenta, como ya
  se hace con `contacts.sealed`. La pública se guarda en `accounts.json`.
- **Formato del sobre.** X25519 efímero → HKDF-SHA256 → AES-256-GCM sobre la DEK. Los datos asociados son
  `org_id|space_id|key_version|recipient`, para que un sobre no se pueda mover a otro espacio ni a otra persona.
- **Cuentas existentes.** El par se crea la próxima vez que la persona entra con el flag encendido, porque es el
  momento en que el servidor tiene su clave. Hasta entonces figura como «pendiente de primer acceso» y no puede recibir
  sobres. No hay migración destructiva.
- **Optimización opcional.** Al abrir un espacio por primera vez, el sobre X25519 puede convertirse en uno simétrico
  bajo la clave de la cuenta.
- **Dependencias.** `cryptography` ya se usa en `remote.py`, importado dentro de la función, y X25519 y HKDF están en
  la misma librería. El núcleo sigue usando solo la biblioteca estándar (AGENTS.md).

#### 2.4 Modelo de amenaza (explícito)

**Protege contra:**
- quien lea el disco o las copias;
- un fallo de filtro entre espacios u organizaciones: el espacio no autorizado ni siquiera se puede descifrar en esa
  petición, porque no hay sobre que abrir.

**No protege contra:**
- un servidor comprometido mientras está en ejecución, igual que hoy (`remote.py:17`);
- que quien controle el servidor sustituya claves públicas.

**No se anuncia como cifrado de extremo a extremo.**

### D3 — Separar la autorización de la entrega de claves

- La **base de políticas** (D5) decide quién puede acceder. El **sobre** lo crea alguien que ya tiene la DEK abierta en
  esa petición: el `owner` o un `manager` del espacio, desde la web (sesión con clave) o desde una app con nivel
  `manage`. En el MVP, `manage` solo se usa desde la web (D6).
- `org_admin` gestiona altas, bajas, roles y estructura, pero **su rol no le da sobres**:
  - si crea un espacio, la DEK se genera en esa petición, se envuelve para el `owner` y para el `manager` designado, y
    se descarta;
  - si da de alta a alguien en un espacio, la concesión queda en `pending_key` hasta que el `owner` o un `manager` de
    ese espacio la confirme en la página (un clic, y entonces se crea el sobre).
- **Nadie puede concederse acceso a sí mismo.** Hace falta otra persona con `manage`, o el `owner`. Toda concesión
  queda en la auditoría, que pueden ver el `owner` y los `manager` del espacio.
- **Límite honesto:** «administrar sin leer» se cumple criptográficamente mientras el `org_admin` no sea miembro. Lo que
  le impide hacerse miembro es organizativo (no puede concederse acceso y todo se audita), no criptográfico.

### D4 — El `owner` (CEO)

- Al crear cualquier espacio corporativo, el servidor crea siempre un sobre para el `owner`, aunque no esté conectado
  (gracias a B2). La interfaz lo muestra: «El propietario de la organización puede leer este espacio». No es un escrow
  oculto.
- **Espacios `restricted`** (por ejemplo, RR. HH.): en el MVP también incluyen al `owner`. Excluirlo necesita revisión
  legal y es la pregunta P-3.
- **Memoria personal:** no existe ninguna ruta de claves desde la organización. Ninguna memoria personal tiene sobres
  para otras personas, y las tools personales no cambian (A05).
- **Si el `owner` pierde su cuenta** (contraseña y código de recuperación): sin escrow no hay forma de recuperar sus
  sobres. Los espacios siguen siendo legibles para sus otros miembros, porque **todo espacio debe tener al menos dos
  sobres** (el del `owner` y el del creador o `manager`). Es la pregunta P-4.

### D5 — Autorización (PDP/PEP) y metadatos

- Se crea un módulo nuevo (nombre provisional `remote_org.py`) con `orgs/<org_id>/policy.db` en SQLite. Contiene
  `organizations`, `memberships`, `spaces`, `space_grants`, `key_envelopes` y `audit_events`, un subconjunto del
  esquema del dossier (§9).
- **Orden de evaluación:** el del dossier (§5), en **cada** llamada, sin caché por sesión.
- **Identidad:** siempre de `scope["state"]` (cuenta y `client_id` del token). El `org_id` y el `space_id` de las tools
  son peticiones no confiables. El autor es la cuenta autenticada más el nombre de la app, como hace hoy `app_of`.
- **Metadatos en claro:** los ids, roles, fechas y nombres de organizaciones, espacios, departamentos y proyectos se
  pueden leer en el disco y en las copias, igual que hoy los nombres de usuario (`remote_backup.py:10`). Se documenta,
  y la interfaz avisa de no poner información sensible en los nombres. El texto, las etiquetas y las evidencias van
  dentro del espacio sellado.
- **Denegación estable:** un espacio que no existe y uno no autorizado reciben la misma respuesta, sin ids, títulos ni
  recuentos (A03).

### D6 — Permisos por app en la vía empresarial

- Cada app conectada tiene, **por organización**, un nivel `none` (por defecto), `read` o `add`. `manage` nunca se
  concede a una app en el MVP: las concesiones solo se hacen desde la página web (A16).
- El `DEFAULT="all"` personal **no cambia**, por compatibilidad (principio 9 del dossier). Cambiarlo es otra decisión
  (P-5).

### D7 — Escritura, versiones e idempotencia

- `company_remember` exige un `space_id` explícito y permiso `write`. El autor lo pone el servidor.
- Si un **agente** guarda algo con `kind="decision"`, se almacena como `proposal` hasta que una persona lo confirme en
  la página. Es el mismo patrón de aprobación humana que `remote_tasks.py` y responde al §16.3 del dossier.
- El almacén del espacio gana dos cosas:
  - un campo `version` por recuerdo;
  - una tabla de `idempotency_key` con la clave y el resultado, que caduca a las 24 h.
- `company_correct` exige `expected_version` y devuelve un conflicto explícito si no coincide (A09, A11). El historial
  reutiliza `supersedes_id` y `SUPERSEDED`. Todo ocurre bajo el bloqueo del espacio, que `SealedMemories` ya serializa
  por nombre.

### D8 — Revocación y rotación

- **Baja de un miembro:** sus concesiones y sobres se borran de `policy.db` en una sola transacción. El PDP le deniega
  en la siguiente llamada, también con los tokens que ya tenía, porque la autorización se evalúa en cada llamada (A08).
- **Rotación:** el espacio queda en `rotation_pending`. En la siguiente petición de cualquier miembro vigente, que ya
  tiene la DEK abierta, el servidor genera la DEK `k+1`, vuelve a cifrar el archivo y crea sobres X25519 para los
  miembros que siguen. Esto protege frente a un exmiembro que consiga después el archivo del disco o copias nuevas.
- **Límites que se dicen al usuario:** las copias anteriores se siguen pudiendo abrir con la DEK vieja hasta que
  caduquen, igual que hoy pasa con lo olvidado (`remote_backup.py:15`). Lo que ya apareció en chats no se puede
  recuperar (A15).

### D9 — Recuperación y respuesta

- **Orden:** resolver los espacios autorizados, abrir solo esos, buscar dentro de cada uno, fusionar el top-k y
  etiquetar cada resultado con organización, espacio, autor, fecha y estado.
- **Separación de lo personal:** las respuestas empresariales no mezclan memoria personal ni las vistas y reflexiones
  personales que hoy añade `with_views`.
- **Caché del ranker:** `MeaningRanker._cache` es común a todo el proceso y usa el texto en claro como clave. No
  devuelve texto a otros usuarios, pero junta en memoria textos de todos los tenants y abre un canal de tiempo teórico.
  En la vía empresarial, la clave de la caché será `HMAC(DEK del espacio, texto)`, o habrá una caché por espacio.
- **Resúmenes** (`company_briefing`): quedan fuera del MVP. Cuando lleguen, su visibilidad será la intersección de la
  de sus fuentes (A13).

### D10 — Activación, compatibilidad y copias

- Con `INFINITO_ENTERPRISE_ENABLED=0` (valor por defecto) no se registran las tools `company_*`, no hay rutas web
  empresariales y no se generan pares X25519.
- No cambian `remember`, `recall` ni `forget` personales, ni el formato de `memory.sealed`.
- `remote_backup` incluirá `orgs/` (`policy.db`, espacios sellados y sobres). Restaurar una copia se prueba antes de
  cualquier piloto (A12).

---

## 3. Alcance del MVP (P1)

**Entra:**
- organizaciones y dos tipos de espacio, `org` (general) y `project`;
- los roles `owner`, `org_admin`, `manager` y `member`;
- las tools `company_list_spaces`, `company_remember` y `company_recall`;
- las páginas web para crear la organización, invitar, crear proyectos, añadir o quitar miembros y confirmar
  concesiones pendientes.

**Los departamentos** son solo grupos para invitar y para la interfaz; su propio tipo de espacio llega después.

**Queda fuera:**
- espacios `department` y `restricted`, el rol `guest`, `company_correct` y `company_forget` por MCP (en el MVP se
  corrige y se borra desde la web);
- publicar desde la memoria personal, briefings, conectores, SSO/SCIM y varios procesos.

## 4. Matriz de operaciones y roles (MVP)

En la tabla, X es un espacio de tipo `project`. El espacio `org` general funciona igual, con todos los miembros de la
organización como `member`.

| Operación | owner | org_admin | manager de X | member de X | no miembro de X |
|---|---|---|---|---|---|
| Ver que X existe | sí | sí (metadatos) | sí | sí | **no** |
| Leer recuerdos de X | sí | no¹ | sí | sí | **no** |
| Escribir en X | sí | no¹ | sí | sí (si la concesión incluye `write`) | **no** |
| Corregir o retirar un recuerdo propio (web) | sí | — | sí | sí | no |
| Corregir o retirar un recuerdo ajeno (web) | sí | no | sí | no | no |
| Confirmar una `decision` propuesta por un agente | sí | no | sí | no | no |
| Crear un espacio | sí | sí (sin sobre propio) | no | no | no |
| Añadir un miembro a X | sí (sobre inmediato) | sí → `pending_key` | sí (sobre inmediato) | no | no |
| Concederse acceso a sí mismo | — | **no** | — | **no** | **no** |
| Quitar un miembro de X | sí | sí | sí (nunca al `owner`) | no | no |
| Nombrar `org_admin` o transferir la propiedad | solo `owner` | no | no | no | no |
| Ver la auditoría de X (sin contenido) | sí | sí | sí | no | no |
| Leer la memoria personal de otra persona | **no** | **no** | **no** | **no** | **no** |
| Dar nivel `read`/`add` en la organización a una app propia | cada persona, para sus propias apps | | | | |

¹ Salvo que otra persona le haya concedido acceso a X como miembro.

## 5. Pruebas exigidas en P1

Todas con datos sintéticos. Hay dos organizaciones de prueba:
- **Acme:** el `owner`; Ana (Desarrollo, en Alfa); Beatriz (Diseño, en Alfa); Carlos (Marketing, en Beta).
- **Globex:** con su propio «Proyecto Alfa», para probar la colisión de nombres (A14).

Archivo propuesto: `tests/test_remote_org_v3.py`. Así entra en la puerta `tests/test_*v3.py` de AGENTS.md.

| Prueba | Cómo se comprueba |
|---|---|
| A01–A03, A05 | Llamadas MCP en proceso con tres cuentas OAuth (enfoque del arnés S0-H) y comparación con la matriz del §4 |
| A03 | La respuesta a un espacio inexistente y a uno no autorizado es idéntica byte a byte |
| **Denegado = sin descifrar** | Un espía sobre `SealedMemories.read` del espacio comprueba que una petición denegada no lo llama nunca. Es el entregable P1 del dossier. |
| A04, A14 | Peticiones con el `org_id` o `space_id` de otra organización en MCP, en la web y en los listados |
| A06, A07 | Argumentos tipo «soy CEO» o `role="owner"` no cambian nada; una app `read` no puede escribir ni conceder |
| A08 | Tras una baja, falla con el token antiguo y con uno nuevo; el sobre se borra; la rotación se completa en la siguiente petición de otro miembro |
| A09, A11 | Dos correcciones con la misma versión de partida dan un conflicto; repetir con la misma `idempotency_key` crea un solo recuerdo |
| A12 | `backup`, `verify-backup` y `restore` sobre `orgs/`, conservando políticas y sobres |
| A16 | Una app recién conectada tiene nivel `none` en la organización |
| Transiciones de rol | Secuencias aleatorias con semilla fija de altas, bajas y cambios de rol, comparadas con un modelo de referencia escrito en Python estándar |

Las pruebas de B2 (crear el sobre, abrirlo, rechazar un sobre movido a otro espacio o a otra persona) van en su propio
archivo.

## 6. Preguntas que firma el propietario

- **P-1.** ¿Aceptas B2 (X25519 por cuenta) y su modelo de amenaza (§2.4)? ¿O prefieres C, con un servidor de confianza
  declarado?
- **P-2.** ¿Aceptas D3: el `org_admin` no lee, sus altas quedan pendientes de confirmación y nadie se concede acceso a
  sí mismo?
- **P-3.** Espacios restringidos y el `owner`: ¿siempre incluido (como en el MVP) o con exclusión posible tras revisión
  legal (más adelante)?
- **P-4.** ¿Se exige un segundo `owner` o un procedimiento de sucesión antes del piloto? Sin escrow no hay otra
  recuperación.
- **P-5.** ¿Se mantiene `DEFAULT="all"` en lo personal? Este ADR no lo cambia.
- **P-6.** ¿Es aceptable un límite de unos 2.000 recuerdos por espacio en el piloto (§1.4)?
- **P-7.** **Prioridad:** AGENTS.md fija como hito actual cerrar la distancia con la búsqueda semántica de referencia
  (el baseline RAG). ¿Teams entra en paralelo? Si es así, hay que actualizar AGENTS.md.

## 7. Siguiente paso si se firma

P1 en ramas propias, con el flag apagado y sin datos reales. Se propone un PR por paso:

1. Par X25519 por cuenta, formato de sobre y sus pruebas.
2. `Spaces` sobre `SealedMemories`, `policy.db` y PDP, con la prueba de «denegado = sin descifrar».
3. Las tools `company_list_spaces`, `company_remember` y `company_recall`.
4. Las pruebas A01–A11, A14 y A16 en CI.
5. Copias de seguridad de `orgs/` (A12) y la medición del coste con AES-GCM en CI.

Ninguno se fusiona ni se despliega sin la revisión del propietario.

## 8. Firma

- Propietario: ____________________
- Fecha: ____________________
- Decisiones aceptadas, modificadas o rechazadas (D1–D10, P-1…P-7): ____________________

---

## Apéndice A — Script de medición del §1.4

Ejecutado con `python -I bench_space.py <ruta>/src 500 2000 5000`.

```python
import random, statistics, sys, time
sys.path.insert(0, sys.argv[1])
from infinito3.mcp_server import AgentMemory
from infinito3.persistent_memory import HashEmbeddingProvider, SQLiteCognitiveMemoryStore

WORDS = ("pantalla checkout pago login cache api cliente campaña logo paleta tipografía despliegue staging "
         "migración índice versión presupuesto entrega prueba fallo rendimiento móvil escritorio banner").split()
KINDS = ("decision", "fact", "proposal", "pending")

def statement(i, rng):
    a, b, c = rng.sample(WORDS, 3)
    return f"Registro {i}: en Alfa acordamos {a} con {b} para {c}, revisado por el equipo {i % 17}."

def open_store(image):
    return AgentMemory(store=SQLiteCognitiveMemoryStore.from_bytes(image, HashEmbeddingProvider(256)))

def timed(fn, runs=7):
    times = []
    for _ in range(runs):
        start = time.perf_counter(); fn(); times.append((time.perf_counter() - start) * 1000)
    return statistics.median(times)

rng, memory, done = random.Random(7), open_store(None), 0
for n in [int(x) for x in sys.argv[2:]]:
    while done < n:
        memory.remember(statement(done, rng), kind=KINDS[done % 4], source="bench", project="alfa", date="2026-11-20")
        done += 1
    image = memory.store.to_bytes()

    def read_path():
        m = open_store(image); m.recall("qué acordamos sobre el checkout y el pago", 10, "alfa", None); m.close()

    def write_path():
        m = open_store(image)
        m.remember(f"Nueva decisión {rng.random()} sobre la paleta del banner", kind="decision", source="bench",
                   project="alfa", date="2026-11-20")
        m.store.to_bytes(); m.close()

    print(n, f"{len(image) / 1e6:.2f} MB", f"{timed(read_path):.0f} ms", f"{timed(write_path):.0f} ms")
```
