# INFINITO Teams — ADR P3: acciones del negocio, con piloto en una gestoría

**Estado:** PROPUESTO el 24/11/2026; respuestas del propietario el mismo día (§10). Solo documentación. No autoriza código en producción, despliegue, datos reales,
conexiones con la Agencia Tributaria ni gasto.
**Fecha:** 24/11/2026.
**Base de código:** `master` en `a2e7a43` (comprobado con `git fetch` el 24/11/2026). Las referencias `archivo:línea`
son a ese commit.
**Entrada:** petición del propietario del 24/11/2026: que a INFINITO Teams se le puedan «incrustar herramientas en
modo agente» para cada empresa (dar altas en un hotel, hacer las gestiones informáticas de una gestoría…), con bajo
presupuesto, para que INFINITO «también maneje el negocio». El propietario eligió una **gestoría** como primer piloto.
**Alcance:** producto (INFINITO Teams), no investigación. Se apoya en el ADR P0
(`docs/INFINITO_TEAMS_ADR_P0_2026-11-20.md`) y no cambia nada de lo que ahí se decidió.

---

## 0. Resumen

Las IA que ya usan los empleados (ChatGPT, Claude, Grok…) son agentes: saben llamar a herramientas. Lo que falta es un
puente **seguro** entre esas IA y los programas de la empresa. La propuesta es que INFINITO Teams sea ese puente:
el mismo conector que hoy da la memoria de la empresa daría también sus **acciones**, con los permisos de los
espacios, la aprobación de una persona y el registro de cambios que ya existen.

| # | Decisión propuesta | Por qué |
|---|---|---|
| A1 | Una **acción** pertenece a un **espacio**. Quien no puede abrir el espacio no puede usar sus acciones. | Reutiliza `can()` y las llaves por espacio del ADR P0 sin inventar otro modelo de permisos. |
| A2 | Solo **dos herramientas MCP fijas**: `company_actions` (qué puedo hacer, y en qué quedaron mis peticiones) y `company_act` (pedir una acción). Las acciones de cada empresa son datos, no herramientas nuevas. | El 24/11/2026 se comprobó que ChatGPT guarda la lista de herramientas de cuando se conectó: no vio las `company_*` hasta que el propietario actualizó el conector. Una herramienta por acción obligaría a cada empleado a actualizar su conector cada vez que la empresa añada una. |
| A3 | **Ejecuta el servidor, nunca el modelo.** La IA pide; el servidor valida los argumentos contra el esquema de la acción y llama al adaptador. | La IA no ve credenciales ni direcciones, y no puede saltarse la validación. |
| A4 | Cada acción tiene un **riesgo**: `consulta` (se ejecuta), `aprobación` (espera a una persona en `/teams`) o `directa` (se ejecuta y queda registrada). Por defecto, todo lo que escribe es `aprobación`. | Es el mismo principio de la cola de tareas (`remote_tasks.py:1-17`): ninguna IA aprueba, porque no existe herramienta para aprobar. |
| A5 | Las **credenciales** del programa de la empresa se guardan **selladas con la llave del espacio**, junto a su memoria. La rotación de llave del espacio las vuelve a sellar. | No hay claves persistentes en el servidor: la acción solo se puede ejecutar durante una petición de alguien que tiene abierta la llave del espacio (quien pide, o quien aprueba). |
| A6 | En el MVP, los **adaptadores** (a qué dirección se llama) los instala el administrador del servidor en una lista cerrada. Una organización elige entre los instalados; nunca escribe una URL. | Evita que el servidor llame a direcciones internas (SSRF), como `127.0.0.1:8500`, por orden de una organización. |
| A7 | Un nivel de app nuevo, **`act`**, por encima de `add`. Las apps empiezan en `none`, como hoy. | Leer y guardar memoria no es lo mismo que mover datos de un cliente. |
| A8 | Cada petición deja **una línea en el registro** (quién, con qué IA, qué acción, resultado; nunca los argumentos) y, si la acción lo dice, **un recuerdo** del resultado en su espacio. | Memoria y acción juntas: el negocio recuerda lo que hizo y por qué. |
| A9 | Todo detrás de `INFINITO_TEAMS_ACTIONS_ENABLED=0`, además de `INFINITO_ENTERPRISE_ENABLED`. | Con el flag apagado, el comportamiento actual es idéntico. |

---

## 1. Contexto verificado

### 1.1 Lo que ya existe y se reutiliza

| Pieza | Dónde | Para qué sirve aquí |
|---|---|---|
| Permisos de una organización y sus espacios, decididos en un solo sitio | `remote_teams.py:202` (`can()`) | Quién puede pedir, aprobar y administrar acciones de un espacio. |
| Llave por espacio en sobres sellados a cada miembro | `remote_teams.py:242` (`_space_key`) | Sellar las credenciales y las peticiones pendientes del espacio. |
| Rotación de la llave al quitar a alguien | `remote_teams.py:264-284` (`_opened`) | Hoy vuelve a sellar solo `memory.sealed`; habrá que incluir los archivos de acciones (§3). |
| Registro de cambios, con los rechazos aparte | `remote_teams.py:167` (`_audit`) | Una línea por petición, aprobación, ejecución y fallo. |
| Nivel de cada app por organización (`none`, `read`, `add`) | `remote_teams.py:495-528` (`app_level`, `set_app_level`); `APP_TOOLS` en `:61` | Añadir `act` y exigirlo en `company_act`. |
| Herramientas `company_*`, con la identidad sacada del token y nunca de los argumentos | `remote_teams.py:761` (`add_tools`) | Añadir `company_actions` y `company_act` con el mismo patrón `who()` / `allowed()`. |
| Aprobación solo por una persona en su página, sin herramienta para la IA | `remote_tasks.py:1-17`, `remote_memory.py:746` | El modelo de aprobación de las acciones. |
| Página `/teams` con sesión, CSRF y registro en frases | `remote_teams_page.py` (PR #188) | Sección «Acciones»: alta de acciones y peticiones que esperan aprobación. |

### 1.2 Lo que no existe

- Ningún código de INFINITO llama hoy a sistemas externos en nombre de una organización.
- No hay sitio para credenciales de una empresa: la bóveda (`vault`) es personal.
- No hay esquema de argumentos ni validación de entradas para herramientas definidas por datos.
- La revisión de seguridad independiente de Teams sigue pendiente (ADR P0 §9.4). Con acciones, un fallo deja de ser
  «se lee lo que no se debe» y pasa a ser «se hace lo que no se debe».

### 1.3 Trabajo previo del propietario que sirve de referencia

Asistentes de reservas hechos antes con FastAPI, LangChain y SQLite (`webtilians/backA`, `webtilians/geminiAgent`,
`webtilians/camping-rosaleda-backend`). Muestran el patrón «herramientas sobre una base de datos del negocio», pero
dentro de un chatbot propio. Aquí la IA es la del empleado y las herramientas llegan por INFINITO.

---

## 2. Cómo funciona una acción

### 2.1 Definición (en `policy.json`, en claro, como los nombres de los espacios)

```json
{
  "id": "a1b2c3d4e5f60718",
  "space": "<space_id>",
  "name": "Registrar factura recibida",
  "description": "Añade una factura de proveedor al libro del cliente.",
  "adapter": "gestoria-demo",
  "operation": "registrar_factura",
  "risk": "approval",
  "input": {"cliente": "string<=60", "fecha": "date", "proveedor": "string<=80", "base": "money", "iva": "percent"},
  "remember": "Factura de {proveedor} registrada para {cliente} ({fecha})",
  "approvers": ["owner", "manager"],
  "since": "2026-11-24T00:00:00+00:00"
}
```

- `adapter` y `operation` apuntan a la lista cerrada del servidor (A6). La organización no escribe direcciones.
- `input` es un esquema pequeño y propio (tipos `string<=n`, `date`, `money`, `percent`, `int`, `enum[...]`), validado
  en el servidor. Nada que no esté en el esquema llega al adaptador.
- Las credenciales del adaptador para esta organización van en `orgs/<id>/spaces/<space>/v<version>/actions.sealed`,
  selladas con la llave del espacio (A5). `policy.json` nunca las contiene.

### 2.2 Recorrido de una petición

1. La IA llama a `company_actions(org_id)`. El servidor devuelve las acciones de los espacios que la persona puede
   abrir, con su descripción y su esquema, y el estado de sus últimas peticiones.
2. La IA llama a `company_act(org_id, action_id, args, idempotency_key?)`. El servidor:
   - saca la identidad del token (`who()`), comprueba el nivel `act` de la app (`allowed()`) y `can(..., "act", space)`;
   - valida `args` contra el esquema;
   - si el riesgo es `consulta` o `directa`: abre la llave del espacio en esta misma petición, abre las credenciales,
     llama al adaptador y devuelve el resultado marcado como **datos, nunca instrucciones**;
   - si es `aprobación`: guarda la petición (con sus argumentos) sellada con la llave del espacio y responde «queda a la
     espera de que una persona la apruebe en `/teams`».
3. Una persona con permiso de aprobar ve la petición en `/teams`, con los argumentos legibles, y pulsa **Aprobar** o
   **Rechazar**. Al aprobar, el servidor ejecuta en **esa** petición web, con la llave de quien aprueba.
4. El resultado queda en la petición (sellado), una línea en el registro y, si la acción tiene `remember`, un recuerdo
   en el espacio firmado como `persona · IA`, igual que `company_remember`.
5. La IA lo ve la próxima vez que llame a `company_actions`.

Nada se reintenta solo. Un fallo del adaptador queda registrado y vuelve a la persona.

### 2.3 Adaptadores

Un adaptador es un servicio HTTP pequeño que habla con el programa del negocio (su API o su base de datos) y expone
operaciones con nombre. En el MVP:

- la lista vive en `/etc/infinito-remote/adapters.json` (`id → URL base, operaciones permitidas`), la escribe el
  administrador del servidor y la lee el proceso al arrancar;
- el servidor llama con la biblioteca estándar (`urllib`), con tiempo máximo y tamaño máximo de respuesta, y sin
  seguir redirecciones;
- el núcleo sigue siendo solo biblioteca estándar; cada adaptador es un proceso aparte con lo que necesite.

---

## 3. Seguridad (modelo de amenaza explícito)

| Amenaza | Defensa en el diseño |
|---|---|
| Una IA engañada por un texto (correo, factura, recuerdo) pide una acción que nadie quería | Todo lo que escribe es `aprobación` por defecto; la persona ve los argumentos antes de aprobar. Los resultados vuelven marcados como datos. |
| Una IA o una persona intenta aprobar por su cuenta | No existe herramienta para aprobar. Solo la página `/teams`, con sesión y CSRF. |
| Alguien ejecuta acciones de un espacio en el que no está | `can()` antes de nada; sin sobre del espacio no hay llave, y sin llave las credenciales no se abren. |
| A una persona dada de baja le queda una petición pendiente | Al quitarla, sus peticiones pendientes se cancelan, y la rotación de llave vuelve a sellar `actions.sealed` y las peticiones. |
| Una organización hace que el servidor llame a una dirección interna (SSRF) | Las direcciones solo salen de la lista del administrador del servidor (A6). |
| Las credenciales aparecen en una respuesta, un registro o una copia en claro | Nunca salen de `actions.sealed`; el registro no guarda argumentos; las copias de seguridad llevan los archivos ya sellados. |
| Una IA repite la misma petición muchas veces | `idempotency_key` (una sola ejecución) y un límite de peticiones por persona y acción. |
| El adaptador devuelve muchísimo texto o tarda | Tiempo y tamaño máximos; lo que sobra se corta y se dice. |
| Datos personales y fiscales de los clientes de la gestoría | Minimización: cada operación devuelve solo los campos necesarios. Se necesita contrato de encargado del tratamiento (RGPD) antes de cualquier dato real. |

**Requisito previo a cualquier dato real:** la revisión de seguridad independiente de Teams (ADR P0 §9.4), que ahora
incluye las acciones.

---

## 4. Piloto: «Gestoría Demo» (datos inventados)

### 4.1 Qué es

Un adaptador de demostración, `adapters/gestoria_demo/`: un proceso con `http.server` y `sqlite3` de la biblioteca
estándar, sobre una base de datos con **datos inventados**:

- `clientes`: nombre, tipo (autónomo o sociedad), identificador ficticio;
- `obligaciones`: cliente, modelo (por ejemplo 303), periodo, vencimiento ficticio, estado;
- `facturas`: cliente, emitida o recibida, fecha, base, tipo de IVA;
- `expedientes`: cliente, asunto, responsable, estado.

### 4.2 Operaciones del piloto

| Operación | Riesgo | Qué hace |
|---|---|---|
| `consultar_vencimientos` | consulta | Obligaciones de un cliente, o de todos, entre dos fechas. |
| `estado_cliente` | consulta | Expedientes abiertos y documentación que falta. |
| `registrar_factura` | aprobación | Añade una factura al libro del cliente. |
| `preparar_borrador_303` | aprobación | Calcula un borrador del IVA del trimestre a partir de las facturas y lo guarda como borrador **para revisión del gestor**. |
| `marcar_presentada` | aprobación | Anota que una obligación ya se presentó. Lo presenta una persona, fuera de INFINITO. |

**Fuera del piloto, a propósito:** presentar nada ante la Agencia Tributaria o la Seguridad Social, usar certificados
digitales, mover dinero, enviar correos a clientes y cualquier dato real.

### 4.3 Guion de demostración

1. En «Estudio Demo» (o en una organización «Gestoría Demo»), el propietario da de alta las cinco acciones en un
   espacio «Clientes».
2. Ana, desde ChatGPT: «¿Qué le vence este mes a Panadería Sol?». Es una consulta y se ejecuta al momento.
3. Ana: «Registra la factura de Harinas del Sur de 1.200 € más IVA para Panadería Sol». La IA pide y queda esperando.
4. El propietario la ve en `/teams`, revisa los datos y aprueba.
5. Ana, desde Claude: «¿Cómo va el borrador del 303 de Panadería Sol?». La IA ve la factura y el recuerdo que dejó la
   acción, con quién la pidió, quién la aprobó y con qué IA.

### 4.4 Piloto real después

Paszoom (app de pagos y gestoría de Pascual, que ya es miembro de la organización `paszoom`) es el candidato natural si
tiene una API y Pascual quiere. No se sabe todavía si tiene API: es una pregunta (§7).

---

## 5. Alcance del MVP (P3-1)

1. `remote_teams.py`: acciones en `policy.json`, `can(..., "act" | "approve" | "manage_actions", space)`, nivel `act`,
   peticiones selladas, cancelación al quitar a alguien y rotación que vuelve a sellar todos los archivos de la versión.
2. `remote_teams_actions.py` (nuevo, solo biblioteca estándar): esquema y validación de entradas, lista de adaptadores,
   llamada HTTP con límites.
3. Herramientas `company_actions` y `company_act` en `add_tools`, detrás de `INFINITO_TEAMS_ACTIONS_ENABLED`.
4. `/teams`: sección «Acciones» del espacio (alta, baja y lista) y bandeja «Esperan tu aprobación».
5. `adapters/gestoria_demo/` con datos inventados y su propio README.
6. Pruebas (§6). Toda la batería de Teams sigue en verde.

No entra en P3-1: adaptadores escritos por la organización, acciones encadenadas sin persona, ejecución programada,
y que el motor cognitivo use acciones.

---

## 6. Pruebas exigidas en P3-1

- Con el flag apagado no hay herramientas nuevas, ni sección en la página, ni cambio de comportamiento.
- Una app en `add` no puede usar `company_act`; en `act`, sí, y solo en espacios que la persona puede abrir.
- La identidad sale del token: ningún argumento cambia quién pide ni quién aprueba.
- Una acción `aprobación` nunca se ejecuta antes de que una persona apruebe en la página; ninguna herramienta aprueba.
- Quien no tiene permiso de aprobar no ve el botón, y su envío falla y queda en la lista de rechazos.
- Quitar a alguien cancela sus peticiones pendientes, y la rotación vuelve a sellar credenciales y peticiones.
- Las credenciales no aparecen en respuestas, registro, `policy.json` ni en los mensajes de error.
- Una URL fuera de la lista del servidor no se puede usar, ni siquiera editando `policy.json` a mano.
- Argumentos de más, de menos o con tipo erróneo se rechazan antes de llamar al adaptador.
- La misma `idempotency_key` ejecuta una sola vez, aunque lleguen dos peticiones a la vez.
- Tiempo agotado, error del adaptador y respuesta demasiado grande quedan registrados y no se reintentan.
- El registro guarda quién, qué acción y el resultado, nunca los argumentos.
- La memoria personal sigue sin ninguna ruta hacia las acciones.

---

## 7. Preguntas que firma el propietario

- **P-1** ¿Quien pide una acción puede aprobarla él mismo, o hace falta siempre una segunda persona (doble control)?
  Recomendación: segunda persona por defecto, y que el propietario pueda permitirlo acción por acción.
- **P-2** ¿Quién da de alta acciones en un espacio: solo el propietario, o también su responsable?
- **P-3** ¿Se guarda un recuerdo de cada acción ejecutada, y con qué datos? Recomendación: solo el texto de `remember`,
  sin importes ni identificadores, salvo que la empresa lo pida.
- **P-4** ¿Adaptadores solo instalados por nosotros en el MVP? Recomendación: sí (A6).
- **P-5** ¿Va esto antes que P2 (departamentos y destino automático) y que el hito de memoria frente a RAG de
  `AGENTS.md`?
- **P-6** ¿Tiene Paszoom una API, y quiere Pascual que sea el primer piloto real?
- **P-7** Modelo de negocio: ¿se cobra la integración de cada cliente aparte de la suscripción? Cada gestoría trabaja
  con sus programas; está por comprobar cuántas comparten los mismos y si un adaptador sirve para varias.

---

## 8. Siguiente paso si se firma

1. Rama desde `master` para P3-1, en PR en borrador.
2. Primero `remote_teams_actions.py` y sus pruebas (esquemas, lista de adaptadores, límites), sin tocar las
   herramientas.
3. Después permisos, sellado y rotación en `remote_teams.py`, con sus pruebas.
4. Después las herramientas, la página y el adaptador de demostración.
5. Prueba de principio a fin en un servidor local con datos inventados; activación en el servidor solo con el visto
   bueno del propietario, y solo en una organización de demostración.

---

## 9. Firma

- Propietario: ____________________
- Fecha: ____________________
- Decisiones aceptadas, modificadas o rechazadas (A1–A9, P-1…P-7): ____________________

---

## 10. Adenda 1 (24/11/2026): respuestas del propietario y cambios

### 10.1 Respuestas

- **P-1** «Las acciones las aprueba uno mismo», y cada modelo de negocio será distinto: en un camping o un hotel la
  misma persona hace su alta; en una empresa de carteles, alguien pide tóner o consulta proyectos.
- **P-2** Las acciones las damos de alta **nosotros**: se estudian las necesidades de la empresa y después se crean
  sus acciones y su modelo de negocio. Además, a Teams se le podrán añadir **disparadores (dispatch, triggers)** que
  lancen eventos de la empresa.
- **P-3** De cada acción se guarda un recuerdo con **quién la hizo, cuándo, y los datos de la acción o su
  referencia**. Si los datos se pueden recuperar (por ejemplo, una fila de una base de datos), se guarda el
  identificador, no toda la información.
- **P-4** Sí: los adaptadores los instalamos nosotros.
- **P-5** Es un complemento de Teams. Ahora no se configuran herramientas concretas de ninguna empresa (como mucho una
  demo en la portada): se deja **preparado** para conectarlas cuando haga falta, y `/teams` **muestra las herramientas
  que tiene conectadas** cada empresa.
- **P-6** Paszoom no, de momento. En la presentación del 25/11/2026 se comentará y se ofrecerá preparar un prototipo
  para la empresa que lo quiera.
- **P-7** La integración de cada cliente se cobra aparte, y su precio depende de lo que cueste.

### 10.2 Cambios en las decisiones

- **A4 (riesgo → modo).** Cada acción lleva un **modo** que fijamos al darla de alta, según el negocio:
  `direct` (se ejecuta al pedirla), `self` (la confirma en `/teams` quien la pidió; es el modo por defecto, por P-1) y
  `second` (la confirma otra persona con permiso). En todos los modos sigue sin existir una herramienta para
  confirmar: confirma una persona en la página, nunca una IA.
- **Catálogo del servidor (A1, A6, P-2).** Las acciones y los adaptadores de cada organización viven en un archivo que
  solo escribe el administrador del servidor (`INFINITO_TEAMS_ACTIONS_FILE`; por defecto `actions.json` junto a los
  datos, permisos `0600`). Las organizaciones lo ven en `/teams`, pero no lo editan. No entra en las copias de
  seguridad, porque es configuración con secretos y su original lo guardamos nosotros.
- **Credenciales (A5).** Como el alta la hacemos nosotros sin la llave de ningún espacio, las credenciales del
  programa de la empresa se quedan **en el adaptador**, fuera de INFINITO. INFINITO solo guarda, en el catálogo, el
  secreto con el que se identifica ante cada adaptador. Lo que sí va sellado con la llave del espacio son los
  **datos de cada petición** pendiente y su resultado, que la rotación de llave vuelve a sellar.
- **Recuerdo de cada acción (A8, P-3).** Al terminar, la acción deja un recuerdo en su espacio:
  «`persona` hizo «`acción`» con `IA`: `referencia o datos`». Si el adaptador devuelve una referencia (`ref`), se
  guarda solo esa; si no, los argumentos en forma compacta. Una acción puede no dejar recuerdo (`remember: false`),
  pensado para consultas repetitivas.
- **Disparadores y eventos (P-2), en dos sentidos:**
  - *De INFINITO a la empresa:* un disparador es una acción más cuyo adaptador lanza un evento en los sistemas de la
    empresa (un webhook, una automatización). Entra en P3-1 sin código nuevo.
  - *De la empresa a INFINITO:* que un sistema de la empresa avise («ha llegado un pedido») y quede en la memoria del
    espacio. Llega cuando nadie tiene la llave abierta, así que necesita un buzón por espacio: una clave pública con la
    que cualquiera sella el evento al llegar, y cuya mitad privada va sellada con la llave del espacio; el buzón se
    vacía en la memoria la próxima vez que un miembro abre el espacio. Queda para **P3-2**.
- **Alcance de P3-1 (P-5).** El marco y su vista, sin herramientas de ninguna empresa real: catálogo, `company_actions`
  y `company_act`, modos y confirmaciones, recuerdo de cada acción, nivel de app `act` y la sección «Herramientas
  conectadas» de `/teams`. El adaptador de la gestoría pasa a ser un ejemplo opcional para una demo, no parte del MVP.
