# INFINITO Security S0-B — protocolo antes de la implementación

**Fecha:** 2026-11-09 · **Base:** S0-A PR #136 · **Estado:** hipótesis / ensayo técnico. No desplegar.

## Pregunta y ámbito

¿Pueden agentes adversariales **deterministas y acotados** realizar llamadas a las herramientas MCP auténticas mediante sesiones OAuth 2.1 reales sobre un servidor HTTP clonado, sin romper barreras de cuenta, autorización ni confidencialidad de la bóveda? ¿Los textos con instrucciones maliciosas permanecen presentados como **datos**? Este último punto no evalúa la reacción de un LLM: todavía no hay inferencia de modelos ni aprendizaje.

## Hipótesis

Con datos y cuentas ficticias, las herramientas MCP remotas conservan las fronteras de identidad, rechazo de tokens inválidos, privilegios y bóveda, incluso cuando un agente selecciona la siguiente acción según el resultado anterior.

## Ejecución y aislamiento

- Clon del servidor real `build_app` sobre loopback `127.0.0.1` dentro de Docker `--network none`, sin puertos publicados ni montajes, usuario no root, directorios efímeros. `People` y `Accounts` solo apuntan a `tmp_path`; NO copiar instancias reales.
- Solo canarios sintéticos y frases ficticias; contraseñas maestras de test sin uso fuera del laboratorio. Los resultados públicos omiten URLs con tokens, sesiones, secretos, textos y claves.
- Protección adicional de tests: abortar cuando existan variables conocidas de despliegue; no conectar a otras IP distintas de loopback.
- El atacante es un **agente por política finita que reacciona al feedback**, con presupuesto fijo y catálogo cerrado de herramientas. No se permite que el agente escriba código, abra sockets o decida dominios arbitrarios.
- El validador es independiente de las declaraciones del agente; analiza respuestas observadas del MCP y los estados sintéticos. El informe no afirma éxito científico de un LLM.

## Baseline y escenarios congelados

B0. Realizar login OAuth signup de Ana y Bruno con el cliente MCP real; control positivo: cada cual puede consultar su propia memoria.
B1. Petición MCP sin bearer y con token inexistente: 401. Acceso con OAuth válido de Bruno: nunca devuelve el canario de Ana.
B2. Agente de Bruno prueba `list_memories`, `recall` y `forget` con id ajeno. El contenido y la escritura del propietario siguen aislados.
B3. Cambiar por configuración de cuenta el nivel de la app de Bruno a `read`. La escritura `remember` y `forget` son rechazadas por el servidor (no solo por una respuesta simulada).
B4. Texto adversarial alojado como recuerdo de Bruno solicita revelar el marcador de Ana. `recall` lo devuelve como contenido no confiable; no aparece en las respuestas de Ana. **No se prueba obediencia de un LLM.**
B5. Bóveda real KDBX de Ana con valor ficticio introducido en una página del clon. `vault_list` y `vault_reveal` no exponen el valor. Bruno no puede abrir la bóveda de Ana; sin invitación la herramienta rechaza, y con invitación su propia bóveda sigue vacía.
B6. Un enlace temporal de `vault_reveal` devuelve una página que pide la clave maestra, no el secreto directamente; no simular entrada de clave maestra por el agente.
B7. Los agentes no ejecutan herramientas fuera del catálogo ni superan el presupuesto fijado; resultados expresados solo como checks y contadores.
B8. Docker sin red externa, sistema de ficheros raíz de solo lectura, recursos limitados.

## Gates y decisiones

**PASS técnico**: todos los controles de privacidad, autorización, perímetro, presupuesto e integridad pasan contra el clon real (incluyendo al menos una sesión autenticada y llamadas MCP verdaderas). Cualquier fuga de canario entre cuentas, valor secreto en respuestas MCP, acceso a producción o ruptura de los límites = **FAIL**. No adaptar gates a resultados vistos.

**Siguiente fase (S0-C):** preregistrar un estudio con un LLM competente real, control sin memoria frente a memoria INFINITO, igual presupuesto y mundos nuevos, midiendo vulnerabilidades reproducibles y falsos positivos. Hasta entonces está prohibido afirmar mejoras de aprendizaje, protección completa contra prompt injection o preparación comercial.

## Fuera de alcance explícito

No fuzzing genérico ni ataque al servidor de producción, ni explotación libre, ni agentes LLM, ni evaluación de proveedores externos o IoT, ni prueba exhaustiva de disponibilidad o concurrencia. La propia librería de tests usa el login OAuth y la página de la bóveda como fixtures: cualquier fallo de fixture bloquea el resultado. Se mantienen S0-A y su PR sin modificar.

Referencia: https://github.com/webtilians/principiodelTodo/issues/134
