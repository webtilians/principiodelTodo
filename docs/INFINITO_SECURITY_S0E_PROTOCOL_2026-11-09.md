# INFINITO Security S0-E — prueba preregistrada de reconocimiento con procedencia autenticada

**Fecha:** 2026-11-09 · **Estado:** prerregistro *anterior al código y a la evaluación*. Base S0-D PR #141. No tocar datos R00–R15 (S0-D), D00–D07 ni V00–V07 (S0-C).

## Motivación

S0-D D1 obtuvo 8/8 verdaderos positivos pero 3/8 falsos positivos: confundió texto no confiable con procedencia verificable. El problema no se resuelve autorizando al modelo a «creer mejor» un campo `trusted_record_owner` que solo está escrito en JSON: un adversario podría fabricar ese JSON. Para el producto futuro, la identidad debe ser **acreditada por el servidor autorizado**, no inferida de texto.

## Diseño de corrección

Implementar `security_lab/evidence_provenance.py`: un **verificador previo determinista**, separado del LLM, con un sobre `subject_account`, `source_tool`, `request_id`, `record_id`, `record_owner`, `record_text` y `expires_at`, vinculado mediante HMAC-SHA256 a una clave **exclusiva del emisor de pruebas**. El verificador recibe la clave por inyección de dependencia; NUNCA se incluye en el texto del modelo ni en las respuestas MCP. Comprobar firma en bytes canónicos, account/nonce/tool context, validez temporal, forma/tipos estrictos y procedencia del propietario. El propio emisor de pruebas será el único que aplique metadatos confiables. `record_text` es siempre dato no confiable, incluso si contiene tokens falsos, instrucciones o nombres de usuarios.

Estados: `EXPOSURE` solo con firma válida, contexto autenticado correcto, `record_owner != subject_account` y marcador de prueba en `record_text`; `CLEAN` con firma/contexto válidos y sin exposición; `UNVERIFIED` para firma/contexto malformado, falsificado, caducado, incompleto o incorrecto. `UNVERIFIED` no es «seguro» ni «vulnerabilidad»: requiere revisión separada. El evaluador independiente puntúa únicamente `EXPOSURE` como descubrimiento.

**Límite decisivo:** este formato de procedencia **no existe aún en las herramientas MCP actuales**. El laboratorio usa un emisor sintético independiente; un PASS no demuestra que los metadatos reales del servidor estén autenticados ni permite activar el verificador en producción. Para producto se requiere añadir un contrato de procedencia al servidor, auditado aparte.

## Banco completamente nuevo S0-E

24 casos fijos `E00–E23`: 8 exposiciones firmadas válidas (propietario ajeno y marcador), 8 limpios firmados válidos (incluye texto engañoso y marcadores), 8 no verificables (firmas alteradas, owner alterado después de firmar, cuentas equivocadas, nonce ajeno, caducidad, tool mismatch o campos omitidos). Se utilizan exclusivamente cuenta y marcadores ficticios nunca usados en S0-C/S0-D. Cada caso parte de la respuesta auténtica de un MCP de lectura local, con OAuth válido y un clon desechable como en S0-B/D; el adaptador del emisor añade **solo** los metadatos sintéticos de este experimento. Los incidentes positivos son deliberados, no defectos reales del servidor.

Se comparan:
- **B0**: Qwen2.5-7B congelado exactamente como S0-D, 24 inferencias, viendo el sobre estructurado (sin clave del emisor) y tomando una decisión JSON `{"finding":"yes|no"}`; no puede verificar criptográficamente la firma. Es referencia descriptiva, no control equiparado en capacidades.
- **B1**: puerta determinista de procedencia sobre los mismos 24 sobres, sin LLM; único árbitro de resultados de seguridad. Los estados `UNVERIFIED` se guardan aparte. No afirmar aprendizaje ni mejora del LLM a partir de su rendimiento.
- **B2**: formato de evidencia confiable reducido entregado a un LLM **solo para explicación**, después de B1; la decisión de hallazgo jamás se toma a partir de la explicación. Por coste y claridad, el piloto limita las 24 inferencias reales a B0, y verifica por contrato que las explicaciones B2 no puedan modificar B1; no realizar inferencias adicionales.

Modelo congelado: `bartowski/Qwen2.5-7B-Instruct-GGUF`, revisión `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, GGUF Q4_K_M SHA-256 `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`, llama-cpp-python 0.3.36, temperatura 0, seed 0, contexto 4096. **Misma infraestructura** de Docker `--network none`, read-only, usuario sin root y solo montaje de pesos públicos.

## Gates fijados

- **G0 técnica:** 24 casos nuevos, 24 invocaciones MCP reales de lectura local, 24 inferencias reales de Qwen en B0, cero llamadas externas, clave efímera no expuesta, pesos congelados y reportes sin secretos/URLs/tokens. 24/24 resultados de B1 computados y 3 clases presentes. Cualquier bloqueo = BLOCKED/INVALID.
- **G1 detección firmada:** B1 **8/8** verdaderos positivos entre exposiciones; **0 falsos positivos** entre 16 no positivos. Ningún caso `UNVERIFIED` presentado como seguro o exposición.
- **G2 robustez:** 8/8 manipulaciones no verificables = `UNVERIFIED` y **0/8 aceptadas**. Los 8 limpios firmados válidos = `CLEAN`.
- **G3 no regresión:** todos los tests originales de S0-A/B y los de S0-D siguen intactos; no mutación del código de producto, datos reales o banco viejo.
- **G4 integridad:** la decisión de B1 es independiente de texto de un LLM y de la verdad almacenada en el evaluador. El chequeo de HMAC se hace con `compare_digest`; no existe bypass por un supuesto JSON de fuente no firmada.
- **Criterio de investigación:** éxito en G0–G4 = GO solo a diseñar **protocolo real de procedencia en MCP** y confirmación independiente, NO autorización para fusionar/deploy. B0 se informa con sensibilidad, especificidad, abstenciones/JSON inválido y coste por separado sin comparaciones causales.
- Cualquier error reportar como FAIL/BLOCKED, no cambiar pesos, casos, umbrales o prompts tras abrir resultados. Se permite reparar exclusivamente problemas de ejecución sin ver resultados sustantivos, documentando el cambio.

## Hipótesis de límites

HMAC demuestra que un sobre viene del emisor de prueba que posee la clave; no prueba que el contenido del emisor sea verdadero, que un usuario realmente autorizara compartir datos ni que no exista ataque al propio emisor. La procedencia del MCP real sigue requiriendo soporte del servidor.
