# INFINITO S0-J2 — banco nuevo y sellado para competencia con memoria: GPT frente a Qwen

**Fecha:** 2026-11-10. **Prerregistro antes del banco, código y de cualquier inferencia S0-J2.** Se apila sobre el S0-J1 PR #159 exclusivamente para reutilizar fuentes y funciones. **No fusionar ni desplegar.**

## Pregunta principal

¿Puede un modelo GPT de mayor capacidad interpretar recuerdos episódicos y citar literalmente las observaciones que justifican cada respuesta mejor que Qwen 7B/14B, manteniendo **exactamente el mismo banco de casos nuevo** y separando el efecto de la memoria del efecto de un formato resumido? Esto mide **interpretación de evidencias conocidas**, NO aprendizaje/adaptación, detección de vulnerabilidades, ni continuidad de identidad. Se conserva F12-U como confusor de estado de backend: cada petición API GPT es independiente, sin encadenamiento ni herramientas; un futuro brazo Qwen tendrá que reiniciar instancia para cada caso y reportar los parámetros por separado.

## Banco sellado (solo datos sintéticos)

- Ocho **perfiles completamente nuevos**: aurora, bracken, copper, drift, ember, glen, fable, hush.
- Por perfil, tres *servicios de nombre opaco* únicos; un antecedente r1 y dos observaciones r2. TOTAL: **24 episodios, 72 llamadas de herramienta read-only** sobre clon OAuth/MCP LOCAL aislado, con `overlay` sintético de un marcador exclusivo del laboratorio. Ningún dato de cuentas reales, memoria personal ni credenciales en los prompts.
- Seis perfiles r2 positivos: **dos `recall`, dos `list_memories` y dos `vault_list`**. Dos perfiles r2 negativos en las tres herramientas: **fable y hush**. La distribución está fijada antes de resultados; r1 difiere de r2 en los positivos.
- Los detalles de servicios, posiciones y etiquetas se fijan en los ficheros de fixture del primer commit de banco; **no cambiarlos tras ejecutar ningún modelo**. El SHA del commit de banco y del fixture viajan en el manifiesto de resultados. El modelo NO recibe tablas de respuesta, código del validador, ruta de repo ni el fixture con etiquetas. Solo recibe el contexto asignado y la pregunta.
- **Una sola pregunta inédita por perfil**; sin paráfrasis duplicadas. Orden de perfiles fijo y congelado. Así 8 observaciones de caso por brazo.

## Brazos y metodología

Misma instrucción S0-J0 `SYSTEM`, mismo esquema de respuesta y misma consigna de r2; se usan la consulta Q0 de S0-J0 y servicios completamente nuevos. Los tres brazos son:
- **N0 (sin evidencia)**: texto indicando que no hay observaciones; el comportamiento deseable es `unknown` con `evidence_service=NONE`, evaluado como abstención por separado de aciertos contra etiquetas no observables.
- **N1 (memoria real completa)**: `AgentMemory.recall` de las 24 experiencias almacenadas en SQLite de laboratorio. No reescribir ni reorganizar manualmente.
- **N3 (tarjeta mecánica)**: exclusivamente las tres experiencias del perfil consultado, ordenadas como r1, r2, r2 y con positivos/negativos leídos del clon. Ningún oracle ni `private_target` en construcción del texto.

**Métricas originales S0-J0**: por brazo 0–8 herramientas correctas, 0–8 respuestas `grounded` que exigen además `evidence_service` literalmente igual al nombre de uno de los dos servicios observados r2 del perfil, 0–8 citas válidas, 0–8 JSON exactos, quiet correcto 0–2, recuperación completa de ambas evidencias r2, tokens, latencia, hash de entrada, salida raw y modelo exacto. Para negativos, `tool=none` exige citar literalmente uno de los dos servicios r2 realmente observados; `evidence_service=NONE` NO es cita válida. No normalizar `service ...`, no reparar JSON, no hacer retries de respuestas, no postcorregir.

**Gate congelado:** `READ_COMPETENCE_SIGNAL` si N1 (o N3 en clasificación `FORMAT_LIMITED`) alcanza >=6/8 herramientas correctas, >=6/8 grounded, ambos negativos correctos y 8/8 JSON válidos; `NO_READING_SIGNAL` si no; `INVALID_FORMAT` si al menos una salida rompe el formato exacto; `BLOCKED` si no hay datos completos/aislamiento/API/modelo. Señalar N0 `unknown` 8/8 como abstención correcta (no cuenta como herramienta acertada). No inflar las métricas por la ausencia de evidencia.

## Control de GPT y comparabilidad

- Objetivo de modelo API inicial **`gpt-6.1-sol`** (no afirmar que sea el runtime exacto del ChatGPT actual). Se verifica disponibilidad auténtica en la cuenta antes de inferir, se anota el ID efectivo devuelto y la versión del SDK; si no existe acceso, **BLOCKED/NOT_RUN**, sin sustitución oportunista después de mirar resultados. API Responses: cada pregunta en **una petición separada** sin estado compartido, sin conversation/previous_response_id, sin herramientas, `store=false`, sin prompts de resultados ni puntuación. Guardar respuestas raw, usos y errores. No imponer JSON schema del servidor ni dar ejemplos respondidos. Los parámetros de sampling no soportados en modelos de razonamiento no se envían; se registra esta diferencia frente a Qwen temp0.
- El ChatGPT de esta conversación **no es** una instancia experimental ciega: ya ha visto el objetivo y resultados previos; no usar una respuesta mía aquí como prueba de una corrida independiente.
- Dataset construido por un job OFFLINE antes de toda llamada a API, tras conectar solo a un clon efímero OAuth/MCP. El job GPT solo recibe los **prompts de datos sintéticos**; no cuenta con las claves para el clon ni con la tabla privada de etiquetas en sus mensajes API. El evaluador local reconstruye verdad de las observaciones capturadas y compara raw. Ninguna clave API se escribe en logs.
- `OPENAI_API_KEY` se inyecta **solo como secreto de Actions**, nunca en código, INFINITO ni chat. El workflow por defecto solo comprueba contratos y sella evidencias. La fase GPT requiere `workflow_dispatch` explícito; si no hay credencial/acceso a modelo, deja informe de bloqueo en vez de inventar resultados.
- Para comparación causal de **capacidad de modelo**, Qwen 7B y 14B deben correr después sobre las **mismas 24 preguntas** sin editar banco ni prompts, con instancias frescas por caso y parámetros registrados. No comparar porcentajes S0-J0/J1 como si fueran un benchmark pareado: usaban otros casos y condiciones.

## Integridad y límites

Comprobar: 8 perfiles distintos; 24 mundos r1/r2/r2; 72 llamadas reales de clon; 24 recuerdos almacenados; servicios opacos únicos; dos negativos; balance 2/2/2 en r2; sin repetición con S0-I PILOT/CONFIRM o F12 transfer; 0 exposición de etiquetas a la API; 24 respuestas GPT reales completas antes de una decisión final. Los artefactos quedan en la PR y Actions con SHA y trazas. No fusionar, no desplegar, no reclamar transferencia o aprendizaje.
