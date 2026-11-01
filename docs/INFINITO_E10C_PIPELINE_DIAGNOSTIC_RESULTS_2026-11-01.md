# INFINITO E10-C — Diagnóstico completo de la tubería de memoria

**1 de noviembre de 2026 — completado, SIN abrir una nueva confirmación ni llamar al LLM.**

## Procedencia y límites
- [Protocolo congelado antes de ejecutar](./INFINITO_E10C_PIPELINE_DIAGNOSTIC_PROTOCOL.md).
- [GitHub Actions #41308525833 — SUCCESS, tests y prueba semántica real](https://github.com/webtilians/principiodelTodo/actions/runs/41308525833).
- [Artifact de las 12 historias DEV, 48 etapas por brazo (todos los IDs, cadenas, rangos, razones y contextos)](https://github.com/webtilians/principiodelTodo/actions/runs/41308525833/artifacts/13083614909).
- [PR #30, borrador sobre la rama E10-B, sin cambios en producción](https://github.com/webtilians/principiodelTodo/pull/30).
- **12 mundos NUEVOS de desarrollo** \`1900000+index*1000+j\` (2×6 familias); 2 pistas ×2 momentos =**48 muestras de evidencia por brazo**. Particiones confirmatorias E10-B \`1800000+\` y F9 **no utilizadas**.
- **Embeddings OpenAI reales \`text-embedding-3-small\`** en los tres brazos; **0 llamadas de generación LLM**, 0 puntuaciones de decisiones. El modelo no se simula para crear resultados de calidad. Mismo \`top_k=8\` y techo 320 tokens estimados de evidencia.
- Núcleo \`src/infinito3/\` **intacto**; contratos y 75 regresiones existentes aprobaron la CI. Las rutas son idénticas al pipeline vigente, inspeccionando \`CognitiveDecision.context\` y \`ContextPacket.items\` del **mismo** turno, sin llamar una segunda vez a \`search\`.

## Resultado por primera pérdida de evidencia, 48 ocasiones por brazo

| Brazo | Hecho vigente almacenado | Registro activo | Entre los 8 recuperados | Elegido para contexto | Texto enviado al LLM simulado SOLO como solicitud |
|---|---:|---:|---:|---:|---:|
| RAG semántico cronológico | **48/48** | **48/48** | **48/48** | **48/48** | **48/48** |
| **INFINITO 3.0 congelado** | **16/48** | **16/48** | **16/48** | **16/48** | **16/48** |
| **INFINITO con gate abierto** | **48/48** | **48/48** | **48/48** | **40/48** | **40/48** |

Ninguna de las 48 muestras mostró que una memoria recuperada y seleccionada desapareciera durante la serialización/transmisión del contexto. Tampoco hubo fallos de permanencia ni de ranking **top-8** para el hecho vigente una vez almacenado en estos fixtures. **La falta de entrega provino de dos etapas diferentes: captura y filtrado de contexto.**

## Desglose por familia: recuerdos efectivamente transmitidos

| Familia (8 decisiones por brazo) | RAG | INFINITO original | Gate abierto |
|---|---:|---:|---:|
| Avisos | 8/8 | **0/8 — no capturados** | 8/8 |
| Formatos | 8/8 | **0/8 — no capturados** | 8/8 |
| Escalados | 8/8 | **0/8 — no capturados** | 8/8 |
| Frecuencia de revisiones | 8/8 | **0/8 — no capturados** | **0/8 — recuperados pero filtrados** |
| Color favorito | 8/8 | 8/8 | 8/8 |
| Ciudad actual | 8/8 | 8/8 | 8/8 |

En original la primera pérdida se concentra en **32/48 casos**, todos de cuatro familias nuevas. La captura heurística \`RuleBasedMemoryGate.evaluate\` se dispara para expresiones como \`mi color favorito es\` o \`vivo en\`, pero las frases operativas nuevas no alcanzan su umbral. Por ello INFINITO **no guardó** esos hechos, con independencia de la calidad semántica posterior del vector.

## Causa concreta del caso de «frecuencia de revisión»

El caso de cadencia mostró un fenómeno diferente en las 8/8 trazas del gate abierto:

- La declaración vigente **sí estaba almacenada y activa** en SQLite.
- La búsqueda semántica del **mismo** \`CognitiveEngine.process\` la encontraba entre sus 8 candidatos. En varios casos era el **primer candidato**, por delante de los distractores.
- Sin embargo, \`ContextPacket.items\` tenía **0 recuerdos**, \`precision_dropped=8\`, \`selected_by_source.user_model=0\`. El presupuesto sobrante era 288/320 tokens estimados; **no** fue falta de espacio.
- **Causa identificada en el código:** \`BalancedContextBuilder._asks_for_goals\` hace una búsqueda regex de \`plans?\` sin limitarla a peticiones explícitas sobre tareas; por ello la palabra española **«plan»** del enunciado «frecuencia de revisión del plan ...» se interpreta como una solicitud de *goals*. 
- Después, \`GeneralizedContextBuilder._apply_precision_filter\` comprueba \`asks_goals=True\`, pero \`requested_facts\` queda vacío. Como \`StoreAllGate\` había guardado todos los eventos como \`USER_MODEL\`, ningún \`ContextItem\` de ese grupo supera la rama de filtrado: **se pierden los 8 candidatos recuperados**.

Esto se verificó para los 2 mundos nuevos de cadencia (ambos cues, antes y después), y el regex \`plans?\` coincidió con «plan» únicamente en dicha familia. Es una colisión de intención **entre idiomas**, no un fallo de similitud semántica ni un problema de longitud de memoria. El diagnóstico causal de línea de código se apoya en el examen estático de la condición + traza operacional; **aún no se ha probado en un experimento LLM una política corregida**.

## Del contexto a la acción: evidencia histórica y límite de esta auditoría

La confirmación E10-B, antes de esta auditoría, había observado que en INFINITO original el dato correcto estaba ausente del contexto en **64/96** decisiones y presente en otras 32, de las cuales **7** acabaron en \`ABSTENER\`. En gate abierto hubo **16/96** ausencias y **8 abstenciones con dato visible**; el RAG acertó **96/96** decisiones.

**E10-C NO ejecutó un LLM:** no podemos atribuir esas abstenciones posteriores exclusivamente al modelo, al wording, a la temporalidad o al formato de los recuerdos. Es un problema real de evidencia→decisión cuya prueba causal específica queda pendiente; no debe disfrazarse de una tercera pérdida en la tubería medida aquí.

## Interpretación científica

Los resultados soportan **dos defectos de diseño localizados, observados sin cambiar el núcleo**:

1. **Captura poco general:** el gate decide qué conservar mediante patrones léxicos. Al cambiar a acuerdos naturales no vistos, descarta la información que después importa.
2. **Falso positivo de intención:** el término \`plan\` coincide con un indicador de goals en inglés y el constructor elimina todos los recuerdos de perfil, incluso cuando el recuperador devolvió correctamente la información.

Las diferencias entre el baseline RAG y el núcleo **no implican** que la memoria cognitiva aprendida sea inferior por naturaleza. E10-B compara diferentes reglas de captura, presentación cronológica y contexto efectivo. Pero **sí implica** que el INFINITO 3.0 actual no tiene demostrada ventaja práctica sobre un RAG semántico sencillo. El éxito offline 128/128 de E10-A era específico de hechos estructurados.

## Decisión de parada / recomendación acotada

**E10-C está COMPLETADO y cerrado.** No producir F9 adicionales ni nuevas rondas de diagnóstico. El siguiente paso más económico no es construir una nueva red ni añadir 15 reglas particulares:

- **Corregir como regresión independiente y controlada** la coincidencia accidental de \`plan\` con el patrón de goals \`plans?\` (distinción de intención de tareas vs tema del plan). Es un defecto localizado y se puede cubrir con tests ES/EN, sin meter listas manuales de dominios.
- **Mantener RAG semántico con cronología como baseline de referencia** y contrastar **una sola** política generalizable de captura de hechos en conversaciones naturales, con límites explícitos de coste, privacidad y contaminación por ruido. Ese cambio debe buscar ventaja frente a RAG real, no solo recuperar los ejemplos de E10-B.
- Antes de cualquier nuevo ensayo confirmatorio LLM, escribir un **nuevo protocolo y nuevas semillas**, con controles adversariales, cambios reales de preferencias, presupuesto de tokens usado y episodios independientes; nunca reutilizar \`1800000+\`.
- Si ese único cambio no supera un baseline semántico con igualdad razonable de presupuesto y calidad de respuestas, **parar E10 y pasar a E11/E12 usando RAG como base**. Sería avanzar por la hoja de ruta, no fracasar como proyecto.

**No se ha modificado, fusionado ni desplegado el código de producción.** El histórico E10-B negativo no se reinterpreta como positivo.
