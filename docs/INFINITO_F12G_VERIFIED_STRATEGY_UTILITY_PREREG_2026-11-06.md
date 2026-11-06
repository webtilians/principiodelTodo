# INFINITO F12-G — ¿Mejoran las decisiones las estrategias verificadas?
**Prerregistro prospectivo — 6 noviembre 2026. Se congela ANTES de programar el experimento F12-G y consultar nuevos resultados.**

## Hipótesis y limitación crucial
F12-E halló utilidad descriptiva de memoria episódica en unas condiciones, pero una estrategia inferida por el LLM falsificó la relación entre acciones y fuentes. F12-F verificó 7/8 afirmaciones individuales y rechazó una inventada con evidencia estructurada, pero su prueba adversarial estaba parcialmente construida a partir de la misma semántica de su filtro. F12-G plantea un test distinto y causal **sobre resultados posteriores de un agente congelado**: ¿agregar afirmaciones *verificadas contra su experiencia real* supera un RAG episódico competente, y supera también agregar afirmaciones de LLM sin verificar?

No alegar inteligencia general ni transferencia a reglas nuevas: las combinaciones perfil/revisión serán conocidas en TRAIN, pero los servicios de evaluación serán nuevos y sus casos no usados para diseñar el código. Un éxito de A3 debe ser superior a A1 **y** al baseline temporal A4, no simplemente a A0.

## Actor, fuente y privacidad
- Único actor y sintetizador **Qwen2.5-7B-Instruct GGUF Q4_K_M** congelado (repo `bartowski/Qwen2.5-7B-Instruct-GGUF`, fichero `Qwen2.5-7B-Instruct-Q4_K_M.gguf`, revisión HF `8911e8a47f92bac19d6f5c64a2e2095bd2f7d031`, SHA256 pesos `65b8fcd92af6b4fefa935c625d1ac27ea29dcb6ee14589c55a8f115ceaaa1423`); `llama-cpp-python==0.3.36`, CPU, temperatura 0, contexto n_ctx=4096, cuatro hilos. Ningún otro modelo, fine-tuning o herramienta externa.
- Memoria REAL de INFINITO, `SQLiteCognitiveMemoryStore(path=":memory:")` con `HashEmbeddingProvider`, solo episodios sintéticos, sin memoria personal/MCP ni escritura al producto.
- El verificador de F12-F recibe exclusivamente episodios públicos con `source_id, profile, revision, events` (acciones ejecutadas, feedback, `verify` posterior). NO se le pasa mapa de compatibilidad, acciones que el agente no realizó ni resultados contrafactuales.

## Entorno y secuencias sin políticas expertas
- Nueva familia **de despliegues temporales con operadores opacos** `apply_delta`/`apply_sigma`, `inspect`, `verify`. Cuatro perfiles nuevos `quartz, reed, cobalt, dusk`, revisiones públicas R1 y R2; en el evaluador la opción compatible puede cambiar entre versiones. Este mapa nunca entra en el actor, generador de afirmaciones, recuperación ni validación; solo se observa aceptación/rechazo de herramienta. Se mantienen la semántica y API pública F12-F para evitar cambios oportunistas.
- **TRAIN**: 8 experiencias nuevas `W0..W7`, primero todas R1, después R2, una para cada perfil/revisión. El mismo LLM decide hasta 5 acciones **sin memoria**, observa resultados, y almacena en SQLite todos los eventos públicos, incluidas derrotas. No se le muestran soluciones correctas.
- El mismo LLM genera una afirmación JSON individual por episodio `source_id,profile,revision,operation,observed_outcome,verified_after_operation`, sin corrección manual de JSON. El filtro C2 de F12-F valida únicamente su soporte observacional; solo afirmaciones **aceptadas y verificadas** son candidatas a estrategia. Conservar las candidatas falsas como evidencia negativa y no descartarlas retrospectivamente del brazo A2. Los candidatos extraídos se congelan **ANTES** de evaluar.
- **TRAIN-heldout**: 8 servicios nuevos `Y0..Y7`, combinaciones perfil/revisión **vistas** en TRAIN, cuatro R1 y cuatro R2, orden preregistrado intercalado, distintas descripciones. Se verifican dos horizontes **tight=2** y **slack=3** acciones por brazo/caso. No usar S/H de F12-E ni V de F12-F, ni particiones F11 DEV/CONFIRM.

## Brazos y recuperación, igualdad de recursos
El mismo modelo, contexto sistema, catálogo, feedback público y horizonte para todos; referencias externas a prompt **<=1200 caracteres**, con SHA256 y **texto exacto renderizado** en artefactos para auditoría. Orden de brazos rotado determinísticamente por caso y horizonte. Cinco brazos:
- **A0**: sin memoria.
- **A1**: control fuerte, memoria **EPISODIC** desde `SQLiteCognitiveMemoryStore.search` y filtro exacto de perfil público, incluyendo ambas revisiones (R2 antes de R1, sin descartar históricas), formato compacto de las acciones/feedback originales.
- **A2**: A1 + afirmaciones individuales estructuradas del LLM **sin verificar**, incluso falsedades, con el mismo límite total de contexto. Distinguir formato inválido y abstención; nunca inventar una regla para rellenar.
- **A3**: A1 + las mismas afirmaciones **filtradas por verificador C2**, solo aceptaciones realmente verificadas de fuentes que coinciden en perfil/revisión; todas se etiquetan hipótesis con evidencia y no reglas del simulador. El validador no mejora el actor directamente por otros medios.
- **A4**: control heurístico de **recencia**, recupera solo el episodio R2 más reciente del mismo perfil y no ve la revisión vieja (misma cuota máxima de contexto). Este baseline imperfecto se registra para detectar la falsa ventaja de una política cuyo único efecto sea priorizar informes nuevos.

## Decisiones y preregistro de gates
- **P0 integridad**: modelo real idéntico al F12-F, 8 episodios train, 8 heldout x dos horizontes x cinco brazos=**80 evaluaciones**; memorias congeladas antes del primer heldout; texto exacto de A1/A2/A3/A4 visible; cero mapa privado en instrucciones/recuerdos. Si falla, INVALID o BLOCKED.
- **P1 materia prima**: al menos cuatro experiencias terminan con `verify` positivo y se obtienen >=2 afirmaciones verificadas por C2; de lo contrario **INCONCLUSIVE** para política.
- **P2 utilidad adicional**: A3 supera a A1 en **>=2 de 8 éxitos tight**, y no es inferior a A1 en slack ni en ninguna revisión R1/R2. Si no, NO-GO de beneficio de verificación. Reportar acciones/retornos, tokens y latencia, no solo éxitos.
- **P3 causalidad del filtro**: A3 supera a A2 **>=2/8 tight**, **o** no pierde éxitos agregados y evita explícitamente que el actor adopte afirmaciones falsas (medir cambios de acción y consecuencias). Si A2 y A3 son idénticos porque no hubo falsedades, **INCONCLUSIVE** para efecto causal de verificar.
- **P4 comparadores**: A3 debe no ser inferior a A4 en éxitos agregados y registrar los estratos R1/R2; no concluir superioridad si la diferencia es trivial o baseline se rompe por diseño.
- **P5 coste**: A3 no debe costar >25% más tokens de entrada que A1 en evaluación para reclamar una ventaja eficiente; medir aparte gasto de TRAIN, síntesis, verificación, bytes almacenados y tiempo de inferencia. Un beneficio de éxitos con coste mayor sigue siendo evidencia descriptiva, no PASS de eficiencia.

**GO prospectivo para nuevas réplicas y confirmación solo si P0–P5 y la auditoría de falsedades pasan**; este es **un piloto diagnóstico de una réplica y banco sencillo**, no evidencia confirmatoria. Un resultado negativo o saturado debe documentarse íntegramente. No reoptimizar con Y0..Y7 después de mirarlos.

## Riesgos
- A3 selecciona afirmaciones mediante comprobaciones deterministas; no aprende el verificador, no modifica pesos del LLM y NO demuestra por sí mismo creación de políticas generalizables.
- A0 puede sufrir una debilidad de planificación sin memoria, lo que no prueba la necesidad universal de INFINITO.
- Un mismo perfil/revisión en TRAIN y heldout favorece recuperación de tabla; **no es OOD de reglas**. La comparación A3 vs A1/A2/A4 es crucial.
- El modelo puede no generar suficientes afirmaciones falsas para aislar causalmente el filtrado, o las sintetizadas pueden ocupar contexto que empeore A1; registrar como INCONCLUSIVE/NO-GO sin reintentos adaptativos.
- Coste de unas 80 ejecuciones, CPU y cuotas del runner; si falla por timeout antes de resultados completos marcar BLOCKED, no un rendimiento del modelo.

**Estado al guardar este archivo: SOLO DISEÑO, sin implementación F12-G ni datos nuevos.**
