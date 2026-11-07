# INFINITO E12-A — Decidir autónomamente cuándo consultar memoria externa

**Protocolo preregistrado antes de escribir el código y antes de abrir DEV/CONFIRM, 1 noviembre 2026.**

## Pregunta y alcance

¿Puede un LLM real identificar lagunas o desactualización de su memoria de trabajo, decidir por sí mismo `SEARCH` antes de responder y recuperar información semánticamente, con menos búsquedas que un sistema que consulta siempre y sin degradar significativamente decisiones correctas?

**E11 cerrado**: E11-B aprendió a priorizar recuerdos en régimen estacionario, pero fracasó bajo inversión de demanda y no cumplió el gate. No reciclar modelos E11 para rescatar este resultado: E12 investiga **control de recuperación** y actualización, no selección fija de escritura.

E12-A es un **controlador de búsqueda mediante una decisión explícita del mismo LLM**, no una red neuronal entrenada ni una política autónoma de objetivos. El archivo largo existe y es accesible para todos; el agente no puede examinar su contenido ni contar sus documentos antes de decidir buscar.

## Entorno y modalidades de verdad

Un episodio presenta una pregunta con 2 posibles respuestas y un **contexto de trabajo breve** distinto de un **archivo externo persistente**. Cuatro clases balanceadas:

1. **`fresh`**: la memoria de trabajo contiene la asignación vigente del asunto; no hace falta buscar. El archivo puede tener una copia histórica redundante, no contradice.
2. **`archived`**: la respuesta existe solo en el archivo semántico; memoria de trabajo contiene hechos de otros asuntos.
3. **`stale`**: la memoria de trabajo incluye una versión previa del hecho identificada como **no verificada**; el archivo contiene una actualización posterior vigente. Responder desde el dato antiguo es un error.
4. **`unknown`**: ningún archivo ni memoria de trabajo contiene la respuesta; lo correcto es `ABSTENER`. Antes de buscar no existe una señal fiable para distinguir `archived` de `unknown`; esto es un límite informacional del entorno.

El enunciado incluye el asunto y dos valores posibles, **no** la etiqueta correcta ni un índice de archivo. Todos reciben los mismos textos de trabajo y de archivo para cada mundo. Hay 15 distracciones en archivo por mundo, con frases de longitud comparable y varios asuntos parecidos. El buscador usa **embeddings reales `text-embedding-3-small`**, coseno y componente léxico, top-4, orden cronológico y un máximo de **260 tokens estimados** de evidencia, igual que el RAG de E10/E11. El contexto de trabajo NO cuenta dentro del presupuesto de archivo, pero es el mismo para todos y se audita por separado.

El LLM real `gpt-5.6-luna`, `reasoning_effort=none`, `max_output_tokens=80`, decide si sabe responder o si debe hacer `SEARCH`. Si decide SEARCH, la aplicación llama al recuperador y envía una **segunda** consulta al mismo modelo para responder; si decide responder desde memoria de trabajo, la respuesta se evalúa sin búsqueda. **NUNCA** se entrega al router el hecho objetivo de archivo, la clase del mundo ni la respuesta esperada. Las acciones válidas son `SEARCH`, `ABSTENER` o uno de dos códigos. Una salida inválida cuenta como error, no se reintenta como éxito.

## Brazos de control y costes

- **`never_search`**: una decisión del LLM usando solo memoria de trabajo.
- **`always_search`**: una sola decisión del LLM con el RAG externo y memoria de trabajo, aún cuando era innecesario.
- **`fixed_heuristic`**: mismo LLM para responder; regla explícita que consulta si el hecho solicitado no aparece o si el único hecho aparece `no_verificado`. No accede al archivo sin SEARCH.
- **`llm_router`**: el LLM elige `SEARCH` o acción final a partir de la pregunta y el trabajo; una segunda llamada al LLM solo si eligió SEARCH.

**El componente que responde** usa las mismas instrucciones, modelo y representación de evidencia para los cuatro brazos. El enrutador añade una llamada extra en los casos SEARCH: reportar **llamadas al LLM, tokens de entrada y salida, llamadas a embeddings, búsquedas y aciertos** por brazo; no declarar ahorro monetario sin medirlo. Un sistema que busca siempre puede acertar más a costa de búsquedas, pero gastar menos tokens LLM que el router: esa comparación es parte del experimento.

La heurística fija es un control fuerte, y puede empatar o ganar. No atribuir a autonomía un simple reconocimiento de campos explícitos.

## Particiones nuevas y métrica

- DEV: `5100000+j`, j=0..7: dos mundos por clase, estrictamente de desarrollo; no ajustar prompt/umbral según tasa de confirmación.
- CONFIRM: `5200000+class*1000+j`, class=0..3, j=0..7: **32 mundos nuevos, 8 por clase**. 4 brazos, 128 respuestas finales; router puede emitir llamadas adicionales. E10 `1300000+/1800000+/2200000+`, E11 `3300000+/4200000+` y F9 `900000+` no se usan.
- La **unidad de evaluación** es mundo, no llamada al modelo. Controles incluyen buscar cuando no hay respuesta, rechazar recuerdo antiguo, evitar invenciones, y no usar instrucciones ocultas en texto recordado.
- La muestra es pequeña; intervalo bootstrap por **32 mundos agrupados por clase** (10.000 remuestreos de 8 mundos dentro de cada clase; semilla 1812) es descriptivo, no prueba de generalidad.

**Primaria:** utilidad = tasa de respuesta correcta menos **0,08 por búsqueda** (`SEARCH`), con el mismo coste preregistrado por búsqueda para todos. La unidad de penalización se fija antes de ejecutar: corresponde a preferir no consultar cuando hay memoria de trabajo suficiente, pero no a costa de perder muchas respuestas. Se reporta además acierto puro y la diferencia con `always_search` y `fixed_heuristic`.

**Gate de avance E12-A:** `llm_router` debe (a) no perder más de **5 puntos porcentuales** de acierto frente a always-search; (b) reducir al menos **20% de búsquedas** frente a always-search; (c) superar a la heurística fija en utilidad al menos **5 puntos porcentuales** (evitar victoria por un control trivial); (d) no responder con un valor antiguo en más de un caso stale ni inventar en unknown en más de un caso. Mostrar los cuatro valores y sus incertidumbres; si falla uno, **no declarar éxito** ni ajustar las preguntas retrospectivamente. 

**Secundarias:** exactitud por modalidad, decisión `SEARCH` apropiada, llamadas adicionales, tokens y coste relativo, similitud/retrieval hit, abstenciones/errores, contraste del router frente a la heurística cuando hay versiones conflictivas.

## Reglas de parada

Si un router explícito no supera la heurística fuerte, o cuesta significativamente más tokens que always-search, cerrar E12-A como resultado negativo; en lugar de añadir prompts para ganar, definir E12-B **solo si una nueva hipótesis mecanicista lo justifica** y con nuevo holdout. La búsqueda no resuelve el caso de datos nunca almacenados; esta limitación informacional se declara explícitamente.

No modificar `src/infinito3/`, ni fusionar a producción. Tests de integridad antes de una única confirmación real; única activación mediante fichero de lanzamiento; publicar también cualquier fracaso.
