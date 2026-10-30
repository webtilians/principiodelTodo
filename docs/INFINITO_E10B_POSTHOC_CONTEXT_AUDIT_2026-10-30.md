# INFINITO E10-B — Auditoría exploratoria posterior de disponibilidad de evidencia

**30-10-2026. EXPLORATORIO, NO PREREGISTRADO.** No altera resultados, métricas ni umbrales de E10-B; **no autoriza** volver a confirmar sobre el mismo conjunto \`1800000+\`.

## Procedencia y método

Origen: artifact original [E10-B #41109251106 / 13002599280](https://github.com/webtilians/principiodelTodo/actions/runs/41109251106/artifacts/13002599280), fichero \`report.json\`, **24 mundos confirmatorios pareados ×2 variantes de pista ×2 momentos =96 decisiones por brazo**. Se inspeccionó literalmente el \`context\` que recibió cada LLM real, separado de la pregunta actual (\`question\`).

Como las alternativas son enumeraciones acotadas y sintéticas, se estableció, **ex post**, si la cadena del valor vigente real aparecía en el contexto, por coincidencia de texto **insensible a mayúsculas**. El valor correcto nunca forma parte de la propia pregunta; esa pregunta enumera las dos opciones pero no su historial. Una coincidencia de cadena **no garantiza** un recuerdo válido o una inferencia correcta (el contexto también puede contener el valor anterior). Se comparó el \`action\` real, el \`raw\` y la puntuación congelada \`expected_first/second\`.

**No se ha instrumentado el almacén pre-búsqueda, el ranking antes del constructor, ni el gate de manera causal adicional.** La auditoría localiza dónde **deja de verse** la evidencia, pero no demuestra por sí sola exactamente cuál módulo la eliminó.

## Descomposición exhaustiva de las 96 decisiones por brazo

| Brazo | Valor correcto NO presente en contexto enviado al LLM | Presente y acertó | Presente pero ABSTENER | Otros errores |
|---|---:|---:|---:|---:|
| Sin memoria | 96 | 0 | 0 | 0 *(96 abstenciones sin evidencia)* |
| RAG semántico cronológico | **0** | **96** | **0** | **0** |
| INFINITO 3.0 original | **64** | **25** | **7** | **0** |
| INFINITO gate abierto | **16** | **72** | **8** | **0** |

\`sin memoria\` se omite de comparaciones mecanicistas por no aspirar a recibir evidencia. El total de abstenciones sin evidencia en INFINITO es **64**, y en la ablación \`gate_open\` es **16**. Todas las decisiones que fallaron en cualquiera de los brazos fueron **ABSTENER**: ningún actor seleccionó un código de alternativa incorrecta, lo que es consistente con instrucciones de prudencia pero no sustituye decisiones correctas cuando el dato existe.

En el RAG semántico, el **valor vigente estaba visible en las 96 decisiones**. En las 48 decisiones **posteriores a la actualización** el contexto incluía además el valor antiguo y el LLM seleccionó el nuevo correctamente. Por tanto, presentar las dos versiones no impidió resolver la cronología en este pequeño banco.

### Por qué la ablación ayuda a localizar el problema

**INFINITO original:** las **64 decisiones de las cuatro familias nuevas** (avisos, formato, escalado, cadencia, 4 mundos/familia ×4 decisiones/mundo) carecían de la evidencia histórica vigente en el contexto. En las 32 de color/ciudad el dato correcto sí estaba visible, pero hubo **7 abstenciones**. Coincide con que el gate actual fue escrito para reconocer tipos concretos de hechos, aunque este conteo solo evidencia **ausencia de contexto**, no una medición individual de cuántas escrituras fueron rechazadas.

**INFINITO gate abierto:** la evidencia vigente pasó a estar disponible en **80/96** decisiones; se acertaron **72/80** y hubo **8 abstenciones con evidencia visible**. Las **16 ausencias** pertenecen exclusivamente a **cadencia**, pese a que su ingestión usa un gate de almacenar declaraciones. Debe analizarse, en una única prueba diagnóstica, si la pérdida se produce en \`search\` o en \`GeneralizedContextBuilder._apply_precision_filter/_prune_memory_pool\`. **No asumir** sin trazas que sea exclusivamente un problema de retrieval ni modificar ya el conjunto confirmatorio.

**Color y ciudad:** incluso con un hecho semánticamente claro y renderizado literalmente en \`[USER MODEL]\`, el LLM real devolvió \`ABSTENER\` en algunos casos. Es un fenómeno del **camino contexto→acción** que puede depender de las instrucciones y presentación del contexto, además del LLM. No atribuirlo inequívocamente al modelo sin comparación controlada con el mismo contenido.

## Conclusión de esta auditoría

El fracaso de E10-B no es un único «fallo de memoria»:
1. **Cuello de botella A — disponibilidad de evidencia antes de actuar:** 64/96 decisiones sin el valor actual en INFINITO original; al abrir gate baja a 16/96. En tareas de la vida real esto puede hacer que el agente **se abstenga pese a disponer de la información en la conversación original**.
2. **Cuello de botella B — lectura/uso aun cuando la evidencia está presente:** 7/32 fallos sobre evidencia disponible en original; 8/80 en gate abierto. Se observó abstención, **no selección errónea**.
3. **Ventaja concreta del RAG:** recuerda todas las declaraciones, selecciona semánticamente y **preserva orden cronológico** de hechos que se reemplazan. Usó aproximadamente 145 tokens estimados de evidencia por decisión vs 36 original y 56 gate abierto, bajo el **mismo máximo de 320**.

Este análisis es **post-hoc descriptivo**, útil para proponer una sola auditoría de trazas de captura/búsqueda/contextualización en **datos de desarrollo o fixtures nuevos**, no para proclamar un nuevo hallazgo confirmado o reajustar INFINITO en \`1800000+\`.

La hipótesis E10-B **permanece negativa**: INFINITO original 2/24 pares de cuatro decisiones, RAG semántico 24/24. Se conserva sin fusionar el núcleo de INFINITO 3.0 y no se abren líneas F9 paralelas.
