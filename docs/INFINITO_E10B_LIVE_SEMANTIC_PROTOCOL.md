# INFINITO E10-B — Protocolo prospectivo de decisiones longitudinales con LLM REAL y RAG semántico

**Preregistro: 30-10-2026; guardar en Git ANTES del desarrollo y de la primera llamada de confirmación.**

## Propósito y límites

Tras E10-A (lector offline determinista, tres predicados estructurados, RAG vector hash), investigar si el núcleo **original** INFINITO 3.0 ofrece utilidad real ante un **modelo de lenguaje OpenAI** y frente a un **baseline RAG semántico fuerte, temporalmente ordenado**, con escenarios abiertos menos coincidentes con el extractor de hechos. No modificar \`src/infinito3/\`, sus reglas ni el modelo LLM para aumentar resultados. Este E10-B tampoco constituye prueba de memoria autónoma aprendida, ni autoriza saltar E11/E12.

## Modelo / embeddings / presupuesto fijo

- LLM real **\`gpt-5.6-luna\`**, \`reasoning_effort=none\`, temperatura por defecto del proveedor, \`max_output_tokens=80\`, mismo modelo, mismo contrato normalizado \`CognitiveLoop\` y mismas instrucciones de salida para todos los brazos. No cambiar a otro modelo tras observar confirmación.
- Embeddings **reales \`text-embedding-3-small\`** para \`infinito3\`, \`infinito3_gate_open\` y \`semantic_rag\`; proveedor y caché compartida de textos idénticos, no vectores hash. La caché no usa etiquetas ni resultados.
- Contexto \`max_tokens=320\` **estimados por el mismo estimador existente** y \`top_k=8\` recuerdos. Historial corto **0** en todos los brazos durante las decisiones. Esta igualdad del presupuesto de evidencia no garantiza igualdad de tokens totales API: registrar los tokens reales.
- Ingestión de información de configuración mediante \`CognitiveEngine.process\`, exactamente la capa usada por el \`CognitiveLoop\` antes de generar respuestas; **no generar LLM en cada setup** porque supondría gasto repetido, y ninguna arquitectura usa las respuestas de setup como memorias. En decisión: \`CognitiveLoop.turn\` invoca el **LLM real**, no el oráculo del E10-A.
- Si falta \`OPENAI_API_KEY\`, falla la comprobación de credenciales o no puede realizarse una llamada real: **abortar**; nunca sustituir silenciosamente por offline ni fingir confirmación. El pipeline de CI separa la barrera de configuración + pruebas, DEV y confirmación.

## Brazos, manteniendo actor/embedding iguales

1. **\`no_memory\`**: mismo LLM, prompt de decisión e instrucciones, pero no recibe recuerdos; control de adivinación/filtración.
2. **\`semantic_rag\`**: guarda **todos** los textos de setup sin etiquetarlos como respuestas ni usar los hechos futuros. Recupera con el mismo modelo de embeddings semánticos, mezcla similaridad y pequeño solapamiento léxico, máximo 8 registros, y **entrega el material seleccionado en orden temporal** con índice de turno para que el LLM pueda resolver actualizaciones. Tiene presupuesto común de 320 tokens y la misma instrucción general de que los acuerdos posteriores sustituyen los anteriores. No es un strawman hash/FIFO.
3. **\`infinito3\`**: \`CognitiveEngine.persistent\` sin cambios, \`RuleBasedMemoryGate\`, \`SQLiteCognitiveMemoryStore\`, \`GeneralizedContextBuilder\` verdaderos, con el mismo embedding semántico y LLM real.
4. **\`infinito3_gate_open\`**: **diagnóstico secundario preregistrado**, mismo SQLite, retrieval y Context Builder del núcleo, solo sustituye el gate por uno que persiste todos los textos declarativos no sensibles (no los prompts de decisión). No tiene privilegios especiales de recuperación; permite distinguir fallo de captura (gate) frente a fallo de búsqueda/construcción de contexto.

No hay reentrenamiento neuronal, ajuste del LLM, oráculo de respuestas ni extracción con etiquetas ocultas. Los brazos reciben **los mismos hechos, distractores, actualizaciones, pregunta y reglas de acción**. Cada brazo mantiene un almacén aislado, nunca cruza información de otras historias.

## Seis familias, condiciones contrafactuales, dos decisiones

**Cuatro familias nuevas no contempladas en los predicados programados del núcleo:**
- canal de avisos para un proyecto;
- formato de entrega de un expediente;
- equipo de escalado de incidencias;
- frecuencia de revisión de un plan.

**Dos familias de control con predicados estructurados existentes:** color favorito y ciudad actual, para separar fallos de generalización de regresiones básicas.

El generador elige aleatoriamente nombres de proyectos/expedientes y dos valores posibles, y produce una **pareja contrafactual de historias** que difieren solo en la elección vigente. Misma geometría textual (número y orden de turnos), mismo proyecto, mismos distractores, mismas alternativas y opciones de acción. Cada historia:
1. una frase inicial factual, natural, p. ej. «Para el proyecto X hemos acordado recibir los avisos por correo»;
2. 12 distractores escritos en lenguaje natural, varios sobre otros proyectos o sobre temas próximos;
3. **primera decisión**, con dos opciones explícitas y respuesta obligatoria de una sola palabra;
4. una **actualización que sustituye la elección anterior** por la opción contraria;
5. 12 distractores adicionales;
6. cierre y reapertura de SQLite de los dos brazos persistentes, \`reset_history()\` de todos;
7. **segunda decisión**, idéntico prompt y mismas opciones.

En cada mundo se evalúan la historia original e invertida; la decisión debe seguir la preferencia real **antes y después del cambio**. El enunciado nombra las dos alternativas, pero **no indica cuál se acordó**. Dada la historia completa, el resultado debe cambiar al invertir el hecho y después de actualizarlo. Las acciones no modifican el entorno: son decisiones de clasificación dependientes de recuerdos, no navegación autónoma.

## División nueva e independiente

- **DEV**: \`1700000+index*1000+j\`, \`index=0\`, \`j=0..5\` → **6 pares**, uno de cada familia; pueden probarse tests/credenciales/instrumentación solo aquí. No ajustar el núcleo tras DEV. Si se detectan errores instrumentales, corregir y reiniciar DEV **antes** de abrir confirmación.
- **CONFIRM**: \`1800000+index*1000+j\`, \`index=0..3\`, \`j=0..5\` → **24 pares**, 6 familias ×4 índices; 2 historias ×2 decisiones ×4 brazos = **384 llamadas LLM** máximas para la confirmación. Se añade un control negativo de ausencia de evidencia por familia y brazo (hasta 24 llamadas LLM) **con hechos inexistentes**; ningún actor debe inferir algo no recibido. Todas las acciones fallidas, errores y abstenciones permanecen en denominador.
- E10-A \`1200000/1300000+\` y F9 \`600000..900000+\` completamente separados; el holdout histórico F9 \`900000+\` NO SE ABRE. Después de la primera confirmación, **\`1800000+\` queda consumido** y no se reutiliza para seleccionar variantes.
- Las cuatro cohortes no son cuatro semillas de entrenamiento de LLM; las tareas son plantillas compartidas. No inferir robustez general fuera de las seis familias.

## Métricas preregistradas

**Primaria:** proporción de **pares completos con cuatro decisiones correctas** (dos historias contrafactuales × antes/después de actualización), denominador 24. Una respuesta extraña, \`ABSTENER\` con evidencia suficiente, error de API, o texto que no coincida con la acción esperada, cuentan como fallo, no se excluyen.

Secundarias:
- decisiones individuales correctas sobre **96 por brazo**, correcta antes/tras actualización, y por familia (4 pares).
- inversión contrafactual correcta en ambos momentos, contra elecciones constantes.
- falsos recuerdos en los **seis controles negativos** por brazo: deben contestar \`ABSTENER\`; reportar tasas por separado.
- evidencia de recuperación efectivamente transmitida en el último LLMRequest, coste de tokens del proveedor, límites de contexto y tipos de errores.
- contraste diferencial \`infinito3_gate_open - infinito3\` para diagnosticar captura versus recuperación. No es la comparación primaria.
- diferencia emparejada \`infinito3 - semantic_rag\` por **las seis familias** y por los cuatro índices; intervalo bootstrap exploratorio por familia, semilla fija 1729 (2000 remuestreos), sin asumir independencia de las 96 acciones.

**Criterio de progreso de E10-B:** INFINITO debe mejorar en **≥10 puntos porcentuales** los 24 pares completos respecto al \`semantic_rag\`, diferencia positiva en ≥4/6 familias y límite inferior del intervalo bootstrap familiar >0; además tasa de alucinación negativa no mayor que la del RAG. Esto es una regla pragmática de parada y NO una prueba estadística de alta potencia (n=24); debe informarse también cualquier equivalencia, empate o peor resultado.

**Condición de parada:** si no supera a RAG semántico, **no proclamar éxito de E10-B**. No tunear sobre \`1800000+\`. Una sola ronda diagnóstica futura, con confirmación genuinamente nueva, únicamente si emerge un bloqueo concreto de relevancia para E11/E12; de lo contrario conservar baseline/continuar la hoja de ruta.

## CI y trazabilidad

- Workflow de **integridad sin llamadas LLM** al hacer push de código. Pruebas sobre DEV generados sin etiquetas filtradas: el historial opuesto conserva distractores/prompt; los valores inicial/actualizado se invierten; ningún contexto contiene etiquetas de puntuación ni secretos; igualdad de presupuesto.
- Workflow **live activado una sola vez** por commit del fichero \`docs/INFINITO_E10B_LAUNCH.md\` escrito tras pasar los tests; exige \`OPENAI_API_KEY\`, ejecuta preflight y DEV, verifica integridad, y solo entonces abre CONFIRM. No ejecutar automáticamente en cada edición.
- En fallo de credenciales, modelo o embeddings: abortar antes de confirmación. En fallo tras abrir confirmación: partición consumida y ejecución reparada es exploratoria, no nueva validación.
- Los resultados detallados de cada par se conservan como artefactos GitHub Actions, junto a hashes del protocolo/script, SHA del commit, modelo, embeddings y todos los costes; informes agregados se versionan después de terminar.
- No modificar, fusionar ni desplegar INFINITO 3.0 en este experimento.

## Advertencia explícita

Este es el **primer ensayo E10 con un modelo real**, no un sistema autoentrenado. Dos de seis familias sí están alineadas con el extractor; cuatro se diseñaron para romper esa dependencia. El baseline semántico con cronología no tiene extracción contradictoria de hechos estructurados mediante LLM; por tanto todavía no representa el máximo estado del arte. Si INFINITO gana, esa limitación debe reconocerse; si no gana, el resultado negativo es informativo y suficiente para justificar no añadir complejidad sin una hipótesis concreta.
