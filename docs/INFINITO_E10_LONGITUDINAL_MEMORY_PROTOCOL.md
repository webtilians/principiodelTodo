# INFINITO E10 — Protocolo preregistrado de utilidad longitudinal de memoria

**30-10-2026. Primera prueba tras la revisión estratégica F1–F9-S.**
**Objetivo:** verificar si el núcleo REAL de INFINITO 3.0 convierte hechos previamente recibidos en **decisiones correctas** después de muchos distractores, cambios de preferencia y reinicio del historial de conversación. No entrenar nuevas redes ni adaptar F9. No retocar el benchmark histórico de 40 casos.

## Hipótesis falsable

Frente a controles con el mismo actor/resolvedor, instrucciones y presupuesto de contexto, INFINITO 3.0 con su **CognitiveEngine, RuleBasedMemoryGate, SQLiteCognitiveMemoryStore, GeneralizedContextBuilder y CognitiveLoop originales** debería tomar más decisiones longitudinales correctas que el mejor baseline práctico de recuperación.

**El primer E10 es un banco offline de *disponibilidad y utilización de evidencia* con un adaptador de decisión DETERMINISTA común a las cuatro variantes. NO se ejecuta un LLM real ni se afirma que pruebe calidad del lenguaje/razonamiento semántico.** La ventaja de este diseño es eliminar varianza del modelo para aislar almacenamiento, recuperación y contexto. El siguiente nivel de E10 debería usar el mismo protocolo con un LLM real sin cambios de casos, solo cuando exista acceso autorizado y presupuesto; no confundir resultados offline con calidad integral.

## Cuatro brazos con el mismo motor de decisión

1. \`no_memory\`: recibe el mismo prompt y no tiene almacén a largo plazo; control negativo.
2. \`fifo_8\`: conserva las últimas **8** observaciones de entrenamiento recibidas (memoria corta explícita), sin selección semántica.
3. \`rag_hash\`: memoria de todas las observaciones + búsqueda híbrida léxica/vectorial con **embeddings hash deterministas**. Es un baseline RAG **offline NO semántico profundo**; no se le atribuirá equivalencia a RAG con embeddings semánticos reales.
4. \`infinito3\`: núcleo congelado actual, persistente en SQLite, regla/gate y GeneralizedContextBuilder reales, sin modificar pesos/reglas del código productivo.

Todos los brazos comparten la **misma instancia conceptual de decodificador de decisiones**, que solo examina los mensajes finales transmitidos al modelo (historial reciente y contexto recuperado) y el *enunciado actual* con la regla de acción. **Nunca recibe la etiqueta esperada**, semilla, futuro, ni contenido almacenado que no aparezca en la petición final. Responde a una de dos acciones o \`ABSTENER\` si no hay pista. Es un *lector controlado*, no un LLM ni un oráculo. Los controles tienen el mismo límite de 8 recuerdos recuperados y **180 tokens estimados para contexto**; ningún estado de la conversación anterior se deja visible en las decisiones: se llama \`reset_history\` antes de cada evaluación, igual para todos.

\`rag_hash\` y FIFO guardan todos los mensajes de setup de manera idéntica; INFINITO aplica su gate real y exclusión/supersesión de hechos. No son idénticos internamente por definición, porque **E10 compara las arquitecturas de extremo a extremo**. Para aislar después gate vs retrieval vs context builder, se abrirán ablaciones solo si E10 las justifica.

## Secuencia longitudinal por mundo

Se generan dos historias contrafactuales iguales salvo el valor de un **hecho estructurado vigente**, con mismo número, tipo, secuencia y posiciones de distractores. El hecho usa tipos que el núcleo admite **antes** de E10 (\`mi color favorito\`, \`mi bici\`, \`vivo en\`). Cada historia:
- primera observación de un hecho A o B;
- **12–18 distractores** irrelevantes y de temas próximos, idénticos entre las dos historias;
- **primera decisión**: prompt idéntico en ambas historias, que especifica las dos acciones contrafactuales, sin revelar cuál hecho ocurrió. Su respuesta debe depender del hecho recordado;
- actualización explícita del mismo hecho a la alternativa B/A;
- otros **12–18 distractores**, idénticos entre historias;
- **segunda decisión** después de reiniciar el historial reciente de conversación. La política debe usar el valor actualizado, no el antiguo.

El hecho inicial y el actualizado llevan al resultado contrario, así que un agente que elige un lado fijo no puede superar el criterio de **cuatro decisiones correctas por par** (dos cues × dos momentos). No se inyectan resultados anteriores ni etiquetas en el estado del actor. Las acciones no modifican el mundo físico: es un **simulador de decisiones basadas en preferencias**, no control end-to-end de un mundo físico.

Se añaden **controles negativos sin hecho**: deben responder \`ABSTENER\`, no inventar preferencias a partir de las alternativas presentes en el prompt. Su resultado se reporta por separado para evitar elevar éxito mediante controles fáciles.

## Particiones y protección frente al sobreajuste

- **DEV**: \`seed = 1200000 + index*1000 + j\`, índices \`0..2\`, \`j=0..11\` → **36 pares**; solo para comprobar integridad, fijar código y descubrir fallos. Se reportarán resultados, NO se tunearán reglas de INFINITO para ganar el benchmark.
- **CONFIRM**: \`seed = 1300000 + index*1000 + j\`, índices \`0..7\`, \`j=0..15\` → **128 pares**, 256 historias ×2 decisiones = **512 acciones por brazo**, 128 unidades primarias emparejadas (no 512 muestras independientes). **Sin abrir hasta que DEV y tests pasen**. Semillas nuevas de E10, completamente separadas de las de F9; el holdout histórico F9 \`900000+\` permanece cerrado.
- **8 índices confirmatorios** de generación para resumir diferencias por unidad independiente de aleatorización; **no equivale a ocho semillas de entrenamiento**, ya que no se entrena un actor estocástico.
- Los casos se generan por código estable, con variación de hecho/distractores; la familia de plantillas permanece igual para DEV/confirmación. **Por ello E10 NO demuestra generalización fuera de la distribución diseñada; eso corresponde a E13.**
- Tras abrir confirmación no cambiar reglas, etiquetas, dataset, controlador, parámetros ni evaluación. Un error encontrado después queda documentado y exige una partición futura nueva, no refinar y declarar nuevo hallazgo con \`1300000+\`.

## Métricas congeladas

1. **Primario: proporción de pares con LAS CUATRO decisiones correctas**, dos pistas originales/invertidas × primera decisión/decisión posterior a actualización; los \`ABSTENER\` cuentan como fallo cuando hay hecho.
2. Exactitud individual de las 512 decisiones, por momento y por arm.
3. **Sensibilidad contrafactual:** ambos cambios de acción tras invertir el hecho, sin observarlo en la pregunta.
4. Cambio explícito de preferencia: aciertos después de actualizar sin recitar el valor antiguo.
5. Controles de falso recuerdo \`ABSTENER\` sin hecho: tasa de alucinación de acción.
6. Tokens estimados de contexto, instancias de recuperación pertinentes y errores, tiempos instrumentales (diagnóstico).
7. Por índice: diferencias pareadas de \`infinito3\` respecto al **mejor baseline práctico** de \`fifo_8\` y \`rag_hash\`, seleccionado **solo por desempeño agregado del mismo conjunto en un contraste de máximos preregistrado**, no elegido retrospectivamente para formar un resultado favorable. Informar ambos contrastes igualmente.

**Criterio orientativo de avance:** INFINITO supera a **ambos** controles prácticos en al menos **10 puntos porcentuales** de pares completos, con diferencia positiva en **≥6/8 índices** y bootstrap pareado *por índices* (1000 remuestreos, semilla 1730) de diferencia vs cada baseline con límite inferior >0. No atribuir independencia a decisiones dentro del mismo mundo. El umbral no es una demostración de potencia adecuada: **esta primera confirmación es un piloto confirmatorio modesto**, y el resultado negativo/ambiguo no autoriza ajustes indefinidos. Debe reportarse el IC completo incluso si no se cumple.

## Gate y ejecución

1. Tests de que cambio de A↔B NO modifica prompt, distractores, presupuesto, ni objetivos de control; que no existe fuga de claves/etiquetas; que el decodificador solo lee contenido transmitido; que no inventa ante falta de memoria.
2. Ejecutar DEV 36 pares, conservar todos los intentos. No usarlo para optimizar el núcleo.
3. Registrar SHA de la rama y hashes del script/protocolo en outputs, verificación presupuestos, reinicio real \`CognitiveLoop.reset_history()\` y un test de cierre/reapertura **física** de SQLite sin pérdida de memoria en el brazo INFINITO.
4. Solo tras superar tests y DEV, abrir \`1300000+\` por una única ejecución, generar cuatro reportes individuales y agregado, documentar fallos y casos enteros.
5. No automatizar despliegue, no fusionar sin revisión ni tocar ramas F9.

## Interpretación

Un E10 positivo mide **memoria útil para una decisión reglada bajo contexto de prueba**, no autonomía general, aprendizaje de relevancia E11, recuperación activa E12 ni comprensión abierta por un LLM. Un E10 negativo identifica que el núcleo actual no entrega o actualiza la evidencia necesaria incluso a un lector determinista competente; antes de añadir una red nueva conviene corregir la tubería con una única ronda de diagnóstico y **nueva confirmación**, o mantener el baseline. Una mejora aparente sin margen sobre el RAG hash no justifica complejidad.
