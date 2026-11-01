# INFINITO E10-D — UNA sola intervención de captura general y colisión «plan»

**Preregistro: 1 noviembre 2026, ANTES de escribir el código de la intervención, ejecutar DEV o abrir la confirmación.**
**Relación:** cierre de la línea E10 tras E10-A (offline positivo), E10-B (LLM real negativo) y E10-C (diagnóstico de etapas). No modifica resultados ni reutiliza los casos confirmatorios consumidos.

## Hipótesis decisiva

La combinación de **una única política general de captura de declaraciones** y **una corrección de intención de goals multilingüe** puede recuperar decisiones longitudinales sin añadir listas de campos, nombres de dominios o reglas específicas del benchmark. Se compara contra un **RAG semántico cronológico competente**, que sigue siendo la referencia principal. El núcleo original permanece congelado y se implementa la intervención en **clases experimentales inyectables**, sin fusionar a producción por ganar unos ejemplos.

### Modificaciones preautorizadas — exactamente dos, en un solo candidato congelado

1. **Gate declarativo genérico**: si el \`RuleBasedMemoryGate\` original decide guardar, conservar su clasificación y confianza. De lo contrario, guardar **toda frase declarativa segura de >= 5 tokens** como \`EPISODIC\`, importancia fija \`0.60\`, sin acceso al futuro, etiquetas, embeddings ni nombres de campos. No almacenar preguntas (\`?\`/ \`¿\`), mensajes triviales ni textos con prohibición de seguridad; la seguridad de \`CognitiveEngine\` sigue vigente. Este gate es un **baseline genérico de alta recuperación y potencial sobrealmacenamiento**, NO un aprendizaje E11. Evaluar coste de recuerdos y distractores.
2. **Intent de goals**: modificar **solo la detección de «plan» por la regex inglesa \`plans?\`**, que malinterpreta español \`plan\` de «frecuencia del plan». La nueva detección conserva \`plans\` plural y \`my/our/the/current/pending/upcoming/future plan\` en inglés, pero **no** \`plan\` sin un contexto posesivo inglés. Conservar el resto de señales \`_asks_for_goals\`. Sin añadir excepciones para «cadencia» o dominios.

No se autoriza afinar el top-k, presupuesto, embedding, duración, selector de ContextBuilder, regex de hechos, política de supersesión, texto de instrucciones ni LLM tras observar datos de confirmación. **No introducir una tercera modificación** para rescatar el resultado.

## Contrafactuales y controles

Cinco brazos con idéntico **LLM real \`gpt-5.6-luna\`**, reasoning \`none\`, \`max_output_tokens=80\`; embeddings **\`text-embedding-3-small\`** para todos los almacenamientos semánticos (mismo cache por texto); \`top_k=8\`, máximo **320 tokens estimados** de referencia y \`history_limit=0\`:

- \`no_memory\` control de abstención.
- \`semantic_rag\` **primario**: guarda todas las declaraciones en crudo, recupera semánticamente y ordena históricamente; ni extrae hechos ni consulta oráculo.
- \`infinito3\` versión ORIGINAL congelada, control de regresión de producción.
- \`infinito3_gate_open\` ablación de E10-B (guarda todas las declaraciones, ContextBuilder original sin corregir).
- \`e10d_intervention\` **único candidato** descrito arriba (gate declarativo general + corrección intencional); nunca modifica core original.

Se conservarán todos los brazos, resultados negativos, tiempos/tokens, declaraciones almacenadas, dimensiones de recuperación y contexto; las decisiones se puntúan sin juez LLM adicional ni filtrado de respuestas por contenido.

## Nuevos mundos, pruebas fuera de las plantillas antiguas

**Seis familias nuevas**, con nombres y frases distintos a E10-B:
1. \`owner\`: responsable de aprobación de un documento;
2. \`backup\`: destino operativo de copias de seguridad;
3. \`audience\`: grupo destinatario de un resumen;
4. \`retention\`: plazo acordado de conservación de archivos;
5. \`favorite_tool\`: herramienta preferida para una tarea;
6. \`schedule_plan\`: horario acordado de revisión de un plan (control deliberado del falso positivo \`plan\`).

Ninguna familia utiliza la literalidad de los patrones «mi bici es», «mi color favorito es» o «vivo en» que favorecían E10-A. Las opciones y códigos de salida son distintas, generadas por seed. Cada mundo tiene **2 historias de hechos invertidos**, mismo prompt/opciones/orden de **12 distractores** antes de decidir; después una **actualización contradictoria** al valor contrario y **12 distractores**, cierra/reabre SQLite y decide otra vez. Éxito = **4 decisiones correctas** del par. Los distractores permanecen constantes entre historias.

- **DEV exclusivamente**: \`2100000+j\`, \`j=0..5\`, un mundo por familia, para test de integridad/preflight del SDK. No retocar las dos modificaciones tras inspeccionar su puntuación; solo arreglar errores verificables de instrumentación antes de abrir confirmación, registrándolos.
- **CONFIRM virgen**: \`2200000+i*1000+j\`, \`i=0..3\`, \`j=0..5\`: **24 pares**, 96 decisiones por brazo, **480 peticiones de decisión** + hasta 6 controles negativos por brazo (=30 llamadas). No hay repetición automática ni reintentos de toda la cohorte para seleccionar una buena ejecución. Los errores de API durante confirmación se registran y dejan la cohorte consumida.
- \`1300000+\` E10-A, \`1800000+\` E10-B confirmatorios consumidos; \`1900000+\` E10-C solo diagnóstico. No reutilizar para evaluación científica. Holdout de F9 intacto.

## Indicadores y decisión de parada

**Primaria:** porcentaje de **24 pares** con todas las cuatro acciones correctas, con mismo denominador en todos los brazos. Comparación \`e10d_intervention - semantic_rag\`. **La intervención no pasa por empatar si añade complejidad**.

**Puerta de avance preregistrada (conservadora y pragmática, poco potente por n=24):** ganar ≥10 puntos porcentuales absolutos frente a RAG semántico, mejora en ≥4 de las 6 familias y límite inferior del bootstrap por familias (2000 muestras, semilla 1781) >0, además de no empeorar los controles negativos de «ABSTENER». Reportar todos los resultados, aun si todos fallan.

**Secundarias importantes:** (a) \`e10d_intervention - infinito3\` y \`-gate_open\`; (b) aciertos individuales 96/arm; (c) hechos en memoria, ruido almacenado y tokens de contexto; (d) errores por familia antes/después y pair flip; (e) 6 controles negativos sin dato; (f) 75 tests regresión v3 y 6+7+4 tests E10-A/B/C.

**Sin éxito práctico adicional:** incluso si la intervención supera al original pero empata/perde frente a RAG, **no continuar diseñando más parches E10**. Retener RAG cronológico como baseline y pasar a E11 (aprender relevancia) o E12 (recuperación autónoma), sin afirmar que hemos obtenido una nueva arquitectura superior. **No promover experimental a producción** sin un beneficio medido que justifique sus costes y una revisión de riesgos de guardar demasiado.

## Límite de interpretación

Una mejora demostrada aquí es utilidad de recuperación contextual ante decisiones cerradas con un LLM real, no inteligencia general, razonamiento autónomo ni aprendizaje de qué conviene recordar. El gate declarativo almacena más y puede implicar privacidad/coste si se generaliza. No llamarlo «aprendido».

## CI y auditoría

Primero tests de reglas E10-D, regresiones completas y DEV real con OpenAI API; verificación de modelo/token/embedding/SQLite. Solo entonces permitir una única confirmación prospectiva en la misma ejecución CI. La ejecución **live solo se activa mediante fichero de lanzamiento explícito**, separado del workflow de tests offline. Rechazar sin secreto API, no usar mocks como datos científicos. PR borrador apilada sobre E10-C, producción inalterada, sin merge automático.
