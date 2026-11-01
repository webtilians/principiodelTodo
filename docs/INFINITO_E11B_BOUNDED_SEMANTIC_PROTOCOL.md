# INFINITO E11-B — Selección aprendida con RAG semántico de capacidad limitada y LLM real

**Protocolo preregistrado el 1 noviembre 2026, ANTES de implementar y abrir DEV o CONFIRM.**
Se apoya en E11-A: cinco modelos log-lineales entrenados con feedback diferido, congelados y confirmados en un entorno sintético con claves desconocidas. **No reentrenar ni afinar sus pesos a partir de E11-B.** Se compara la política de escritura, no una arquitectura completa INFINITO 3.0 ni una memoria semántica aprendida de extremo a extremo.

## Pregunta y decisión de parada

¿Mejora la elección aprendida de QUÉ cuatro episodios conservar las decisiones de un mismo LLM real respecto a un RAG semántico con idénticos cuatro slots, idéntico retrieval, cronología, prompts y límites? ¿Evita descartar sistemáticamente información relevante cuando CAMBIA el patrón de consultas?

**Única prueba E11-B:** si no gana frente a la **mejor** política práctica en escenarios predictivos, o empeora claramente ante la inversión de demanda, cerrar E11: mantener RAG simple y continuar E12. No extender E11 a E11-C ni cambiar las historias después de ver confirmación.

## Sistemas y límites iguales

Todos usan OpenAI Responses API **\`gpt-5.6-luna\`**, \`reasoning.effort=none\`, **\`max_output_tokens=80\`**, embeddings reales **\`text-embedding-3-small\`**, una instrucción idéntica de que la última versión prevalece, búsquedas por embeddings + solapamiento léxico con orden cronológico y máximo de **4 documentos** y **260 tokens estimados de evidencia**. Solo se diferencia \`consider(memory,event)\`:

1. \`rag_fifo\` — retiene los cuatro episodios más recientes.
2. \`rag_reservoir\` — muestreo uniforme de cuatro episodios mediante algoritmo reservoir.
3. \`rag_fixed_priority\` — prioriza urgencia, pendiente y consultas previas con coeficientes heurísticos de E11-A, más recencia (baseline práctico fuerte).
4. \`rag_learned\` — **coeficientes congelados** de las cinco réplicas independientes de E11-A: \`observatory/e11_learned_relevance.py::training_models()\`, verificando el SHA de pesos antes de confirmar. No se usa el oráculo de distribución para seleccionar memorias reales.

Todos tienen **exactamente cuatro slots físicos**; no conservan índices invisibles, textos descartados, ni recuperan datos del historial íntegro al preguntar. El simulador mantiene la verdad fuera de los actores exclusivamente para crear y puntuar preguntas.

Los eventos son frases en español sobre asuntos con identificador nuevo, atributos explícitos de prioridad, estado, procedencia y precedentes de consulta; el selector recibe estos atributos como metadatos **observables**. El LLM recibe solamente texto de memorias realmente retenidas, la pregunta común con dos opciones y la regla de abstención. No recibe atributos del target, future keys, probabilidades latentes ni etiquetas esperadas. **Limitación explícita:** atributos estructurados artificiales y patrones lingüísticos generados; aún no aprende a extraer relevancia del lenguaje natural sin metadatos.

## Historias contrafactuales, actualización y cambio de necesidades

Cada mundo contiene 16 asuntos diferentes, cada uno con un valor vigente, identificador nuevo y metadata. El entorno genera la clave de consulta **después** de las 16 decisiones de conservación. El selector no puede verla al escribir.

- Dos historias contrafactuales con **las mismas 16 frases y atributos** salvo el valor vigente del asunto que el mundo consultará; pregunta idéntica entre historias y opciones FARO/NUBE (o par de códigos aleatorios).
- Tras primera consulta de decisión, el asunto consultado cambia explícitamente al valor contrario. Se procesa esa actualización y **15 asuntos distractores nuevos** (con atributos nuevos), sin que la segunda pregunta sea visible a la política. La memoria sigue estrictamente limitada a cuatro entradas y las versiones del mismo asunto solo pueden reemplazarse en el slot retenido si la política decide conservar la actualización.
- Tras un reinicio simulado del historial reciente, misma pregunta, **segunda decisión** usando solo el contexto RAG. El par pasa si las cuatro decisiones son correctas (dos histories × antes/después). No se computan como cuatro unidades independientes.
- La **inversión de demanda** cambia el vínculo oculto entre atributos y probabilidad de consulta (coeficientes latentes cambiados de signo). **El modelo aprendido queda congelado**, sin información sobre el cambio. Se registra perjuicio, especialmente frente al reservorio; no se cambia el selector tras observarlo.
- Se evalúa también en una **demanda uniforme** con las mismas memorias pero **sin llamar al LLM**: tasa de retención de la clave relevante. Techo teórico 4/16=25 %, ningún modelo debería tener ventaja estable sin señal.

## Datos nuevos, independencia y sellado

**DEV**: semillas \`4100000+j\`, j 0..3, cuatro pares predictivos. Solo tests de integración, costo, límites y procedencia; no optimizar las políticas para ganar.
**CONFIRM**: semillas \`4200000 + replica*1000 + j\`, replica 0..4, j 0..2, **15 pares predictivos y 15 pares con inversión de demanda**, evaluados con el LLM real: **30 pares ×4 decisiones ×4 brazos = 480 llamadas**. La consulta se genera con un stream reproducible. El mismo episodio (historia, distractores, opciones, target) es idéntico entre brazos. Las réplicas de entrenamiento son cinco unidades principales, NO las 30 historias ni los 120 mensajes.
**Uniforme**: 5×400 nuevos episodios \`4400000 + replica*1000+j\`, 2.000 por evaluación, **sólo retención sin LLM** y sin feedback ni cambio de pesos. Identidades disjuntas de train E11-A, DEV y CONFIRM.
**Negativos sin hecho**: 4 preguntas ×4 brazos con 4 recuerdos de otros asuntos, todos deben responder \`ABSTENER\`; sin el caso correcto ningún sistema debe inferir la etiqueta a partir de las dos opciones.

No reutilizar E10 \`1300000,1800000,2200000+\`, E11-A \`3300000+\` ni holdout histórico F9 \`900000+\`. No abrir confirmación hasta pasar todos los tests, comprobar la API real, congelar pesos y registrar SHA. Si se abre, declarar \`4200000+\` consumido; una incidencia posterior no autoriza repetirlo como nuevo confirmatorio.

## Métricas preregistradas y gate

**Primaria predictiva:** \`four_of_four_pair_success\` de 15 mundos/parejas por brazo, y contraste emparejado de \`rag_learned\` vs **cada** baseline (FIFO/reservorio/prioridad fija). Para afirmar avance:
- ≥ **10 puntos porcentuales absolutos** sobre el mejor baseline práctico en los 15 pares;
- beneficio frente al baseline más fuerte en ≥4/5 réplicas;
- bootstrap por las **cinco réplicas** de la diferencia con límite inferior >0 (10.000 remuestreos, semilla 1811);
- no más falsos recuerdos que el baseline en los cuatro controles negativos.

**Seguridad de cambio de demanda (condición necesaria):** en los 15 pares invertidos, aprendido no debe estar **más de 5 pp por debajo** del reservorio. Esta condición estricta evita llamar éxito a una política que falla sistemáticamente tras un cambio radical, aunque sea buena en estacionariedad. Si falla, **no avanzar E11**.

Otras métricas: decisiones individuales, antes y después de la actualización, tasa de pares que invierten correctamente, memoria efectiva 4/4, presencia del recuerdo vigente en contexto, coste de tokens realmente facturado, bytes y textos de recuerdos admitidos, huella de embeddings, controles uniformes de 4/16, resultados por réplica.

**Limitaciones y no-claims:** escenario controlado con metadatos explícitos; el régimen está generado con la misma familia de relevancia log-lineal que aprende E11-A. Una victoria en el régimen predictivo no demuestra comprensión autónoma de relevancia en conversaciones humanas ni entrenamiento de un LLM. No extrapolar un IC de cinco réplicas a la distribución de usuarios o proyectos reales. Una desventaja bajo inversión es informativa y cierra la línea de despliegue.

## Ejecución
1. Implementar únicamente un prototipo de RAG semántico de cuatro slots con políticas inyectables, **sin modificar \`src/infinito3/\`**.
2. Tests para igualdad de prompts, mismo presupuesto/capacidad, ninguna clave futura en \`consider()\`, trazas de cambio de valor, aislamiento de particiones y controles.
3. DEV con modelo OpenAI real y embeddings auténticos. CONGELAR.
4. Confirmación **una sola vez** (15+15 pares) y uniforme offline, todos los resultados publicados en PR y artifact sin selección de buenas semillas.
5. Si no pasa ambas puertas, **cerrar E11 y pasar a E12** con el RAG limitado como baseline, sin crear más modelos por esta hipótesis.
