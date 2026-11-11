# INFINITO E11-A — Aprender relevancia futura sin memorizar claves

**Preregistro 2026-11-01. Código y particiones definidos ANTES de abrir las confirmaciones.**
**Relación con F4:** F4 aprendía probabilidades de consulta asociadas a **ocho identidades de evento** y adaptaba cambios de frecuencia. E11 no repetirá ese hallazgo: exigimos **generalización de un selector aprendido a claves y asuntos nunca vistos**, usando rasgos transferibles de experiencias y feedback posterior, con memoria *estrictamente limitada*.

## Pregunta falsable

¿Puede una política de escritura/expulsión aprender, a partir de **consultas futuras observadas solo después del episodio**, cuáles características de un recuerdo predicen utilidad posterior, y conservar información más útil que reglas de recencia, prioridad o reservorio aleatorio, **sin haber visto antes su identidad**?

No se enseña una lista de «tipos importantes». La política solo aprende coeficientes numéricos que relacionan rasgos observados de los mensajes con demandas posteriores. **No es un LLM, comprensión semántica ni una red neuronal autoorganizada.** E11-A valida un mecanismo causal y eficiente de *retención predictiva*. E11-B (condicional) lo conectará al RAG semántico de INFINITO 3.0 y a decisiones de un LLM real.

## Entorno común y límites

- **16 eventos nuevos** en orden secuencial por episodio, con identificadores únicos desconocidos entre episodios; cada uno contiene un bit/payload y atributos públicos genéricos, *no una etiqueta de relevancia*: «pendiente» (0/1), «urgente» (0/1), «consultado previamente» (0/1), «origen de cliente» (0/1), «origen automático» (0/1), índice temporal; más interacción derivada \`pendiente × consultado_previamente\`.
- Los rasgos son metadatos observables de un evento y nunca incluyen la identidad del futuro objetivo o el secreto de demanda. La selección está cerrada **antes** de escoger qué clave se consultará en el episodio. Las señales se muestrean independientes del orden, salvo rasgo explícito de recencia.
- **Capacidad 4 registros completos**, sin sumario comprimido, mapa persistente de claves caídas, acceso a historial completo en inferencia ni buscar la clave pedida en la conversación: \`recall\` solo examina los 4 elementos.
- Al terminar las 16 escrituras, el mundo genera **una consulta** sobre una de las 16 claves. La precisión es 1 si el recuerdo existe (el payload puede responderse), 0 en caso contrario. **No se adivina**: si no se conservó responde ABSTENER.
- El entrenamiento, fuera de la memoria del agente, recoge los rasgos de la historia **después** del episodio y qué clave fue solicitada. Actualiza un modelo log-lineal con gradiente de máxima verosimilitud de la *demanda futura*. No recibe la regla latente, pesos del generador, futura clave en la interfaz \`consider\` ni etiquetas de importancia antes de guardar. Esto es **aprendizaje con feedback demorado de consultas** (supervisión retrospectiva de utilidad), no RL end-to-end ni aprendizaje sin ningún feedback.

## Familias de demanda — tres regímenes congelados

1. **Predictivo (primario):** probabilidad de demanda asociada de manera estable a combinación de atributos observados de los eventos, con interacción de rasgos; ninguno de los controles utiliza los coeficientes latentes. Se evalúa si el aprendiz transfiere a claves y nombres nuevos.
2. **Uniforme (control negativo):** consulta uniformemente aleatoria e independiente de los rasgos. **Techo teórico de cualquier selector que retiene 4/16 claves: 25%**. Un resultado alto aquí sugiere fuga/oráculo.
3. **Inversión de distribución (stress OOD):** inversión de las relaciones latentes en consulta futura. El modelo permanece congelado. Debe reportarse aunque el resultado sea muy malo; **NO ajustar para corregir este estrés**. Un modelo aprendido puede fallar con demanda no estacionaria.

Mismo contenido, orden y consulta exactos para todas las políticas de cada episodio. La consulta se genera una sola vez tras completar todas las decisiones de memoria.

## Políticas que deben batirse

- **FIFO**: últimas 4 observaciones.
- **Reservorio uniforme**: muestra uniforme de 4 entre los 16, sin rasgos.
- **Prioridad fija (fuerte)**: combina los tres atributos fácilmente interpretables (urgente, pendiente, consultado antes) y recencia, con pesos no ajustados al generador.
- **Recencia + prioridad**: variante que pone más peso en reciente.
- **Aprendida (E11)**: clasificación log-lineal entrenada a partir de consultas históricas en episodios de TRAIN. Durante test no actualiza pesos; admite/expulsa comparando únicamente su candidato actual y los 4 slots.
- **Aprendida con targets barajados (negativo de aprendizaje)**: mismo algoritmo y mismo historial de entrenamiento, pero permutando la clave final de feedback de forma independiente de los atributos, para comprobar que la ventaja proviene de señal aprendida y no del código del selector.
- **Oráculo de distribución (techo explicativo, NO competidor práctico)**: conoce la función latente que genera la consulta antes de seleccionar, no la consulta concreta; su rendimiento debe ser mayor o igual, en expectativa. Se reporta con etiqueta inequívoca y **no recibe la clave objetivo específica**.

Todos salvo el oráculo tienen la misma interfaz de \`consider(memory,current_event)\`; **la futura clave no es un argumento** y no existe acceso al episodio entero en inferencia. La política aprendida solo almacena 7 coeficientes + 4 registros; los coeficientes son parámetros aprendidos de episodios anteriores, no claves almacenadas.

## Particiones y réplicas independientes

- **TRAIN**: 5 políticas aprendidas independientes, seeds \`3100000 + 10000*rep + i\`, i=0..2499 (2500 episodios/rep). **No reutilizar** ninguna consulta TRAIN para la confirmación ni seleccionar modelos según confirmación.
- **DEV**: seeds \`3200000 + 10000*rep + i\`, rep 0..4, i=0..159, 800 episodios. Únicamente contratos/métrica/mecanismo. **Congelar código, pesos de TRAIN y procedimiento** antes de CONFIRM. No entrenar modelos sobre DEV.
- **CONFIRM**: seeds \`3300000 + 10000*rep + i\`, rep 0..4, i=0..639, **3.200 episodios por régimen**, 5 réplicas independientes de entrenamiento y generación. Se evalúan **predictivo, uniforme e inversión**; uniform e inversión usan *el mismo stream por seed*, variando solamente la distribución de query posterior. No son 3.200 políticas independientes: las 5 semillas de entrenamiento son unidades principales de replicación.
- Los nombres/identidades tienen **prefijos disjuntos** TRAIN/DEV/CONFIRM. Ningún parámetro identifica el futuro objetivo por clave o texto. Caso con 2 eventos iguales/IDs iguales inválido.

## Métricas y gate de avance

**Primaria:** tasa de \`hit\` en CONFIRM predictivo con capacidad fija 4, para \`learned\` frente a **cada baseline práctico**: FIFO, reservorio, prioridad fija, recencia+prioridad y modelo con targets barajados. Reportar proporción agregada, resultados por cada réplica independiente, diferencias emparejadas y **bootstrap de 10.000 remuestreos por réplica** (no por 3.200 episodios) con semilla 1711.

**Regla de avance preregistrada (pragmática y no una prueba de gran potencia con 5 réplicas):** incremento absoluto ≥**5 pp** sobre el **mejor baseline práctico**, mejora positiva en **≥4 de 5** réplicas frente a ese baseline e intervalo bootstrap por réplica con límite inferior >0. Solo si se cumple autoriza E11-B integrado con RAG/LLM. Si no, reportar resultado negativo sin seguir afinando pesos.

**Control uniforme:** el intervalo de acierto de aprendido debe ser razonablemente compatible con 25%, sin ventaja aparente sistemática >5 pp frente al reservorio. **Stress inversión:** no se exige victoria; reportar pérdida de transferencia para evitar alegar generalización fuera de distribución.

**Secundarias:** pérdida logarítmica de entrenamiento, pesos de modelo congelados, capacidad máxima, complejidad/latencia por episodio, tasa de abstención, regret descriptivo frente a oráculo de distribución, tasa de fallo por señales/recencia, control de que \`target\` solo se revela al ejecutar \`feedback\`.

## Prevención de desviaciones

**Máximo un mecanismo aprendido, una confirmación**. No tocar F9, E10-A/B/C/D ni \`src/infinito3/\`. E11-A es un laboratorio que **difiere del F4 por transferir a nuevos IDs y aprender relaciones entre atributos de una experiencia** en vez de memorizar frecuencias de IDs anteriores.

Si E11-A no mejora baselines fuertes, **no iniciar E11-B** ni cambiar el generador para hacerlo ganar. Si pasa, E11-B queda como etapa aparte con prueba REAL de costo/beneficio en tareas de INFINITO (RAG semántico original vs RAG con gate aprendido, mismos embeddings, misma capacidad de recuerdos y mismo LLM). Ningún resultado E11-A deberá anunciarse como una ventaja general de IA o comprensión semántica.
