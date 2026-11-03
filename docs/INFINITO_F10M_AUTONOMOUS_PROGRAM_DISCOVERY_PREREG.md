# INFINITO F10-M — descubrimiento por experiencia de secuencias de operaciones

**PRERREGISTRO PROSPECTIVO — 2026-11-03 (Europa/Madrid).** Registrado antes de crear cualquier generador/modelo/evaluación F10-M y antes de generar TRAIN/DEV/OOD. Experimento F10-L y confirmación F10-L-C están congelados (run original 41685499939; confirm 41699416468). El hallazgo previo fue composición FUNCIONAL de un selector y XOR entrenados por recompensa, pero la cadena SELECT→READ→XOR fue programada por nosotros. **F10-M exige que el agente aprenda a elegir una secuencia de operaciones a partir de intentos propios, SIN etiquetas de programas correctos**. Los propios operadores, sus precondiciones y el catálogo de programas de longitud limitada SÍ son suministrados por el investigador.

## Hipótesis falsable y significado de «descubrimiento»

¿Puede un controlador que observa objetivos y consecuencias de sus intentos aprender qué combinación ORDENADA de primitivas congeladas utilizar, y transferirla **a una combinación de condiciones jamás entrenada**? Descubrimiento aquí significa **selección aprendida entre secuencias candidatas, incluida su longitud/orden, por recompensa de episodios ejecutados**. NO invención de operadores nuevos, planificación abierta con pasos ilimitados, ni desarrollo espontáneo de un lenguaje de programación.

El agente recibe en el estado inicial solo (i) marca SOURCE=PUBLIC/MEMORY, (ii) bandera FLIP=0/1 y (iii) un bit público (señuelo independiente cuando SOURCE=MEMORY); el contenido de F4 NO está en la observación inicial ni en los logits del selector de secuencias. Dispone de operaciones:

- **P**: cargar el bit público al registro de trabajo (si SOURCE=MEMORY, solo se ofrece un señuelo independiente; no se enmascara P para que pueda probar errores).
- **S**: pedir al modelo ORIGINAL congelado \`sel_uniform_pair\` que seleccione un slot sobre firmas/tiempos/fuentes públicos, sin ver valores.
- **R**: leer a memoria de trabajo el contenido F4 del slot seleccionado; si se ejecuta antes de S, falla, cuenta como acción y recibe penalización.
- **X**: transformar el bit de trabajo con el modelo ORIGINAL congelado \`xor_two_actions\` usando flip=1; sin bit cargado falla, cuenta como acción y recibe penalización.
- **A**: entregar el bit de trabajo como respuesta terminal, o 0 si no fue inicializado. A siempre es la terminación implícita y está disponible sin precondiciones.

Enumerar mecánicamente **TODAS** las cadenas de cero, uno, dos o tres operaciones sobre \`{P,S,R,X}\` y A terminal: \`1+4+16+64=85\` programas, incluidas secuencias inválidas, inútiles, invertidas, redundantes y alternativas válidas. **NO filtrar cadenas por la tarea, no proporcionar la respuesta ni una secuencia target, no imponer el orden SELECT→READ→XOR**. Los precondiciones constituyen semántica básica de las operaciones y son visibles a través de fallos/recompensas al probarlas. Los modelos de operadores, entradas/salidas y máximo de tres pasos han sido dados, no descubiertos.

Para eficiencia, el simulador puede cachear por mundo (a) el índice elegido por el selector ORIGINAL y (b) los resultados \`X(-1)\` y \`X(+1)\` calculados por el XOR ORIGINAL. Tests compararán la caché con llamadas reales de los modelos originales. Esa caché NO incluye la verdad, programa ganador ni las recompensas de programas no ejecutados; entrenamiento solo consulta \`environment.step\` de los DOS programas efectivamente elegidos por ejemplo.

## Datos sintéticos y generalización

Tres combinaciones TRAIN (cada una 256 mundos base × 4 contrafactuales = 1024 ejemplos/réplica):
- \`PUBLIC,FLIP=0\`: cargar el bit público y responder sin invertir.
- \`PUBLIC,FLIP=1\`: cargar bit público y transformarlo con X.
- \`MEMORY,FLIP=0\`: encontrar un informe firmado, leer valor y responder sin X.

**Combinación jamás vista en TRAIN ni DEV: \`MEMORY,FLIP=1\`**, exclusivamente en OOD, con 128 mundos base × cuatro contrafactuales = 512 ejemplos/réplica. Una política que memorizara las secuencias de las tres tareas TRAIN podría fallar al tener que seleccionar S,R,X juntos. El objetivo se evalúa respecto al último informe acreditado (no al último visto), y XOR del valor si FLIP=1. El bit público en MEMORY es **independiente** del resultado correcto. En las cuatro variantes del mundo \`U_before,U_after,S_before,S_after\`, solo S_after cambia la respuesta de MEMORY; en PUBLIC las cuatro conservan respuesta.

Nuevos namespaces:
- TRAIN: \`25100000 + 10000*r + 1000*t + j\`, t=0,1,2 para los tres contextos TRAIN, j=0..255; 4 variantes por mundo.
- DEV: \`25200000 + 10000*r + 1000*t + j\`, j=0..63; nunca contiene MEMORY+FLIP1.
- OOD: \`25400000 + 10000*r + j\`, j=0..127; MEMORY+FLIP1 solamente.
- CONFIRM **\`25300000+\` SELLADA**, no generable ni utilizada por F10-M.

TRAIN/DEV contienen 5 o 7 informes (2/3 firmados), OOD 9 u 11 (4/5 firmados) y cuatro procedencias: iid, fuente0 asociada con firmado, fuente1 asociada y alternancia en almacenamiento. Todo se cruza para controlar atajos. Los índices de almacenamiento no informan de fecha. Valores F4 ocultos hasta operación R. Cuatro réplicas independientes, control de seeds y hashes de identidad de cada caso (no confundir SHA únicos con contenido semántico independiente). El generador no tendrá acceso a los conjuntos de producto Bemula/E10 ni a particiones previas F10.

## Condiciones comparadas, presupuesto y arquitectura

Cuatro controladores que eligen uno de los MISMOS 85 programas. Todos reciben el MISMO contexto (source,flip,public_bit), se entrenan contra las mismas historias, mismo costo, **2 programas realmente ejecutados por ejemplo** y nunca disponen de un oracle de la secuencia correcta. Comparación honesta de diferencias de arquitectura/estimador (NO afirmación de igual capacidad por parámetros):

- **A \`global_bandit\`**: pesos libres para 85 programas, sin condición por contexto; aprende preferencia global vía REINFORCE, control sin adaptación al objetivo.
- **B \`context_mlp\`**: MLP sobre (source,flip,public_bit) que genera 85 logits; puede memorizar tareas vistas, no se estructura para reutilizar operaciones entre contextos.
- **C \`factorized\`**: modelo lineal de puntuación de programa sobre conteos, posiciones y bigramas genéricos de operaciones, con efectos aditivos de SOURCE y FLIP, sin interacción SOURCE×FLIP. Aprende pesos SOLO a partir de recompensas. Ese sesgo factoriza las dependencias entre fuente y transformación y favorece generalizar a MEMORY+FLIP1; no oculta etiquetas de programa.
- **D \`paired_factorized\`**: MISMA arquitectura que C, pero usa la diferencia de recompensas entre las dos secuencias verdaderamente ejecutadas sobre el mismo estado. Solo cambia el estimador de crédito, NO los programas candidatos ni el presupuesto.

La exploración muestrea según distribución conductual real \`μ=(1−ε)softmax(logits)+ε/85\`, \`ε=0.20\`. A/B/C aprenden usando score function **log μ del comportamiento realmente ejecutado**, baseline EMA de retorno pasado \`0.90\`; D utiliza preferencia de par \`−0.5*(R_a−R_b)*(log μ(a)−log μ(b))\`, dos acciones ejecutadas, sin oráculo. A/B/C llevan entropía \`0.002\` sobre distribución política; D idéntica entropía. Adam lr \`0.01\`, clip gradiente 1.0, torch CPU determinista, seeds controlador \`255000+r\`, orden de mundos \`256000+100r+epoch\`, dos muestras por estado. **16 épocas × 3072 ejemplos TRAIN × 2 programas ejecutados = 98.304 episodios reales por brazo/réplica**, \`768\` actualizaciones de batch 64. Cada programa puede tener 0–3 acciones ejecutadas y A terminal; coste operativo \`0.01\` por acción más \`0.10\` por acción inválida. Retorno observado \`+1\` respuesta correcta o \`−1\` incorrecta menos costes. En TRAIN todos los operadores congelados; solo se optimiza controlador. **NINGÚN replay de OOD, ningún gold program, ninguna recompensa de programas no ejecutados**.

El control de resultado usa ejecución sin exploración: argmax de los 85 programas. Se registran secuencias elegidas por contexto, proporción de P/S/R/X, número de fallos de precondición, precisión, recompensa con coste y costes de exploración.

## Auditoría y criterios de aceptación prospectivos

Congelar TODOS los 16 checkpoints de controladores y 8 checkpoints originales de habilidades usados por las cuatro réplicas antes de **generar DEV/OOD**. Verificar sus SHA-256 frente al manifiesto \`f10l-g0.json\` del run ORIGINAL \`41685499939\`. El auditor externo recibe predicciones congeladas y manifiestos; **no importa PyTorch, código de entrenamiento ni generador**. Bootstrap de 2000 resamples por mundo base (cuatro variantes correlacionadas juntas), semillas \`257000+100*r+offset\`. No ajustar al ver DEV, ni repetir experimentos para pasar.

Controles de inferencia: \`F4_shuffle\` rota valores entre slots válidos sin modificar selección ni flags; \`X_disabled\` hace X pasar bit sin transformación; \`P_public_bit_randomized\` cambia bit público en tareas MEMORY para comprobar que no se resuelve por señuelo; programa aleatorio uniforme (aplicación sin recompensa nueva); A global sin contexto; B MLP; D paired; regla simbólica del objetivo como techo privilegiado exclusivamente diagnóstico. No usar prueba de la regla simbólica como actualización.

- **G0 ACQUISITION:** C consigue precisión DEV **>=80% en los TRES contextos vistos** en >=3/4 réplicas. Si falla, no atribuir automáticamente su fallo OOD a composición y reportar G0 separado.
- **G1 NOVEL-PROGRAM-TRANSFER:** C consigue precisión OOD MEMORY+FLIP1 **>=80%**, IC95 por mundo límite inferior **>70%**, y >=70% en todos los estratos N, fuente, variante, en >=3/4 réplicas. Esto exige elegir una secuencia sin entrenamiento de esa combinación, no inventar primitivas.
- **G2 CAUSAL-MEMORY-AND-TRANSFORM:** C precisión \`normal−F4_shuffle\` y \`normal−X_disabled\` cada una **>=+20 puntos** y con IC95 por mundo **límite inferior >0**, en >=3/4 réplicas.
- **G3 CONTEXTUAL-ADAPTATION:** C−A en precisión OOD >=+10 puntos e IC95 inferior >0 en >=3/4 réplicas; en DEV C debe ejecutar al menos 2 programas distintos como mayoritarios entre los tres contextos (evitar éxito de una secuencia fija).
- **G4 PAIRED-CREDIT GAIN**: D−C en OOD >=+5 puntos e IC95 inferior>0 en >=3/4; **reportar, NO requisito** de la hipótesis primaria.
- **SCIENCE PASS = G0 ∧ G1 ∧ G2 ∧ G3** (todos >=3/4). Si no, estado \`F10M_PROGRAM_DISCOVERY_FAIL\`, aunque algún subgate pase. Si sí, \`F10M_PROGRAM_DISCOVERY_PASS\`. Estado técnico independiente, y jamás considerar éxito solo porque el workflow marcó SUCCESS.

**Advertencias:** El controlador selecciona una cadena COMPLETA al inicio del episodio, no replantea acciones tras observar valores intermedios; no afirmar planificación reactiva. Programas de máximo tres pasos sobre catálogo exhaustivo de 85 expresiones y operación A terminal dada. El éxito demostraría transferencia entre *operaciones disponibles* y *política de elección de cadenas* entrenada mediante consecuencias, no invención de nuevos operadores ni capacidad cognitiva general.

**Integridad:** Rama de investigación separada de \`master\` y \`src/infinito3\`; modelos F10-L nunca modificados; no usar F10-L-C CONFIRM ni Bemula para entrenamiento/selección del controlador. La confirmación F10-M \`25300000+\` permanece sellada. Publicar el resultado aunque FAIL; no retocar ni reutilizar los mundos OOD para cambiar criterios.
