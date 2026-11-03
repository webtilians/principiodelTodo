# INFINITO F10-M — el agente descubre cadenas útiles para las tareas vistas, pero NO las recompone en una combinación nueva

**3 de noviembre de 2026. Estado técnico: \`F10M_TECHNICAL_COMPLETE\`. Estado científico original: \`F10M_PROGRAM_DISCOVERY_FAIL\`.** Las cuatro réplicas, todos los 16 controladores, los modelos originales F10-L, los benchmarks DEV/OOD y el auditor independiente finalizaron con SUCCESS técnico. **Los criterios científicos G1/G2/G3/G4 fallaron** conforme al contrato fijado ANTES de implementar F10-M. No se retocó ningún hiperparámetro, umbral, semilla, recompensa ni versión tras mirar DEV/OOD.

## Procedencia y controles de integridad

- **Prerregistro prospectivo**: [INFINITO_F10M_AUTONOMOUS_PROGRAM_DISCOVERY_PREREG.md](INFINITO_F10M_AUTONOMOUS_PROGRAM_DISCOVERY_PREREG.md), commit **\`7f21d7393ad85ab32dcacd991e6df9d56fadcd78\`**, publicado **antes de crear el generador, modelo, entrenamiento o primeros datos F10-M**.
- **Ejecución original**: [GitHub Actions #41707618978](https://github.com/webtilians/principiodelTodo/actions/runs/41707618978). 4/4 entrenamientos y evaluaciones, y auditoría externa de datos en job distinto, todos SUCCESS técnico. Artefactos: \`f10m-original-0..3\` contienen 16 checkpoints originales F10-M SHA-256, curvas completas de TRAIN, programa seleccionado para cada caso y controles de ablación; \`f10m-original-independent-fixed-audit\` contiene \`f10m-fixed-science.json\` con gates, CIs, programas y métricas.
- Operadores inalterados: \`sel_uniform_pair\` + \`xor_two_actions\`, usando los checkpoints originales del [run F10-L #41685499939](https://github.com/webtilians/principiodelTodo/actions/runs/41685499939), con **verificación de los cinco SHA originales por réplica frente a su manifiesto pre-OOD**. Ningún entrenamiento/gradiente de selector o XOR; solo se entrenaron los controladores que eligen el programa.
- Semillas nuevas: TRAIN \`25100000+\` (3 contextos × 256 mundos × cuatro variantes por réplica), DEV \`25200000+\` (3×64×4), OOD \`25400000+\` (128×4 por réplica, exclusivamente MEMORY+FLIP1), **CONFIRM \`25300000+\` sigue SELLADA**. Ningún caso OOD entrenado. 85 programas candidatas exhaustivos de longitud 0,1,2,3 sobre operaciones P, S, R, X, seguidos de ANSWER, incluidos los de orden incorrecto.
- Cuatro condiciones: A \`global_bandit\` sin contexto, B \`context_mlp\`, C \`factorized\` aditivo por SOURCE y FLIP sin interacción source×flip, D \`paired_factorized\` misma arquitectura C con diferencias de recompensas de dos programas ejecutados. **Cada brazo/réplica ejecutó exactamente 98.304 episodios de entrenamiento y 768 actualizaciones**; cero etiquetas de programa correcto, cero rewards de programas no ejecutados. El modelo observa SOURCE/FLIP/bit público, no el contenido F4; la operación READ lee una única posición seleccionada por el modelo congelado. Código y producto en \`master\` intactos.
- Todos los 16 pesos SHA originales controlados; auditor independiente comprobó **17.408 hashes de identidad de casos** TRAIN/DEV/OOD (incluyen seed; no son 17.408 patrones de entrada semánticamente independientes), **20.480 filas de DEV+OOD** de los 4 brazos y 4 réplicas. Bootstrap 2000 por MUNDO BASE, manteniendo las cuatro variantes contrafactuales juntas. Todas las puertas evaluadas según el prerregistro; técnico SUCCESS no equivale a ciencia PASS.

## La hipótesis y el hecho que falla

Una biblioteca de 85 secuencias incluye la solución que F10-L anterior ejecutó porque nosotros la programamos: **S → R → X → ANSWER**. En F10-M el agente debe inferir qué secuencia ejecutar a partir de recompensas de intentos previos.

Para entrenar ve **PUBLIC+flip0**, **PUBLIC+flip1** y **MEMORY+flip0**. Nunca ve **MEMORY+flip1**. Los operadores F10-L son suficientemente capaces (confirmados al 97,02% por el estudio anterior) y permanecen congelados. La pregunta científica era si el controlador podía recombinar *obtener información de memoria* (aprendido en MEMORY0) con *invertir un bit* (aprendido en PUBLIC1) cuando ambos requisitos aparecen juntos por primera vez.

### Adquisición en tareas conocidas (DEV, C factorized)

| Réplica | PUBLIC flip0 | PUBLIC flip1 | MEMORY flip0 | G0 |
|---|---:|---:|---:|---|
| 0 | 100,00% | 51,56% | 100,00% | FAIL |
| 1 | 100,00% | 100,00% | 100,00% | PASS |
| 2 | 100,00% | 100,00% | 89,84% | PASS |
| 3 | 100,00% | 100,00% | 100,00% | PASS |

**G0 PASS 3/4**: el agente aprendió programas funcionales para todos los objetivos conocidos en tres réplicas. No debería confundirse este positivo con la capacidad de recombinar sus partes en un objetivo inédito.

### Combinación nunca entrenada (OOD): MEMORY+flip1

| Réplica | A global | B MLP contextual | C aditivo composicional | D aditivo con crédito emparejado | Programa elegido por C |
|---|---:|---:|---:|---:|---|
| 0 | 0,00% | 0,00% | **0,00%** | 0,00% | \`SSR\` |
| 1 | 0,00% | 25,20% | **0,00%** | 0,00% | \`PSR\` |
| 2 | 12,50% | 12,50% | **44,92%** | 12,50% | \`PXS\` |
| 3 | 0,00% | 47,27% | **0,00%** | 0,00% | \`SSR\` |

A/B/C/D comparten EXACTAMENTE el mismo entorno. C no alcanza el umbral de 80% en ninguna de las cuatro réplicas; los intervalos agrupados de C en réplicas 0/1/3 son **[0,0]%**, y en r2 **[40,63;48,83]%**. **G1 TRANSFER FAIL 0/4**.

## Diagnóstico post hoc: por qué 0% no es un fallo del modelo de memoria

Los resultados archivaron la secuencia concreta elegida sin interpretarla previamente como gate. Análisis exploratorio realizado **DESPUÉS** de aplicar los gates congelados:

- R0 y R3 eligen **\`SSR\`** (SELECT, SELECT, READ), con último valor firmado correcto, pero **sin inversión XOR**. R1 elige **\`PSR\`** (PUBLIC, SELECT, READ): P copia un señuelo irrelevante y después R lo sobrescribe con el valor F4 correcto; también **sin XOR**. Para MEMORY+FLIP1, si el selector acierta, la respuesta que producen estas cadenas es EXACTAMENTE la contraria de la verdad requerida. Por eso su precisión es **0%** y no ~50%; los operadores congelados realizan la tarea parcial correcta, pero la secuencia elegida es incorrecta para el nuevo objetivo.
- R2 elige **\`PXS\`** (PUBLIC, XOR, SELECT): transforma el **bit público señuelo**, no ejecuta READ y el SELECT al final no cambia el bit de trabajo. En MEMORY ese bit es independiente de F4. Su 44,92% es compatible con una estrategia sin información útil, no con composición.
- En tareas vistas, C eligió P0 \`PPP\` en las cuatro réplicas (redundante); en PUBLIC1 R1/R3 \`PXR\` (la última R es inválida después de P/X y no aporta valor), R2 \`PXS\`, y R0 \`SSS\` (no aprendió PUBLIC1; 51,56% por la distribución). En MEMORY0 eligió \`SSR\` (R0/R2/R3) o \`PSR\` (R1). Los pequeños costes de operación no impidieron muchas operaciones redundantes: **acertar en DEV no implica extraer una subpolítica mínima/transferible**.
- El control causal F4 shuffle produce una curiosidad diagnóstica: en R0/R1/R3, barajar valores F4 aumenta la precisión con respecto a 0% porque rompe la copia fiel del valor que debería invertirse. Los contrastes C normal−shuffle son negativos (~−49 a −53pp), no muestran uso de F4 para el objetivo correcto. En R2 el resultado no depende de F4 porque no se lee.
- Desactivar XOR no empeora a R0/R1/R3 (nunca lo utilizan) y no proporciona una mejora causal legítima en R2 (aplica XOR a un señuelo público). **G2 CAUSALIDAD FAIL 0/4.**
- La ventaja C vs A contextual no superó el gate excepto en R2, donde C−A +32,4pp pero sigue cerca del azar. **G3 CONTEXTUAL ADAPTATION FAIL 1/4** según el requisito conjunto de mejora y variedad de programas. **G4 PAIR CREDIT FAIL 0/4**: el nuevo estimador por programas no demuestra superioridad.

## Prerregistro G0–G4, sin rescatar resultados negativos

| Gate | Umbral original | Resultado |
|---|---|---|
| G0 Vistas | C >=80% en los 3 DEV, >=3/4 réplicas | **PASS 3/4** |
| G1 Nueva combinación | C >=80% OOD, IC inferior>70%, todos los estratos>=70%, >=3/4 | **FAIL 0/4** |
| G2 Causalidad | F4 shuffle y X_disabled reducen >=20pp e IC95 inferior>0, >=3/4 | **FAIL 0/4** |
| G3 Contextual > global | C−A>=10pp IC inferior>0, 2+ programas predominantes vistos, >=3/4 | **FAIL 1/4** |
| G4 Paired > C | D−C>=5pp IC inferior>0, >=3/4; diagnóstico | **FAIL 0/4** |

**GLOBAL \`F10M_PROGRAM_DISCOVERY_FAIL\`**, a pesar de que entrenamiento y auditoría técnica terminaron correctamente.

## Qué demuestra y qué NO

**Sí demuestra:** cuando la elección de programas pasa a ser una política aprendida por recompensas de intentos, el agente puede encontrar cadenas que funcionan en situaciones de entrenamiento. El procedimiento de experimentación con cadenas variadas es implementable y auditable, y el agente no recibió la secuencia correcta.

**No demuestra:** que haya aprendido módulos composicionales transferibles, que el programa nuevo se haya descubierto, que exista adaptación de planificación al valor observado en tiempo real, que el estimador paired sea mejor o que la arquitectura tenga capacidades cognitivas generales. Que F10-L alcanzase 97,02% con una cadena impuesta y F10-M fracase al seleccionar una no constituye contradicción: son **problemas de aprendizaje distintos**.

El controlador F10-M decide un programa completo al inicio; no puede reconsiderar una operación al descubrir un dato después. La biblioteca y las precondiciones las proporcionamos nosotros, y el espacio máximo de tres operaciones es pequeño. Los resultados muestran que **la agregación aditiva de características sintácticas de secuencias, sin enseñar una política que componga subhabilidades de forma coherente, no basta para transferir a una tarea que requiere unir MEMORY y XOR**.

## Próximo experimento potencial: F10-N (NO EJECUTAR AQUÍ)

Congelar F10-M y diseñar prospectivamente un agente con **subpolíticas de adquisición y transformación aprendidas de consecuencias, con reutilización jerárquica de sus trazas**; comparar contra controlador categórico F10-M SIN retocarlo. Cada segmento propuesto por el propio agente debe aprender precondiciones y efectos observados, no recibir cadenas correctas prefijadas. Probar qué ocurre si el contexto requiere cambiar de objetivo a mitad de ejecución y elegir tras READ, con controles de recompensa, costes de acciones y presupuestos idénticos. Basar el nuevo TEST en mundos frescos y dejar **F10-M CONFIRM 25300000+ sin abrir** tras el FAIL.

**STOP F10-M**. No tuning posterior, no volver a ver OOD para seleccionar el mejor brazo como hipótesis nueva, ni fusionar con \`master\`. El resultado negativo permanece como contribución diagnóstica.
