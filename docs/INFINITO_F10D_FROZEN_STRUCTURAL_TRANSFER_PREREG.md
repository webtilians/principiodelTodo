# INFINITO F10-D — Transferencia de estructura con actores completamente congelados

**PRERREGISTRO irrevocable antes de implementar generador o abrir mundos nuevos (2026-11-02).** F10-C produjo 100% DEV tanto con atención relacional como unaria reentrenada: no demuestra una regla transferible, y los cinco informes por clave podrían ser una plantilla aprendida. F10-D NO entrena actores ni cabezas READ; NO abre confirmaciones anteriores ni vuelve a ejecutar sus semillas.

## Hipótesis y restricciones

¿Las políticas F10-C originales congeladas recuperan la **última evidencia firmada** cuando la cantidad y posición de informes cambia y las fuentes dejan de tener correlaciones habituales? Se evaluarán *los mismos checkpoints originales* de F10-C (run `41447171697`) y, adicionalmente, dos F10-B originales (run `41443064754`), descargados como artifacts y validados mediante hashes de sus manifiestos. No hay nuevo parámetro, actualización de checkpoint, selección de réplicas ni optimización con datos F10-D.

Se preservan interfaz pública, 24 claves, **60 informes por mundo**, 29 características por informe, log FIFO64 de observaciones (no F4 MiniGrid original), 61 observaciones con una sola consulta al final, y regla de verdad **último informe con credencial pública =1**. Fuentes A/B no determinan veracidad. Matching por clave ya incorporado en las arquitecturas originales. Todas las observaciones se generan sin conocer resultados de política.

## Generación F10-D nueva y factorial fijo

- Cuatro réplicas `i=0..3`, **64 mundos completamente nuevos por réplica**: `15_200_000+10_000*i+j`, `j=0..63`. No se abre ningún `15_300_000+` (eventual CONFIRM) ni se entrena en mundos F10-D. Cada mundo comparte exactamente la misma historia de **60 informes** en las cuatro consultas correspondientes a `signed|stable`, `unsigned|stable`, `signed|revised`, `unsigned|revised`, con cuatro claves objetivo distintas elegidas antes de generar el historial. **256 casos por réplica, 1024 casos y 256 mundos independientes en total**.
- Los cuatro objetivos de cada mundo tienen exactamente **3, 4, 6 y 8 informes** respectivamente (uno de cada). La asignación del número de informes a los cuatro estratos rota por cuadro latino determinista `(j//4 + index_de_estrato)%4`: cada estrato recibe 16 mundos por cardinalidad en cada réplica. Los 24 informes iniciales son firmados con bits aleatorios. Los cuatro objetivos generan `(3−1)+(4−1)+(6−1)+(8−1)=17` nuevos informes; **19 nuevos informes distractores** para claves no consultadas completan los 60. En las cuatro condiciones la consulta solo llega después del informe 60.
- El programa de informes de cada objetivo respeta el estrato **sin revelar el estrato como observación**. `stable`: la última afirmación firmada reafirma el bit inicial; los distractores no firmados pueden contradecirla. `revised`: al menos una actualización firmada cambia el bit inicial y la última firma conserva el bit revisado; informes no firmados pueden contradecirla. `signed` y `unsigned` se refieren únicamente a la **credencial del último informe cronológico**. En `unsigned` ese informe necesariamente contradice el valor vigente. Para cada objetivo hay al menos dos afirmaciones de valores opuestos. La secuencia interna por clave está ordenada, pero sus posiciones globales son intercaladas aleatoriamente, sin prioridad por calidad y sin ordenar por el objetivo consultado.
- Distribución de procedencia A/B, con **cuatro regímenes** balanceados `j%4`: (i) IID fuente aleatoria; (ii) informes acreditados de fuente A y no acreditados de B; (iii) acreditados B y no acreditados A; (iv) fuente alternante con el orden global. Este es un cambio **intencionado de dominio**, no una ley de autenticación: la credencial firmada es la única regla pública de validez; se reportan resultados por modo de procedencia.
- El resto de claves no consultadas recibe informes de relleno con credencial/valor aleatorios; el patrón de las 4 claves consultadas **no se expone al agente antes de la query**. Las 4 consultas ven los mismos 60 informes y el mismo F4.
- No se excluyen casos por D, primera acción, corrección ni tipo de error. Los mundos que produzcan una forma inválida detienen la ejecución en vez de pasar un filtro de semillas.

## Grupos congelados y contrafactuales

Para cada réplica se descargan y comprueban SHA de **cinco checkpoints**:
1. `F10C_relational_read` — ranking entrenable por pares, top-k5 estructural, GRU64.
2. `F10C_unary_read_control` — atención unaria F10-B reentrenada con F10-C, GRU64.
3. `F10C_recurrent_only_control` — mismo dominio F10-C sin F4.
4. `F10B_memory_available` y 5. `F10B_recurrent_only`, checkpoints originales también congelados, solo comparadores históricos transferidos.

Para el mismo `F10C_relational_read` y **exactamente el mismo historial recurrente/consulta**, calcular `READ_clean` (F4 completa), `SKIP` (memoria=0), `READ_stale` (quitar actualizaciones firmadas posteriores a la primera para la clave), y `READ_forged` (falsificar credencial del último informe no firmado de esa clave). No cambiar eventos públicos previos ni estados H/C entre modalidades; `stale` y `forged` son ataques sintéticos fuera de distribución, no errores naturales del entorno. Repetir READ/ SKIP del **F10C_unary** como diagnóstico de dependencia causal del control, sin modificar sus pesos.

Controles algorítmicos **no aprendidos**, construidos de los mismos informes públicos: último informe observado, último firmado (el *techo determinista de la propia regla*, siempre 100% por definición), y mayoría de los cinco/ocho/… informes de la clave. Su uso es diagnóstico, **no demostrar inteligencia con un oráculo oculto**.

## Métricas y gates fijados antes de DEV

Por réplica, 4 estratos procedencia × evolución y 4 cardinalidades `k∈{3,4,6,8}`, con 16 mundos/celda del cruce estrato×k. Reportar precisión global de los 5 checkpoints, READ/SKIP relacional, ataques, controles simples, D+/D0/D−, y resultados para los 4 modos fuente. Intervalo bootstrap 95% para READ−SKIP agrupado por **mundo**, 2000 resamples, no tratar 4 consultas por mundo como mundos independientes.

Las tres hipótesis principales se juzgan **por separado**:
- **Generalización de validez (G1)**: F10C relacional >=**80%** de aciertos en cada uno de los 4 estratos agregados sobre cardinalidades **y** en cada uno de los 4 bins k agregados sobre estratos, en al menos **3 de las 4 réplicas**; además >=**70%** global por cada régimen fuente para al menos 3/4 réplicas.
- **Valor causal (G2)**: READ_clean−SKIP relacional >=**+0.10** global y límite inferior bootstrap mundo 95% >0, en al menos **3/4 réplicas**, con READ_clean > F10C GRU-only nuevo por al menos +0.05 en >=3/4. Se anotan casos dañinos D−: no esconder resultados.
- **Ventaja relacional transferida (G3)**: en >=**3/4** réplicas, relacional READ debe superar F10C unario congelado en >=**+0.05 global** y >=**+0.08 en el estrato difícil unsigned|revised**. Si los dos generalizan igual, NO atribuir a pairwise ninguna ventaja.

Estado general `F10D_STRUCTURAL_TRANSFER_PASS` únicamente con **G1, G2 y G3** y contrato de integridad; de otro modo `F10D_STRUCTURAL_TRANSFER_FAIL`, mostrando también los subgates. **DEV de transferencia sobre nuevos mundos, no confirmación independiente del F10-C**: diseño de F10-D surgió tras ver su DEV. Los nuevos resultados tampoco prueban la arquitectura F4 original, RAG, LLM o inteligencia general.

## Condiciones de STOP y vulnerabilidades esperadas

La política relacional F10-C tiene `topk(5)` fijo: con `k=8` podría descartar información necesaria; con `k=3/4` puede recibir slots de otra clave, aunque enmascarados. El unario observa los 64 slots. Eso es parte del contraste congelado y **no se corrige a posteriori**. Regímenes de firma y fuente pueden ser factores correlacionados en algunos mundos; los resultados se separan por régimen para detectar atajos. Si falla un gate, no reentrenar con estas semillas, ni abrir confirmaciones. Los artefactos de F10-B/C son **solo checkpoints**, no nuevos ejemplos de entrenamiento.
