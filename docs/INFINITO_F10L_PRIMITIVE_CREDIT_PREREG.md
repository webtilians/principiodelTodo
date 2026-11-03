# INFINITO F10-L — atribución de recompensa y aprendizaje de dos primitivas antes de composición

**Prerregistro prospectivo, 2026-11-03**, registrado ANTES de escribir el generador/modelos F10-L o abrir cualquier TRAIN/DEV/OOD. Basado en F10-K: interpretar un único bit llegó a 100%, pero XOR se estancó en 75% y selección de la última versión en 44–71%. La composición zero-shot no fue aprendida, y su evaluación no probó si *habilidades ya adquiridas* se componen porque G0 falló. F10-L se centra en aprender las dos habilidades por separado, usando solamente consecuencias de acciones ejecutadas.

## Pregunta e hipótesis

**¿Puede una atribución de crédito más localizada, comparando resultados de acciones ejecutadas en el mismo estado, mejorar el aprendizaje de XOR y la selección de versiones respecto a REINFORCE?** Y únicamente si ambas habilidades pasan un control positivo preregistrado, ¿pueden combinarse **sin un solo paso de entrenamiento en la composición**?

En un XOR determinista de dos acciones, comparar las recompensas de ambas equivale informacionalmente a descubrir cuál es correcta. **No confundir el formato de recompensa con ausencia de supervisión:** es feedback experiencial directo que en este entorno binario identifica la respuesta. En selección, la recompensa de escoger un slot identifica si ese slot concreto es válido; no se pasa el índice ganador al entrenamiento, pero episodios exitosos informan indirectamente de él. No hay etiquetas como targets de pérdida RL, solo recompensas posteriores a acciones realmente ejecutadas. El objetivo es atribución de crédito y reutilización, **no inventar operaciones cognitivas inexistentes**.

## Entorno, dos habilidades y una composición sellada

- **XOR**: observación (bit b, flip f), ambas entradas como -1/+1; acción binaria a; recompensa +1 si a=(b XOR f), −1 si no. P y flip balanceados perfectamente en bloques de cuatro. Actor MLP tanh (2→16→1), misma arquitectura/semilla entre brazos XOR.
- **SELECT**: N informes con firma, timestamp único y fuente, posición de almacenamiento permutada; ningún valor de registro visible al selector. Acción: elegir un slot. El simulador intenta operar con el registro seleccionado y devuelve +1 solo si es la versión más reciente **firmada** de la clave, −1 si no; no devuelve ni el índice correcto ni verdad de otros slots. Selector con puntuación lineal de (firma, timestamp, firma×timestamp, fuente) y softmax, misma arquitectura/semilla entre condiciones.
- **COMPOSE** (NO ENTRENADA): elegir slot con selector SEL congelado, leer **solo el valor de ese slot** y pasar (bit, flip=1) al XOR congelado, producir acción binaria; comprobar si coincide con valor del último firmado XOR 1. Ni el selector ni el procesador reciben gradientes de esta evaluación. Las reglas de lectura/conexión las pone el investigador; no se está poniendo a prueba la invención autónoma de interfaces.

SELECT TRAIN/DEV N=5 o 7, 2–3 firmados, una actualización de valor contradictorio y al menos un unsigned posterior al último firmado. OOD COMPOSE N=9 o 11, 4–5 firmados, más versiones y 4 fuentes: iid, fuente0 correlacionada con firmado, fuente1 correlacionada y fuente alternada en almacenamiento. Los valores binarios del registro son aleatorios para el selector, y el último firmado puede ocupar distintas posiciones físicas. Cuatro variantes contrafactuales por mundo base: U_before, U_after, S_before, S_after cambian firma/orden temporal de una actualización contradictoria X, con al menos un unsigned posterior incluso a X. Para COMPOSE, solo S_after cambia el valor acreditado con respecto a las otras tres. Se usan valores contrarios entre la versión previa y X, sin alterar los valores en contrafactuales. No usar fuente para filtrar verdad durante TRAIN.

## Brazos y presupuesto idéntico de experiencias

4 réplicas r∈{0,1,2,3}. Cada modelo: 8 épocas × 4096 estados × **2 acciones verdaderamente ejecutadas/estado** = **65.536 recompensas observadas** y 512 pasos Adam con batch 64 por réplica. Mismo orden de casos, pesos iniciales, Adam lr 0.003, entropía 0.002 para muestreo on-policy, clipping 1.0 y baseline EMA 0.9. Para las distribuciones uniformes el bonus de entropía se mantiene 0.002 sobre la distribución *aprendida*, no sobre la conducta uniforme, ya que esa última no depende del modelo. Sin early stopping ni ajuste posterior.

- **X-A \`xor_onpolicy\`**: dos acciones Bernoulli on-policy, REINFORCE con baseline EMA basada exclusivamente en recompensas anteriores.
- **X-B \`xor_two_actions\`**: ejecuta deliberadamente una acción 0 y otra 1 en el mismo estado; el valor esperado de la política se ajusta con la diferencia de recompensas observadas **\`-(R_1−R_0) × p(action=1)\`**. No se computa resultado de acciones no ejecutadas. **El feedback de dos acciones en un problema binario determinista es un equivalente funcional de la etiqueta correcta:** describirlo con transparencia. No es una demostración de aprendizaje sin maestro en sentido fuerte.
- **S-A \`sel_onpolicy\`**: dos elecciones de slot on-policy; REINFORCE con baseline EMA.
- **S-B \`sel_uniform_credit\`**: dos slots diferentes realmente ejecutados, muestreados uniformemente SIN reemplazo. Se actualiza la política con REINFORCE de comportamiento off-policy (log probabilidad de la política para los dos índices, ponderada por recompensa menos baseline EMA). Se reconoce explícitamente que este estimador es sesgado para el objetivo on-policy; es un control de exploración uniforme, no una comparación algorítmica estrictamente insesgada.
- **S-C \`sel_uniform_pair\`**: EXACTAMENTE las mismas dos acciones uniformes sin reemplazo que S-B (seed de acciones exógenas compartida), pero aprende con la preferencia \`−(R_a−R_b) × (logπ(a)−logπ(b))/2\`. No es REINFORCE on-policy puro: es un **objetivo de ranking a partir de retornos experienciales**. El contraste S-C−S-B aísla el mecanismo de atribución condicionado a la misma exploración.
- **Oracle** solo para diagnóstico, *no* se entrena otro actor privilegiado. F10-K ya mostró que arquitectura supervisada puede alcanzar 100%; el selector simbólico último firmado y XOR simbólico se reportan como 100% de referencia.

Cada brazo siempre ejecuta 2 episodios/estado; igual presupuesto de interacciones, estados, semillas y entrenamientos. B/C en selección prueban conducta *uniforme* en lugar de la política; A vs C compara una intervención conjunta, no se le atribuye exclusivamente al estimador de crédito.

## Seeds, splits y bloqueo OOD

TRAIN: XOR \`23100000+10000r+j\` 4096 estados/réplica, agrupados en 1024 bloques de 4; SELECT \`23105000+10000r+j\` 4096 casos = 1024 mundos × 4 contrafactuales. DEV: XOR \`23200000+10000r+j\` 512 casos (128 bloques), SELECT \`23205000+10000r+j\` 512 casos (128 mundos ×4). OOD COMPOSE: **\`23400000+10000r+j\`** 128 mundos ×4 = 512 casos/réplica. CONFIRM **\`23300000+\` SELLADO**, no generable. Distintas semillas y hashing de todos los casos por split y réplica. La clave caso incluye seed/variante, por lo que hashes únicos **no significan observaciones de bits independientes**.

IMPORTANTE: el workflow tiene dos fases. **Fase 1** entrena 5 modelos por réplica, congela 20 SHA y evalúa exclusivamente XOR/SELECT DEV. **Auditor independiente G0** decide abrir o no la composición: SOLO si al menos **3 de las 4 mismas réplicas** alcanzan simultáneamente en DEV \`xor_two_actions≥95%\` y \`sel_uniform_pair≥85%\` de selección correcta, y ambos IC95 bootstrap inferiores >90% XOR y >75% SELECT, podrá ejecutarse Fase 2. Si G0 falla: **STOP** y \`F10L_PRIMITIVE_COMPETENCE_FAIL\`; OOD \`23400000+\` queda SIN ABRIR. Si pasa G0, un job separado recupera SHA originales y evalúa COMPOSE sin entrenamiento ni selección de checkpoints.

## Puertas científicas preregistradas

Auditor DEV por 2000 bootstrap con unidad bloque (4 combinaciones XOR o 4 contrafactuales SELECT) y semilla \`237000+r\`. Reportar accuracies, tabla completa de XOR, distribución de slots correctos/erróneos y curvas de TRAIN por época; resultados por réplica y no solo promedio.

- **G0 adquisición**, descrita arriba, **obligatoria antes de abrir OOD**.
- **G1 XOR paired advantage**: X-B−X-A >=+5 puntos e IC95 inferior >0 en ≥3/4 réplicas. Diagnóstico separado, no requisito de composición.
- **G2 SELECT credit advantage**: S-C−S-B >=+5 puntos e IC95 inferior >0 en ≥3/4 réplicas. Diagnóstico separado.
- **G3 COMPOSICIÓN ZERO-SHOT** (solo si G0 abrió OOD): precisión OOD >=80% y límite inferior bootstrap por mundo >70%, más >=70% por número de informes, fuente y variante en ≥3/4 réplicas. 0 actualizaciones durante OOD.
- **G4 REUTILIZACIÓN CAUSAL** (solo si OOD se abre): disminución de precisión OOD >=20pp al permutar valores F4 y >=20pp al forzar flip=0, con IC95 inferior de ambos contrastes >0 por mundo, ≥3/4 réplicas. Reportar además \`oracle_slot\` solo como intervención de diagnóstico.
- **GLOBAL PASS = G0 + G3 + G4**. G1/G2 se informan independientes, incluso si su hipótesis FAIL; F10-L no se vende como «descubrimiento de operador». Si no hay OOD por fallo G0, NO inventar resultado de composición. Estados: \`F10L_PRIMITIVE_COMPETENCE_FAIL\`, \`F10L_COMPOSITION_FAIL\` o \`F10L_COMPOSITION_PASS\`. Estado técnico separado y auditable.

**Integridad:** No abrir 23300000+, no usar mundos F10-I/J/K, ni Bemula-Lite o benchmarks de producto. No modificar master ni src/infinito3. Todos los checkpoints SHA congelados antes de ver DEV; NO retocar hiperparámetros/umbrales, reentrenar ni abrir OOD si falla G0. Publicar resultados negativos. PR borrador.
